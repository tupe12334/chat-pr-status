"""Render the active chat's GitHub pull-request states in Hermes's status bar.

A chat owns a PR when an active message contains a canonical GitHub pull URL.
The plugin extracts those URLs from the current session, keeps the most recent
four in the compact status line, and refreshes GitHub state asynchronously with
``gh``. It never blocks terminal rendering or an agent turn.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from typing import Any, Callable

_MAX_DISPLAYED_PRS = 4
_MESSAGE_SCAN_INTERVAL_SECONDS = 3.0
_GITHUB_REFRESH_INTERVAL_SECONDS = 30.0
_GITHUB_TIMEOUT_SECONDS = 5.0
_PR_URL = re.compile(
    r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/pull/(\d+)(?=$|[/?#\s).,\]])",
    re.IGNORECASE,
)
_ORIGINAL_FRAGMENTS_ATTR = "_chat_pr_status_original_fragments"
_ORIGINAL_TEXT_ATTR = "_chat_pr_status_original_text"

_cache_lock = threading.RLock()
_session_cache: dict[str, dict[str, Any]] = {}


def _extract_prs(messages: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Return unique PR identities in most-recent-first message order."""
    found: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for message in reversed(messages):
        content = str(message.get("content") or "")
        for owner, repo, number in _PR_URL.findall(content):
            identity = (owner, repo, number)
            key = (owner.lower(), repo.lower(), number)
            if key not in seen:
                seen.add(key)
                found.append(identity)
    return found


def _state_symbol(payload: dict[str, Any]) -> str:
    """Map GitHub's PR/check state to a compact, terminal-safe status glyph."""
    if str(payload.get("state") or "").upper() != "OPEN":
        return "✓" if str(payload.get("state") or "").upper() == "MERGED" else "×"
    if bool(payload.get("isDraft")):
        return "…"
    checks = payload.get("statusCheckRollup") or []
    states = {str(check.get("conclusion") or check.get("status") or "").upper() for check in checks}
    if states & {"FAILURE", "ERROR", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED"}:
        return "✗"
    if states & {"PENDING", "QUEUED", "IN_PROGRESS", "WAITING", "EXPECTED"}:
        return "…"
    if str(payload.get("reviewDecision") or "").upper() == "CHANGES_REQUESTED":
        return "!"
    return "✓"


def _fetch_state(owner: str, repo: str, number: str) -> str:
    """Read PR state through the user's authenticated gh CLI, fail quietly."""
    try:
        completed = subprocess.run(
            [
                "gh", "pr", "view", number, "--repo", f"{owner}/{repo}",
                "--json", "state,isDraft,reviewDecision,statusCheckRollup",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=_GITHUB_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            return "?"
        import json
        return _state_symbol(json.loads(completed.stdout))
    except (OSError, subprocess.SubprocessError, ValueError):
        return "?"


def _refresh_async(session_id: str, prs: list[tuple[str, str, str]]) -> None:
    """Refresh state in a daemon thread so status-bar rendering stays instant."""
    def work() -> None:
        states = {(owner.lower(), repo.lower(), number): _fetch_state(owner, repo, number) for owner, repo, number in prs}
        with _cache_lock:
            cached = _session_cache.get(session_id)
            if cached is not None:
                cached["states"] = states
                cached["refreshed_at"] = time.monotonic()
                cached["refreshing"] = False

    thread = threading.Thread(target=work, name="hermes-chat-pr-status", daemon=True)
    thread.start()


def _chat_pr_label(cli: Any) -> str:
    """Return a bounded status label for PRs mentioned in the active chat."""
    session_id = str(getattr(cli, "session_id", "") or "")
    db = getattr(cli, "_session_db", None)
    if not session_id or db is None:
        return ""
    now = time.monotonic()
    with _cache_lock:
        cached = _session_cache.setdefault(session_id, {"prs": [], "states": {}, "scanned_at": 0.0, "refreshed_at": 0.0, "refreshing": False})
        if now - float(cached["scanned_at"]) >= _MESSAGE_SCAN_INTERVAL_SECONDS:
            try:
                cached["prs"] = _extract_prs(db.get_messages(session_id, limit=500, latest=True))
            except Exception:
                cached["prs"] = []
            cached["scanned_at"] = now
        prs = list(cached["prs"])
        stale = now - float(cached["refreshed_at"]) >= _GITHUB_REFRESH_INTERVAL_SECONDS
        if prs and stale and not cached["refreshing"]:
            cached["refreshing"] = True
            _refresh_async(session_id, prs)
        states = dict(cached["states"])

    if not prs:
        return ""
    shown = prs[:_MAX_DISPLAYED_PRS]
    labels = [f"#{number}{states.get((owner.lower(), repo.lower(), number), '…')}" for owner, repo, number in shown]
    extra = len(prs) - len(shown)
    if extra:
        labels.append(f"+{extra}")
    return "PR " + " ".join(labels)


def _append_text(cli: Any, text: str, width: int | None) -> str:
    label = _chat_pr_label(cli)
    if not label:
        return text
    combined = f"{text.rstrip()} · {label}"
    if width is None:
        try:
            width = cli._get_tui_terminal_width()
        except Exception:
            width = None
    if width:
        return cli._trim_status_bar_text(combined, int(width))
    return combined


def _install_status_bar_renderer() -> None:
    try:
        from cli import HermesCLI
    except Exception:
        return
    if getattr(HermesCLI, _ORIGINAL_FRAGMENTS_ATTR, None) is not None:
        return

    original_fragments: Callable[..., list] = HermesCLI._get_status_bar_fragments
    original_text: Callable[..., str] = HermesCLI._build_status_bar_text

    def wrapped_fragments(self: Any):
        fragments = original_fragments(self)
        label = _chat_pr_label(self)
        if not fragments or not label:
            return fragments
        if fragments[-1] == ("class:status-bar", " "):
            fragments = list(fragments)
            fragments[-1:-1] = [
                ("class:status-bar-dim", " · "),
                ("class:status-bar-strong", label),
            ]
        else:
            fragments = [*fragments, ("class:status-bar-dim", " · "), ("class:status-bar-strong", label)]
        width = self._get_tui_terminal_width()
        total = sum(self._status_bar_display_width(value) for _, value in fragments)
        if total > width:
            return [("class:status-bar", self._trim_status_bar_text("".join(value for _, value in fragments), width))]
        return fragments

    def wrapped_text(self: Any, width: int | None = None) -> str:
        return _append_text(self, original_text(self, width), width)

    setattr(HermesCLI, _ORIGINAL_FRAGMENTS_ATTR, original_fragments)
    setattr(HermesCLI, _ORIGINAL_TEXT_ATTR, original_text)
    HermesCLI._get_status_bar_fragments = wrapped_fragments
    HermesCLI._build_status_bar_text = wrapped_text


def register(ctx: Any) -> None:
    """Install the compatibility renderer once during native plugin discovery."""
    del ctx
    _install_status_bar_renderer()
