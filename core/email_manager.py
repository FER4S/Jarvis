# ─────────────────────────────────────────────────────────────────────────────
#  core/email_manager.py – Email orchestration engine
#  Owns the account store + a plaintext fetch cache (data/email_cache.json,
#  same persistence pattern as core/memory.py), a background poll thread,
#  proactive-announcement claiming, OAuth state nonces, and the voice-query
#  surface (intent classification + cache lookups + Sonnet-tier reasoning)
#  that core/assistant.py's conversation loop calls into.
#
#  Exactly one EmailManager instance must exist per process - it is created
#  in JarvisAssistant.__init__ and shared with api/server.py via
#  assistant.get_email_manager(). Both initialize() and start_polling() are
#  idempotent because two different call paths race to invoke them: main.py
#  starts assistant.run() in a thread BEFORE uvicorn (and therefore the
#  server lifespan) ever runs, so either one may go first.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import re
import secrets
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Callable

# Ensure the project root is on sys.path so `import config` works when this
# file is run directly (e.g. `python core/email_manager.py`).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import anthropic
from google.auth.exceptions import RefreshError
from loguru import logger

import config
from core.email_accounts import EmailAccountStore, PROVIDER_GMAIL, PROVIDER_IMAP
from core.email_fetch import (
    FETCH_WINDOW_DAYS,
    PROVIDER_FETCHERS,
    sanitize_body_text,
    truncate_text,
)
from core.email_send import PROVIDER_SENDERS, reply_subject
from core.memory import (
    EXTRACTION_MODEL_ID,
    extract_response_text,
    is_valid_email,
    parse_json_object,
)

# ── Storage location (cache only - accounts live in email_accounts.py) ──────
_DATA_DIR: str = os.path.join(_PROJECT_ROOT, "data")
_CACHE_FILE: str = os.path.join(_DATA_DIR, "email_cache.json")
_CACHE_LOAD_ATTEMPTS: int = 3
_CACHE_LOAD_RETRY_DELAY_S: float = 0.25
_CACHE_VERSION: int = 1

# ── Models ────────────────────────────────────────────────────────────────────
# Realtime tier (intent/scope classification) - same fast model core/llm.py
# and core/memory.py's realtime calls use.
INTENT_MODEL_ID: str = "claude-haiku-4-5-20251001"
INTENT_MAX_TOKENS: int = 200
SCOPE_MAX_TOKENS: int = 60
INTENT_TIMEOUT: float = 10.0
# A targeted factual question about ONE already-fetched email ("who sent it?",
# "what did they ask for?") stays on the realtime Haiku tier for snappiness -
# the boss is waiting on a one-sentence answer, unlike the heavier batch
# reasoning (reason_over_emails/summarize_email) which stays on Sonnet.
QA_MAX_TOKENS: int = 150
QA_BODY_CHAR_CAP: int = 4000

# Content-reasoning tier (action-needed / meetings-mentioned / summarize) -
# reuses core/memory.py's designated heavier background tier. NOTE: unlike
# memory extraction, these calls happen mid-conversation with the boss
# actively waiting, so - unlike memory.py's background client - this one
# still gets max_retries=0 (see __init__). Handlers speak a filler line
# before calling reason_over_emails()/summarize_email() to cover the latency.
REASONING_MAX_TOKENS: int = 400
REASONING_TIMEOUT: float = 20.0
REASONING_BODY_CHAR_CAP: int = 2000  # per-email cap when reasoning over several at once

# ── Polling / cache tuning ────────────────────────────────────────────────────
POLL_MIN_INTERVAL_S: int = 30  # floor under config.EMAIL_POLL_INTERVAL_S
SEEN_IDS_CAP: int = 500
CACHED_EMAILS_PER_ACCOUNT: int = 25
# Composite "is this actually new" rule: unseen AND unread AND younger than
# this. Guards against seen_ids-cap eviction or SINCE-window drift producing
# a false "new mail" on an old message re-fetched near the fetch window edge.
NEW_MAIL_MAX_AGE_S: int = 86400
# An unannounced email older than this is dropped from the nudge silently -
# the poller runs under the server lifespan even while the voice pipeline is
# stopped, so without this cap a morning email could get announced at 8pm.
ANNOUNCE_MAX_AGE_S: int = 4 * 3600
OAUTH_STATE_TTL_S: float = 600.0

# On-demand deep historical search ("any email from two weeks ago?") - a live
# provider-side search, bounded so one voice request can't crawl a whole
# mailbox. Stateless by design: results never touch the poll cache, seen_ids,
# or announcement state (see get_emails_for_scope).
DEEP_SEARCH_MAX_DAYS: int = 90
DEEP_SEARCH_MAX_MESSAGES: int = 30  # per account, per search
# Default lookback for vague-but-clearly-not-recent phrasing ("a while ago",
# "not too long ago") that gives the classifier no specific number to anchor
# on. Consistent with the prompt's own "last month" -> 31 worked example;
# safely under DEEP_SEARCH_MAX_DAYS; large enough to escape the 2-day poll
# cache, which is the actual bug a null days_back silently causes.
VAGUE_TIMEFRAME_DAYS_BACK_DEFAULT: int = 30

# ── Sending ───────────────────────────────────────────────────────────────────
# A drafted email is spoken aloud in full before the boss confirms it, so the
# body must stay short enough to listen to without losing the thread.
DRAFT_MAX_TOKENS: int = 400
# Cap on how much of the email being replied to is shown to the drafting model.
DRAFT_REPLY_BODY_CAP: int = 1500
# One word ("yes"/"no"/"revise"/"unclear"); mirrors memory.CLASSIFY_MAX_TOKENS.
CONFIRM_MAX_TOKENS: int = 10
SEND_PARSE_MAX_TOKENS: int = 300
ADDRESS_PARSE_MAX_TOKENS: int = 60

# How long a "waiting for an address" request stays open for the dashboard to
# fulfil. Long enough to find the window and type; short enough that a request
# abandoned by a crashed conversation can't outlive the boss's memory of it.
# The voice handler closes it in a finally regardless.
CONTACT_REQUEST_TTL_S: float = 180.0

_VALID_INTENTS: frozenset[str] = frozenset({
    "not_email", "check_new", "check_now", "list_emails", "from_person",
    "read_email", "email_question", "summarize", "action_needed",
    "meetings_mentioned", "send_email", "dismiss", "load_more",
})

# classify_confirmation()'s three real answers plus the safe default. Only
# "yes" ever authorizes an irreversible send.
CONFIRM_YES: str = "yes"
CONFIRM_NO: str = "no"
CONFIRM_REVISE: str = "revise"
CONFIRM_UNCLEAR: str = "unclear"


def _clean_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truncate_for_prompt(text: str, limit: int) -> str:
    return truncate_text(text, limit)


def _age_seconds(date_iso: str, now: datetime) -> float:
    """Seconds between `now` and an ISO date string. Unparseable -> 0.0
    (treated as fresh rather than silently dropped by an age filter)."""
    try:
        dt = datetime.fromisoformat(date_iso)
    except (ValueError, TypeError):
        return 0.0
    return (now - dt).total_seconds()


class EmailManager:
    """
    Orchestrates connected email accounts: polling, caching, proactive-
    announcement claiming, and the voice-query surface. See module docstring
    for the single-instance/idempotent-startup contract.

    Usage:
        manager = EmailManager(on_new_mail=assistant_wake_callback)
        manager.initialize()
        manager.start_polling()
        ...
        manager.stop_polling()   # only on process/server shutdown
    """

    def __init__(
        self,
        on_new_mail: Callable[[], None] | None = None,
        api_key: str = config.ANTHROPIC_API_KEY,
    ) -> None:
        self._store = EmailAccountStore()
        self._on_new_mail = on_new_mail

        self._cache: dict = {"version": _CACHE_VERSION, "accounts": {}}
        self._cache_lock = threading.Lock()
        # Serializes _run_poll_cycle / refresh_now / delete_account against
        # each other - closes the deleted-mid-poll race and stops a live
        # "check now" from interleaving seen_ids updates with a scheduled cycle.
        self._cycle_lock = threading.Lock()
        self._oauth_lock = threading.Lock()
        # Guards the single pending "waiting for a contact's address" slot,
        # which the voice thread opens and the API thread (a typed dashboard
        # submission) fulfils. Distinct from _oauth_lock so a slow OAuth
        # callback can't block a live conversation.
        self._contact_lock = threading.Lock()
        # Guards one-time initialize() and poll-thread startup - both are
        # called from two independent races (assistant.run() and the server
        # lifespan; see module docstring).
        self._lifecycle_lock = threading.Lock()

        self._initialized = False
        self._poll_thread: threading.Thread | None = None
        self._stopping = False
        self._poll_wake = threading.Event()

        self._oauth_states: dict[str, float] = {}
        self._last_announced_context: list[dict] = []
        # At most one open contact request at a time. Safe because
        # conversations are strictly serial by construction: the dispatch loop
        # joins each jarvis-conversation thread before starting the next.
        self._contact_request: dict | None = None

        if not api_key:
            logger.warning(
                "ANTHROPIC_API_KEY is not set. Email intent/reasoning calls "
                "will fail until it is added to .env."
            )
        # Realtime tier: classify_intent/classify_scope run on the
        # conversation thread with the boss actively waiting - no SDK
        # auto-retries, fail within one timeout (mirrors core/llm.py).
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=0)
        # Reasoning tier: action-needed/meetings/summarize ALSO run
        # mid-conversation with the boss waiting - unlike core/memory.py's
        # background extraction, this is not a fire-and-forget daemon-thread
        # call, so it keeps max_retries=0 too, just a longer timeout for the
        # heavier model.
        self._reasoning_client = anthropic.Anthropic(api_key=api_key, max_retries=0)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Idempotent one-time setup: load the account store and cache."""
        with self._lifecycle_lock:
            if self._initialized:
                return
            self._store.load()
            self._load_cache()
            self._prune_cache_for_deleted_accounts()
            self._initialized = True

    def start_polling(self) -> None:
        """Idempotent: starts the background poll thread if not already running."""
        with self._lifecycle_lock:
            if self._poll_thread is not None and self._poll_thread.is_alive():
                return
            self._stopping = False
            self._poll_thread = threading.Thread(
                target=self._poll_loop, name="jarvis-email-poller", daemon=True
            )
            self._poll_thread.start()

    def stop_polling(self) -> None:
        """
        Stops the poll thread. Only the server lifespan (process shutdown)
        should call this - assistant.stop() deliberately does NOT, since the
        dashboard needs /email/summary to stay fresh even while the voice
        pipeline is stopped.
        """
        self._stopping = True
        self._poll_wake.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=5)

    def request_poll_soon(self) -> None:
        """Wake the poll loop immediately instead of waiting out the interval
        (called after an account is added, so the dashboard populates promptly)."""
        self._poll_wake.set()

    def _poll_loop(self) -> None:
        interval = max(POLL_MIN_INTERVAL_S, config.EMAIL_POLL_INTERVAL_S)
        while not self._stopping:
            try:
                self._run_poll_cycle()
            except Exception as exc:
                logger.error(f"Email poll cycle crashed unexpectedly: {exc}")
            self._poll_wake.wait(interval)
            self._poll_wake.clear()

    # ── Cache persistence (mirrors core/memory.py's pattern) ─────────────────

    def _load_cache(self) -> None:
        """
        Load the fetch cache from disk, creating it if missing. Unlike
        email_accounts.py/memory.py, the cache holds no irreplaceable data -
        it's fully rebuilt by the next poll cycle - so an unreadable cache
        just resets to empty in memory (no quarantine dance needed) and the
        next successful write naturally overwrites the bad file.
        """
        with self._cache_lock:
            if not os.path.isdir(_DATA_DIR):
                os.makedirs(_DATA_DIR, exist_ok=True)
            if not os.path.isfile(_CACHE_FILE):
                self._cache = {"version": _CACHE_VERSION, "accounts": {}}
                self._write_cache_locked()
                return

            for attempt in range(1, _CACHE_LOAD_ATTEMPTS + 1):
                try:
                    with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if not isinstance(loaded, dict) or not isinstance(loaded.get("accounts"), dict):
                        raise ValueError("unexpected cache shape")
                    self._cache = loaded
                    logger.success(f"Email cache loaded from {_CACHE_FILE}.")
                    return
                except (ValueError, OSError) as exc:
                    if attempt < _CACHE_LOAD_ATTEMPTS:
                        logger.warning(
                            f"Failed to load email cache (attempt {attempt}/"
                            f"{_CACHE_LOAD_ATTEMPTS}: {exc}) - retrying…"
                        )
                        time.sleep(_CACHE_LOAD_RETRY_DELAY_S)
                    else:
                        logger.warning(
                            f"Email cache unreadable ({exc}) - starting from an "
                            f"empty cache (it's fully rebuilt by polling)."
                        )
                        self._cache = {"version": _CACHE_VERSION, "accounts": {}}

    def _write_cache_locked(self) -> None:
        """Caller must already hold self._cache_lock."""
        os.makedirs(_DATA_DIR, exist_ok=True)
        tmp_path = _CACHE_FILE + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, _CACHE_FILE)  # atomic on both Windows and POSIX
        except OSError as exc:
            logger.error(f"Failed to save email cache: {exc}")

    def _prune_cache_for_deleted_accounts(self) -> None:
        known_ids = {a["id"] for a in self._store.list_safe()}
        with self._cache_lock:
            stale = [aid for aid in self._cache["accounts"] if aid not in known_ids]
            for aid in stale:
                del self._cache["accounts"][aid]
            if stale:
                self._write_cache_locked()

    @staticmethod
    def _empty_account_cache() -> dict:
        return {
            "last_poll": None,
            "last_error": None,
            "consecutive_failures": 0,
            "unread_count": 0,
            "uidvalidity": None,
            "baselined": False,
            "seen_ids": [],
            "unannounced_ids": [],
            "emails": [],
        }

    # ── Poll cycle ────────────────────────────────────────────────────────────

    def _run_poll_cycle(self, only_account_id: str | None = None) -> None:
        with self._cycle_lock:
            accounts = self._store.list_all()
            if only_account_id is not None:
                accounts = [a for a in accounts if a["id"] == only_account_id]
                if not accounts:
                    logger.warning(f"refresh_now(): no account with id {only_account_id}")
                    return

            any_new = False
            for account in accounts:
                account_id = account["id"]
                label = account["label"]
                provider = account["provider"]
                fetcher = PROVIDER_FETCHERS.get(provider)
                if fetcher is None:
                    logger.error(f"No fetcher registered for provider '{provider}' (account '{label}') - skipping.")
                    continue

                with self._cache_lock:
                    entry = self._cache["accounts"].setdefault(account_id, self._empty_account_cache())
                    prior_uidvalidity = entry.get("uidvalidity")
                    prior_seen = set(entry.get("seen_ids", []))
                    was_baselined = entry.get("baselined", False)

                try:
                    result = fetcher.fetch_recent(account)
                except RefreshError as exc:
                    self._record_poll_failure(
                        account_id, label, "Google authorization revoked — reconnect this account.", exc
                    )
                    continue
                except Exception as exc:
                    self._record_poll_failure(account_id, label, f"Could not check this account: {exc}", exc)
                    continue

                emails: list[dict] = result["emails"]
                uidvalidity = result["uidvalidity"]
                unread_count = result["unread_count"]

                # UIDVALIDITY change (IMAP mailbox rebuilt) -> old seen_ids
                # are meaningless. Reset and silently re-baseline instead of
                # a false "N new emails" burst.
                uidvalidity_changed = (
                    uidvalidity is not None
                    and prior_uidvalidity is not None
                    and uidvalidity != prior_uidvalidity
                )
                if uidvalidity_changed:
                    prior_seen = set()
                    was_baselined = False
                    logger.warning(f"UIDVALIDITY changed for '{label}' - re-baselining (no announcement).")

                now = datetime.now(config.TIMEZONE)
                newly_unannounced: list[str] = []
                if was_baselined:
                    for e in emails:
                        if e["id"] in prior_seen:
                            continue
                        if not e["unread"]:
                            continue
                        if _age_seconds(e["date"], now) > NEW_MAIL_MAX_AGE_S:
                            continue
                        newly_unannounced.append(e["id"])
                # First poll for an account = baseline: mark everything seen,
                # announce nothing (a huge inbox shouldn't dump every message
                # as "new" the moment an account is connected).

                seen_ids = list(prior_seen | {e["id"] for e in emails})
                seen_ids = seen_ids[-SEEN_IDS_CAP:]  # FIFO-ish cap; membership is what matters, not order

                with self._cache_lock:
                    entry = self._cache["accounts"].setdefault(account_id, self._empty_account_cache())
                    entry["last_poll"] = now.isoformat()
                    entry["last_error"] = None
                    entry["consecutive_failures"] = 0
                    entry["unread_count"] = unread_count
                    entry["uidvalidity"] = uidvalidity
                    entry["baselined"] = True
                    entry["seen_ids"] = seen_ids
                    entry["emails"] = emails[:CACHED_EMAILS_PER_ACCOUNT]
                    if uidvalidity_changed:
                        entry["unannounced_ids"] = []
                    if newly_unannounced:
                        existing_unannounced = set(entry.get("unannounced_ids", []))
                        entry["unannounced_ids"] = list(existing_unannounced | set(newly_unannounced))
                        any_new = True
                    self._write_cache_locked()

            if any_new and self._on_new_mail is not None:
                try:
                    self._on_new_mail()
                except Exception as exc:
                    logger.error(f"on_new_mail callback failed: {exc}")

    def _record_poll_failure(self, account_id: str, label: str, message: str, exc: Exception) -> None:
        with self._cache_lock:
            entry = self._cache["accounts"].setdefault(account_id, self._empty_account_cache())
            entry["last_error"] = message
            entry["consecutive_failures"] = entry.get("consecutive_failures", 0) + 1
            failures = entry["consecutive_failures"]
            self._write_cache_locked()
        # WARNING on the first failure, then only every 10th - one dead
        # account (e.g. revoked Gmail access) shouldn't spam the log forever.
        if failures == 1 or failures % 10 == 0:
            logger.warning(f"Email poll failed for '{label}' (failure #{failures}): {exc}")

    def refresh_now(self, account_id: str | None = None) -> None:
        """Synchronous poll cycle, for voice 'check right now' style requests."""
        self._run_poll_cycle(only_account_id=account_id)

    # ── Proactive announcements ───────────────────────────────────────────────

    def claim_announcement(self) -> str | None:
        """
        Atomically compose the pending-announcement text AND clear every
        account's unannounced_ids, applying a staleness filter first: a
        pending id survives only if it's still cached, still unread (read
        elsewhere -> drop silently), and younger than ANNOUNCE_MAX_AGE_S.

        This "claims" the announcement - if the caller then fails to actually
        speak it (e.g. a TTS error), the emails remain in cache and fully
        queryable, but the nudge itself is not retried. That is a deliberate
        trade-off: the alternative (compose-then-mark-separately) can spin
        the main loop announcing the same text forever if marking never runs.
        """
        now = datetime.now(config.TIMEZONE)
        surviving: list[dict] = []
        changed = False
        with self._cache_lock:
            for entry in self._cache["accounts"].values():
                unannounced = entry.get("unannounced_ids", [])
                if not unannounced:
                    continue
                changed = True
                emails_by_id = {e["id"]: e for e in entry.get("emails", [])}
                for eid in unannounced:
                    e = emails_by_id.get(eid)
                    if e is None or not e.get("unread"):
                        continue
                    if _age_seconds(e["date"], now) > ANNOUNCE_MAX_AGE_S:
                        continue
                    surviving.append(e)
                entry["unannounced_ids"] = []
            surviving.sort(key=lambda e: e["date"], reverse=True)
            self._last_announced_context = list(surviving)
            if changed:
                self._write_cache_locked()

        if not surviving:
            return None
        return self._compose_announcement_text(surviving)

    def get_announced_context(self) -> list[dict]:
        """The emails from the most recent claim_announcement() call - used
        to seed the announcement conversation's first turn (e.g. "yes, read
        them" needs to know what "them" refers to)."""
        with self._cache_lock:
            return list(self._last_announced_context)

    @staticmethod
    def _compose_announcement_text(emails: list[dict]) -> str:
        if len(emails) == 1:
            e = emails[0]
            return (
                f"Hey, you've got a new email from {e['sender_name']} about "
                f"\"{e['subject']}\". Want me to read it?"
            )
        names: list[str] = []
        for e in emails[:3]:
            first_name = e["sender_name"].split()[0] if e["sender_name"] else e["sender_email"]
            if first_name not in names:
                names.append(first_name)
        names_str = ", ".join(names)
        if len(emails) > len(names):
            names_str += ", and others"
        return (
            f"Hey, you've got {len(emails)} new emails — from {names_str}. "
            f"Let me know if you'd like me to read them or if you have any questions."
        )

    # ── Accounts ──────────────────────────────────────────────────────────────

    def list_accounts_safe(self) -> list[dict]:
        return self._store.list_safe()

    def has_accounts(self) -> bool:
        return len(self._store.list_safe()) > 0

    def list_accounts_safe(self) -> list[dict]:
        """Credential-free account views (id/label/provider/created_at)."""
        return self._store.list_safe()

    def resolve_accounts_by_hint(self, account_hint: str | None) -> list[dict]:
        """
        Safe account views matching a spoken account name. No hint returns
        every account, so the caller decides between "use the only one" and
        "ask which". Same case-insensitive label-or-provider substring rule
        get_cached_emails() applies, so "Gmail" hits the gmail_oauth provider
        and "Hostinger" hits a label.
        """
        accounts = self._store.list_safe()
        hint = _clean_str(account_hint).lower()
        if not hint:
            return accounts
        return [
            a for a in accounts
            if hint in a["label"].lower() or hint in a["provider"].lower()
        ]

    def get_accounts_context(self) -> str:
        """
        An authoritative one-line summary of the connected email accounts,
        for injection into the conversational LLM's system prompt so it can
        answer "how many / which email accounts are connected?", "do you have
        my Gmail?" etc. from real data instead of confabulating. Reads the
        live in-memory store, so a just-added account is reflected with no
        restart. Never raises.
        """
        accounts = self._store.list_safe()
        if not accounts:
            return "No email accounts are connected yet."
        parts = []
        for a in accounts:
            kind = "Gmail" if a.get("provider") == PROVIDER_GMAIL else "IMAP"
            parts.append(f"{a.get('label')} ({kind})")
        return (
            "You have live access to these connected email accounts: "
            + ", ".join(parts)
            + ". They are ALL fully connected and ready to use right now — there is "
            "no syncing, importing, indexing, or setup still in progress, so never "
            "tell the user an account is still syncing, loading, or connecting. You "
            "can read, search, and summarize their mail through your email tools. If "
            "the user asks how many or which email accounts are connected, answer from "
            "this list and refer to each account by the label shown above."
        )

    def add_imap_account(
        self, label: str, host: str, port: int, username: str, password: str, use_ssl: bool = True,
        smtp_host: str | None = None, smtp_port: int | None = None, smtp_use_ssl: bool = True,
    ) -> dict:
        """Validates the connection BEFORE storing - raises whatever the
        fetcher raised on failure; the API endpoint maps that to a sanitized 400.

        The outgoing (SMTP) server is validated too, but ONLY when smtp_host
        was explicitly supplied. Otherwise it is derived at send time and never
        probed here, so adding an account for reading can't start failing
        because its mail host happens not to accept SMTP logins."""
        credentials = {
            "host": host, "port": port, "username": username,
            "password": password, "use_ssl": use_ssl,
        }
        PROVIDER_FETCHERS[PROVIDER_IMAP].test_connection(credentials)

        if smtp_host:
            PROVIDER_SENDERS[PROVIDER_IMAP].test_connection({
                **credentials,
                "smtp_host": smtp_host, "smtp_port": smtp_port, "smtp_use_ssl": smtp_use_ssl,
            })

        account = self._store.add_imap_account(
            label, host, port, username, password, use_ssl, smtp_host, smtp_port, smtp_use_ssl
        )
        self.request_poll_soon()
        return account

    def add_gmail_account(self, label: str, refresh_token: str, email_address: str) -> dict:
        """
        The OAuth callback already validates the refresh token live (a
        getProfile call) before calling this, so no redundant test_connection
        call is made here.
        """
        account = self._store.add_gmail_account(label, refresh_token, email_address)
        self.request_poll_soon()
        return account

    def delete_account(self, account_id: str) -> bool:
        """Serializes against the poller via _cycle_lock, so a delete can
        never land mid-cycle for this account."""
        with self._cycle_lock:
            removed = self._store.delete(account_id)
            if removed:
                with self._cache_lock:
                    self._cache["accounts"].pop(account_id, None)
                    self._write_cache_locked()
        return removed

    # ── OAuth state nonces ────────────────────────────────────────────────────

    def mint_oauth_state(self) -> str:
        """Short-lived, single-use nonce for the Gmail OAuth callback (which
        can't carry the bearer header - it's a browser redirect). Minting is
        gated behind require_token on the /oauth-url endpoint, so with
        JARVIS_API_TOKEN unset no nonce can ever be minted and the whole
        OAuth chain fails closed."""
        now = time.time()
        with self._oauth_lock:
            expired = [s for s, exp in self._oauth_states.items() if exp <= now]
            for s in expired:
                del self._oauth_states[s]
            state = secrets.token_urlsafe(32)
            self._oauth_states[state] = now + OAUTH_STATE_TTL_S
        return state

    def consume_oauth_state(self, state: str) -> bool:
        """Atomically pop and validate a state nonce - single-use by construction."""
        if not state:
            return False
        now = time.time()
        with self._oauth_lock:
            expiry = self._oauth_states.pop(state, None)
        return expiry is not None and expiry > now

    # ── Pending contact-address request (voice thread <-> API thread) ─────────
    #
    # Email addresses are hard for STT: symbols, dots, unusual spellings. When
    # the send flow needs an address it doesn't have, it opens a request here
    # and asks aloud. The boss may answer by voice (the default) OR type it
    # into the dashboard, which POSTs to /email/pending-contact and lands in
    # this slot. The voice path never depends on the typed one - the UI is
    # strictly optional.
    #
    # Same shape as the OAuth nonce store above: one lock, a TTL, and an
    # atomic claim (cf. claim_announcement).

    def open_contact_request(self, person_name: str) -> None:
        """Begin waiting for `person_name`'s address. Overwrites any stale
        slot - only one conversation can be in flight at a time."""
        with self._contact_lock:
            self._contact_request = {
                "name": _clean_str(person_name),
                "expires_at": time.time() + CONTACT_REQUEST_TTL_S,
                "email": None,
            }

    def _live_contact_request_locked(self) -> dict | None:
        """The open request, or None if absent/expired. Assumes the lock is held."""
        request = self._contact_request
        if request is None:
            return None
        if request["expires_at"] <= time.time():
            self._contact_request = None
            return None
        return request

    def get_contact_request(self) -> dict | None:
        """{"name", "expires_in"} for the dashboard to render an input box, or
        None when Jarvis isn't waiting on an address."""
        with self._contact_lock:
            request = self._live_contact_request_locked()
            if request is None:
                return None
            return {
                "name": request["name"],
                "expires_in": max(0, int(request["expires_at"] - time.time())),
            }

    def submit_contact_email(self, address: str) -> bool:
        """
        Fulfil the open request with a typed address (API thread). Returns
        False when nothing is pending/it expired, or the address is malformed -
        the endpoint maps those to 409 and 400 respectively.

        A second submission simply replaces an unclaimed first one, so a typo
        corrected before Jarvis reads it does the right thing.
        """
        address = _clean_str(address)
        if not is_valid_email(address):
            return False
        with self._contact_lock:
            request = self._live_contact_request_locked()
            if request is None:
                return False
            request["email"] = address
        logger.info("A contact address was submitted from the dashboard.")
        return True

    def claim_contact_email(self) -> str | None:
        """
        Atomically take any typed address (voice thread). Pops only the VALUE
        and leaves the request OPEN, so a boss whose first attempt had a typo
        can type again on the next round without the endpoint 409-ing.
        """
        with self._contact_lock:
            request = self._live_contact_request_locked()
            if request is None:
                return None
            address, request["email"] = request["email"], None
        return address

    def close_contact_request(self) -> None:
        """Always called from the send handler's finally, and again from
        _run_conversation's, so an abandoned request can never linger into an
        unrelated later conversation. Idempotent."""
        with self._contact_lock:
            self._contact_request = None

    # ── Voice: intent classification ─────────────────────────────────────────

    @staticmethod
    def _not_email_result() -> dict:
        return {
            "intent": "not_email", "account_hint": None, "sender": None,
            "subject_hint": None, "timeframe": None, "count": None,
            "ordinal": None, "days_back": None, "unread_only": False,
        }

    @classmethod
    def _coerce_intent_result(cls, parsed: dict) -> dict:
        result = cls._not_email_result()
        intent = _clean_str(parsed.get("intent")).lower()
        if intent in _VALID_INTENTS:
            result["intent"] = intent
        for key in ("account_hint", "sender", "subject_hint", "timeframe"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                result[key] = value.strip()
        for key in ("count", "ordinal", "days_back"):
            value = parsed.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = int(value)
                if value <= 0:
                    continue  # nonsense from the model - treat as absent
                if key == "days_back":
                    value = min(value, DEEP_SEARCH_MAX_DAYS)
                result[key] = value
        result["unread_only"] = bool(parsed.get("unread_only", False))
        return result

    @staticmethod
    def _scope_phrase_for_prompt(timeframe: str | None, days_back: int | None) -> str:
        """Short human phrase for a time scope, for the search-in-progress
        prompt block. "" when neither is set."""
        if days_back:
            return f"around {days_back} days back"
        if timeframe in ("today", "yesterday"):
            return timeframe
        if timeframe == "this_week":
            return "this week"
        return ""

    def classify_intent(
        self,
        text: str,
        context_emails: list[dict] | None = None,
        active_search: dict | None = None,
    ) -> dict:
        """
        One Haiku call classifying a transcribed utterance into the email
        voice-intent taxonomy. Returns a fully-defaulted dict (callers never
        KeyError). Never raises - any failure/unparseable/unrecognized
        response maps to {"intent": "not_email", ...} so a classifier hiccup
        falls through to the normal Claude conversation rather than
        hijacking the turn.

        `context_emails` is the batch the assistant just SPOKE to the user
        (an announcement or a spoken list, newest first) - it is rendered as
        a numbered list so the model can resolve replies that only make sense
        against it ("yes, please", "the second one", "the last one about X")
        into read_email with a grounded ordinal, or a declination into
        dismiss. The numbering must match the caller's stored list exactly,
        since the returned ordinal indexes into it.

        `active_search` is the in-flight "find a specific email" criteria
        (pending_search) when the boss is mid-search. When present, an oblique
        reply or a garbled correction ("no, it's about the KPI report", "a
        couple weeks ago") should be read as REFINING that search
        (read_email, extracting the corrected/added sender/topic/timeframe),
        not dropped to not_email - the misclassification that let a search
        leak into plain chat. A clear topic change still classifies not_email.
        """
        fallback = self._not_email_result()
        text = text.strip()
        if not text:
            return fallback

        account_labels = [a["label"] for a in self._store.list_safe()]
        accounts_desc = ", ".join(account_labels) if account_labels else "(none connected)"

        context_note = ""
        dismiss_line = ""
        search_note = ""
        if active_search:
            known_bits = []
            if active_search.get("sender"):
                known_bits.append(f"from {active_search['sender']}")
            if active_search.get("subject_hint"):
                known_bits.append(f"about \"{active_search['subject_hint']}\"")
            scope = self._scope_phrase_for_prompt(
                active_search.get("timeframe"), active_search.get("days_back")
            )
            if scope:
                known_bits.append(scope)
            known_desc = " ".join(known_bits) if known_bits else "a specific email"
            search_note = (
                "\n\nIMPORTANT: the user is in the middle of trying to find a "
                f"specific email ({known_desc}). Unless they have clearly changed "
                "the subject to something unrelated to finding that email, treat "
                "this reply as REFINING that search: use intent read_email and "
                "extract any corrected or added sender, topic (subject_hint), or "
                "timeframe/days_back the reply provides - even if the wording is "
                "garbled or oblique (\"no, it's the KPI one\", \"it was a couple "
                "weeks back\", \"it's from Michael, not Michelle\"). Only use "
                "not_email if the reply is plainly about a different subject."
            )
        if context_emails:
            numbered = "\n".join(
                f'{i}. "{_truncate_for_prompt(_clean_str(e.get("subject")), 80)}" '
                f'from {_clean_str(e.get("sender_name"))}'
                for i, e in enumerate(context_emails[:10], start=1)
            )
            single = len(context_emails) == 1
            context_note = (
                "\n\nThe assistant just presented these emails to the user (numbered, "
                "newest first) and offered to read them:\n"
                f"{numbered}\n"
                "Rules for replies that refer to this list:\n"
                "- A bare affirmative (\"yes\", \"yes please\", \"sure\", \"go ahead\", "
                "\"read it\", \"read them\") means read_email"
                + (" with ordinal 1." if single else ".")
                + "\n"
                "- A positional reference means read_email with that position as the "
                "ordinal (\"the first one\" = 1, \"the second one\" = 2). Phrasings "
                "built around the word \"last\" split into two OPPOSITE meanings - do "
                "not conflate them:\n"
                "  * POSITIONAL \"last\" - \"the last one\", \"the last one you "
                "listed/mentioned\", \"the last one in the list\" - means the FINAL "
                "numbered item (the oldest of the group, since the list is newest "
                "first).\n"
                "  * TEMPORAL \"last\"/\"latest\"/\"most recent\"/\"newest\" - \"the "
                "last email I received/got\", \"the latest one\", \"the most recent "
                "email\" - means ordinal 1 (the list is newest first), REGARDLESS of "
                "how many items are listed. These describe WHEN the email arrived, "
                "not WHERE it sits in what was just spoken.\n"
                "  * Rule of thumb: if \"last\" is immediately followed by a verb "
                "about receiving/getting/arriving (\"last email I received\", \"last "
                "one that came in\"), treat it as TEMPORAL -> ordinal 1. If \"last\" "
                "stands alone or refers to the enumeration itself (\"the last one\", "
                "\"the last one you mentioned\"), treat it as POSITIONAL -> final "
                "number.\n"
                "- A content cue ALWAYS beats position: \"the one from X\" or \"the "
                "(last) one about Y\" means read_email with ordinal set to the "
                "numbered item matching that person/topic, and sender/subject_hint "
                "filled in from the cue.\n"
                "- A declination (\"no\", \"no thanks\", \"not now\", \"never mind\", "
                "\"later\") means dismiss.\n"
                "- A request for MORE emails, OLDER than the ones listed above "
                "(\"load earlier emails\", \"show me older ones\", \"what else\", "
                "\"more\", \"go further back\", \"the previous ones\", \"anything "
                "before those\") means load_more - NOT a fresh list_emails.\n"
                "- Short replies that only make sense as answers to this list must "
                "NOT be classified not_email. An unrelated request (\"read me a "
                "poem\", \"what's the weather\") is still not_email."
            )
            dismiss_line = (
                "- dismiss: the user is declining the emails the assistant just "
                "offered (\"no thanks\", \"not now\", \"never mind\").\n"
                "- load_more: the user wants MORE emails, older than the ones just "
                "listed above (\"load earlier emails\", \"show me older ones\", "
                "\"more\", \"what else\", \"go further back\").\n"
            )

        try:
            response = self._client.messages.create(
                model=INTENT_MODEL_ID,
                max_tokens=INTENT_MAX_TOKENS,
                timeout=INTENT_TIMEOUT,
                system=(
                    "Classify the user's utterance to a voice assistant into ONE "
                    "email-related intent, or \"not_email\" if it isn't about email at "
                    f"all. Connected email accounts: {accounts_desc}.{context_note}{search_note}\n\n"
                    "Intents:\n"
                    "- not_email: not about email.\n"
                    "- check_new: asking if/how many new or unread emails exist (from cache).\n"
                    "- check_now: explicitly wants a live check right now (\"check my email right now\").\n"
                    "- list_emails: wants a list of emails matching a time range or unread state. "
                    "\"What was the last/latest email I received?\" (with no presented list "
                    "above) is list_emails with count 1.\n"
                    "- from_person: asking about emails from a specific person.\n"
                    "- read_email: wants a specific email READ ALOUD in full (\"read it\", "
                    "\"read me that email\", \"read the one from Sarah\", or a bare pick "
                    "like \"the second one\" / \"yes, read it\").\n"
                    "- email_question: asking a specific factual QUESTION about a particular "
                    "email rather than wanting it read in full - who sent it, what the "
                    "subject is, when it arrived, what the sender asked for, or whether it "
                    "mentions something (\"who sent it?\", \"remind me who that was from\", "
                    "\"what's the subject?\", \"what did they ask me to do?\", \"does it "
                    "mention a deadline?\"). This is answered briefly from the email's "
                    "content - it is NOT read_email (full read) and NOT a broad inbox query.\n"
                    "- summarize: wants a summary or the gist of the inbox or of one email "
                    "(\"summarize it\", \"give me the gist\").\n"
                    "- action_needed: asking whether anything in their email needs action/response.\n"
                    "- meetings_mentioned: asking whether any meetings are mentioned in their email.\n"
                    "- send_email: wants to SEND a new email, or REPLY to an email being "
                    "discussed (\"email Michael and tell him I'm running late\", \"send "
                    "Sarah a note about the budget\", \"reply to it and say I'll review "
                    "it tonight\"). Recipient and message content are extracted by a "
                    "separate call - do not try to fill them in here.\n"
                    f"{dismiss_line}\n"
                    "Respond with ONLY a JSON object (no markdown fences, no commentary) with "
                    "exactly these keys: intent (one of the above), account_hint (string or "
                    "null - an account label/provider name ONLY if the user explicitly said "
                    "it in this utterance; never fill it in just because accounts exist), "
                    "sender (string or null - a person's name or email mentioned as who the "
                    "email is FROM, whether or not a list was just presented), subject_hint "
                    "(string or null - the topic/subject the user described, in their own "
                    "words; do not require it to match an exact subject line). When an "
                    "utterance names BOTH a person and a topic in the same sentence (e.g. "
                    "\"an email from Michael about my KPI monthly\" -> sender=\"Michael\", "
                    "subject_hint=\"KPI monthly\"; \"did Sarah send anything about the "
                    "budget?\" -> sender=\"Sarah\", subject_hint=\"budget\"), extract BOTH - "
                    "never drop the topic just because a sender was also found, and vice "
                    "versa. timeframe (one of "
                    "\"today\"/\"yesterday\"/\"this_week\"/null), count (integer or null), "
                    "ordinal (integer or null - 1 for \"the first one\"/\"the one from X\", 2 "
                    "for \"the second one\", etc.), days_back (integer or null - only when "
                    "the user asks about mail older than this week: \"two weeks ago\" -> 14, "
                    f"\"last month\" -> 31, vague-but-clearly-not-recent phrasing with no "
                    "specific number - \"a while ago\", \"a bit back\", \"sometime last "
                    f"month or so\", \"not too long ago\" - -> {VAGUE_TIMEFRAME_DAYS_BACK_DEFAULT} "
                    "as a reasonable default rather than null; null ONLY when today/"
                    "yesterday/this_week already covers it or NO time reference was made "
                    "at all), unread_only (true/false)."
                ),
                messages=[{"role": "user", "content": text}],
            )
            parsed = parse_json_object(extract_response_text(response))
            return self._coerce_intent_result(parsed) if isinstance(parsed, dict) else fallback
        except Exception as exc:
            logger.warning(f"classify_intent() failed ({exc}) - treating as not_email.")
            return fallback

    def classify_scope(self, answer_text: str) -> dict:
        """
        Interpret the boss's spoken answer to a "how far back / how many?"
        clarifying question. Returns
        {"count": int|None, "timeframe": str|None, "days_back": int|None}.
        Never raises; an unparseable/silent answer returns all None so the
        caller defaults to today's unread and says so aloud.
        """
        fallback = {"count": None, "timeframe": None, "days_back": None}
        answer_text = answer_text.strip()
        if not answer_text:
            return fallback
        try:
            response = self._client.messages.create(
                model=INTENT_MODEL_ID,
                max_tokens=SCOPE_MAX_TOKENS,
                timeout=INTENT_TIMEOUT,
                system=(
                    "The user was asked how far back to look in their email, or how "
                    "many recent emails to check. Extract from their reply ONLY a JSON "
                    "object (no markdown, no commentary) with keys: count (integer or "
                    "null), timeframe (one of \"today\"/\"yesterday\"/\"this_week\" or "
                    "null), days_back (integer or null - only for periods longer than "
                    "this week: \"the last two weeks\" -> 14, \"the past month\" -> 31, "
                    "vague-but-clearly-not-recent phrasing with no specific number - \"a "
                    "while ago\", \"not too long ago\", \"a bit back\" - -> "
                    f"{VAGUE_TIMEFRAME_DAYS_BACK_DEFAULT} as a reasonable default). "
                    "If they gave a number of emails, set count. If they gave a time "
                    "period, set timeframe or days_back. If nothing is clear, all null."
                ),
                messages=[{"role": "user", "content": answer_text}],
            )
            parsed = parse_json_object(extract_response_text(response))
            if not isinstance(parsed, dict):
                return fallback
            result = dict(fallback)
            for key in ("count", "days_back"):
                value = parsed.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    value = int(value)
                    if value <= 0:
                        continue
                    if key == "days_back":
                        value = min(value, DEEP_SEARCH_MAX_DAYS)
                    result[key] = value
            timeframe = parsed.get("timeframe")
            if isinstance(timeframe, str) and timeframe.strip():
                result["timeframe"] = timeframe.strip()
            return result
        except Exception as exc:
            logger.warning(f"classify_scope() failed ({exc}) - defaulting to unspecified scope.")
            return fallback

    # ── Voice: deterministic cache lookups (no LLM) ──────────────────────────

    # Common short words that carry no topical signal on their own - dropped
    # before token-overlap comparison so they can't produce a false match.
    _SUBJECT_STOPWORDS: frozenset[str] = frozenset({
        "the", "a", "an", "my", "about", "of", "to", "for", "and", "is",
        "in", "on", "re", "fwd", "your", "our",
    })

    @classmethod
    def subject_hint_matches(cls, subject_hint: str | None, actual_subject: str) -> bool:
        """
        Loose, word-order-independent match between a spoken topic hint
        ("KPI monthly") and a real subject line ("Monthly KPI Summary") -
        deliberately NOT a substring check, since a correctly-extracted
        paraphrase almost never appears as a contiguous substring of the real
        subject. Tokenizes both, drops stopwords/short words, and requires
        every token to hit for a short (1-2 word) hint, or a majority for a
        longer one. Safe to be this lenient because every caller applies it
        AFTER narrowing by sender - a single strong content-word match on an
        already-small, one-person pool is a reasonable bar; matching it
        against a WHOLE mailbox would be too permissive. No hint at all
        always matches (there's nothing to check).
        """
        if not subject_hint:
            return True
        tokens = [
            t for t in re.findall(r"[a-z0-9]+", subject_hint.lower())
            if len(t) > 2 and t not in cls._SUBJECT_STOPWORDS
        ]
        if not tokens:
            return subject_hint.lower() in actual_subject.lower()
        subject_lower = actual_subject.lower()
        hits = sum(1 for t in tokens if t in subject_lower)
        required = len(tokens) if len(tokens) <= 2 else (len(tokens) + 1) // 2
        return hits >= required

    def get_cached_emails(
        self,
        account_hint: str | None = None,
        sender: str | None = None,
        subject: str | None = None,
        timeframe: str | None = None,
        unread_only: bool = False,
        limit: int = 25,
    ) -> list[dict]:
        """Deterministic filter over the cache. account_hint fuzzy-matches an
        account's label or provider (case-insensitive substring); sender is a
        strict substring filter; subject uses the looser token-overlap match
        (see subject_hint_matches). Newest first."""
        safe_accounts = self._store.list_safe()
        with self._cache_lock:
            pool: list[dict] = []
            for entry in self._cache["accounts"].values():
                pool.extend(entry.get("emails", []))

        if account_hint:
            hint_lower = account_hint.lower()
            matching_ids = {
                a["id"] for a in safe_accounts
                if hint_lower in a["label"].lower() or hint_lower in a["provider"].lower()
            }
            pool = [e for e in pool if e["account_id"] in matching_ids]

        if sender:
            sender_lower = sender.lower()
            pool = [
                e for e in pool
                if sender_lower in e["sender_name"].lower() or sender_lower in e["sender_email"].lower()
            ]

        if subject:
            pool = [e for e in pool if self.subject_hint_matches(subject, e["subject"])]

        if unread_only:
            pool = [e for e in pool if e["unread"]]

        if timeframe:
            pool = [e for e in pool if self._matches_timeframe(e["date"], timeframe)]

        pool.sort(key=lambda e: e["date"], reverse=True)
        return pool[:limit]

    # ── Voice: unified scope routing (cache vs on-demand deep search) ────────

    @staticmethod
    def _scope_window_days(timeframe: str | None, days_back: int | None) -> int | None:
        """
        How many days of mail a query scope needs. days_back wins outright
        (timeframe is ignored when it's set); None means "no window asked
        for". this_week is computed from the actual weekday - on a Monday or
        Tuesday it fits inside the poll cache, avoiding a pointless live
        search.
        """
        if days_back:
            return days_back
        if timeframe == "today":
            return 1
        if timeframe == "yesterday":
            return 2
        if timeframe == "this_week":
            return datetime.now(config.TIMEZONE).date().weekday() + 1
        return None

    def needs_deep_search(self, timeframe: str | None, days_back: int | None) -> bool:
        """Pre-flight for handlers: True when get_emails_for_scope() with
        these arguments will hit the network, so a spoken filler line can go
        out BEFORE the seconds-long live search starts."""
        window = self._scope_window_days(timeframe, days_back)
        return window is not None and window > FETCH_WINDOW_DAYS

    def get_emails_for_scope(
        self,
        account_hint: str | None = None,
        sender: str | None = None,
        subject_hint: str | None = None,
        timeframe: str | None = None,
        days_back: int | None = None,
        unread_only: bool = False,
        limit: int = 25,
    ) -> list[dict]:
        """
        Unified voice-query fetch: answers from the poll cache when the
        requested window fits inside it, and runs a live provider-side deep
        search (fetcher.search) when the boss asked for older mail.

        INVARIANT: the deep path is fully stateless - it never reads or
        writes the poll cache, seen_ids, unannounced_ids, last_error, or
        announcement state, and holds no manager locks during network calls
        (store.list_all() locks internally and returns deep copies). Deep
        results therefore can never perturb polling, announcements, or the
        dashboard - data/email_cache.json stays byte-identical.
        """
        window = self._scope_window_days(timeframe, days_back)

        if window is None or window <= FETCH_WINDOW_DAYS:
            emails = self.get_cached_emails(
                account_hint=account_hint,
                sender=sender,
                subject=subject_hint,
                timeframe=timeframe if not days_back else None,
                unread_only=unread_only,
                limit=limit,
            )
            if days_back:
                # get_cached_emails has no day filter; trim to the asked-for
                # window (the cache spans FETCH_WINDOW_DAYS regardless).
                now = datetime.now(config.TIMEZONE)
                emails = [e for e in emails if _age_seconds(e["date"], now) <= days_back * 86400]
            return emails

        accounts = self._store.list_all()
        if account_hint:
            hint_lower = account_hint.lower()
            accounts = [
                a for a in accounts
                if hint_lower in a["label"].lower() or hint_lower in a["provider"].lower()
            ]

        results: list[dict] = []
        for account in accounts:
            fetcher = PROVIDER_FETCHERS.get(account["provider"])
            if fetcher is None:
                continue
            try:
                results.extend(fetcher.search(
                    account,
                    days=window,
                    max_n=DEEP_SEARCH_MAX_MESSAGES,
                    sender=sender,
                    subject=subject_hint,
                    unread_only=unread_only,
                ))
            except Exception as exc:
                # Log-and-continue so one dead account can't kill the search.
                # Deliberately does NOT write last_error into the cache - the
                # poller owns account health, and this path stays stateless.
                logger.warning(f"Deep email search failed for '{account['label']}': {exc}")

        if unread_only:
            results = [e for e in results if e["unread"]]
        if timeframe and not days_back:
            # The window was derived from the timeframe (e.g. this_week = a
            # superset reaching back to Monday) - re-apply the exact filter.
            results = [e for e in results if self._matches_timeframe(e["date"], timeframe)]
        results.sort(key=lambda e: e["date"], reverse=True)
        return results[:limit]

    def get_older_emails(
        self,
        account_hint: str | None = None,
        before_date: str | None = None,
        sender: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """
        Emails strictly OLDER than `before_date` (an ISO date string, normally
        the oldest email already read aloud) — this is what "load earlier
        emails" pages back with. Live provider-side search bounded by a
        `before` date, so it never just re-returns the newest mail the boss
        already heard.

        Same statelessness invariant as get_emails_for_scope: never reads or
        writes the poll cache, seen_ids, or announcement state, and holds no
        manager locks during network calls.
        """
        if not before_date:
            return []
        try:
            anchor = datetime.fromisoformat(before_date)
        except (ValueError, TypeError):
            return []

        # Provider `before`/BEFORE bounds are date-granular and exclusive, so
        # ask for anchor_date + 1 day and drop same-day-but-newer messages with
        # the exact datetime filter below.
        before_param = anchor.astimezone(config.TIMEZONE).date() + timedelta(days=1)

        accounts = self._store.list_all()
        if account_hint:
            hint_lower = account_hint.lower()
            accounts = [
                a for a in accounts
                if hint_lower in a["label"].lower() or hint_lower in a["provider"].lower()
            ]

        results: list[dict] = []
        for account in accounts:
            fetcher = PROVIDER_FETCHERS.get(account["provider"])
            if fetcher is None:
                continue
            try:
                results.extend(fetcher.search(
                    account,
                    days=DEEP_SEARCH_MAX_DAYS,
                    max_n=DEEP_SEARCH_MAX_MESSAGES,
                    sender=sender,
                    before=before_param,
                ))
            except Exception as exc:
                logger.warning(f"Older-email search failed for '{account['label']}': {exc}")

        older: list[dict] = []
        for e in results:
            try:
                if datetime.fromisoformat(e["date"]) < anchor:
                    older.append(e)
            except (ValueError, TypeError):
                continue
        older.sort(key=lambda e: e["date"], reverse=True)
        return older[:limit]

    @staticmethod
    def _matches_timeframe(date_iso: str, timeframe: str) -> bool:
        try:
            dt = datetime.fromisoformat(date_iso).astimezone(config.TIMEZONE)
        except (ValueError, TypeError):
            return False
        today = datetime.now(config.TIMEZONE).date()
        if timeframe == "today":
            return dt.date() == today
        if timeframe == "yesterday":
            return dt.date() == today - timedelta(days=1)
        if timeframe == "this_week":
            start_of_week = today - timedelta(days=today.weekday())
            return dt.date() >= start_of_week
        return True  # unrecognized timeframe -> don't filter

    def get_unread_overview(self) -> dict:
        """{"total": int, "per_account": [{"label", "unread_count", "last_error"}],
        "latest": <normalized email>|None} - drives check_new/check_now answers."""
        safe_accounts = self._store.list_safe()
        with self._cache_lock:
            per_account = []
            total = 0
            latest = None
            for account in safe_accounts:
                entry = self._cache["accounts"].get(account["id"], self._empty_account_cache())
                per_account.append({
                    "label": account["label"],
                    "unread_count": entry.get("unread_count", 0),
                    "last_error": entry.get("last_error"),
                })
                total += entry.get("unread_count", 0)
                for e in entry.get("emails", []):
                    if e["unread"] and (latest is None or e["date"] > latest["date"]):
                        latest = e
            return {"total": total, "per_account": per_account, "latest": latest}

    # ── Voice: Sonnet-tier content reasoning ─────────────────────────────────

    def reason_over_emails(self, question: str, emails: list[dict]) -> str:
        """
        Sonnet-tier (EXTRACTION_MODEL_ID) reasoning over a batch of cached
        email bodies - used for action-needed / meetings-mentioned / inbox
        summarize, where metadata alone isn't enough. Never raises - returns
        a spoken-safe fallback string on failure.
        """
        if not emails:
            return "I didn't find any emails matching that."
        try:
            transcript = "\n\n".join(
                f"From: {e['sender_name']} <{e['sender_email']}>\n"
                f"Subject: {e['subject']}\nDate: {e['date']}\n"
                f"Body: {_truncate_for_prompt(e['body'], REASONING_BODY_CHAR_CAP)}"
                for e in emails
            )
            response = self._reasoning_client.messages.create(
                model=EXTRACTION_MODEL_ID,
                max_tokens=REASONING_MAX_TOKENS,
                timeout=REASONING_TIMEOUT,
                system=(
                    "You are a voice assistant answering a spoken question about the "
                    "user's email, based on the emails provided below. Answer "
                    "conversationally in plain spoken sentences (this will be read "
                    "aloud by text-to-speech) - no markdown, no bullet points, no "
                    "headers. Be concise - a few sentences at most. If nothing in the "
                    "provided emails is relevant to the question, say so plainly."
                ),
                messages=[{"role": "user", "content": f"Question: {question}\n\nEmails:\n{transcript}"}],
            )
            return extract_response_text(response) or "I looked, but couldn't come up with a clear answer."
        except Exception as exc:
            logger.warning(f"reason_over_emails() failed ({exc}).")
            return "I couldn't get through your emails just now — try again in a moment."

    def summarize_email(self, email: dict) -> str:
        """Sonnet-tier summary of a single (typically long) email body. Falls
        back to a truncated verbatim snippet on failure."""
        try:
            response = self._reasoning_client.messages.create(
                model=EXTRACTION_MODEL_ID,
                max_tokens=REASONING_MAX_TOKENS,
                timeout=REASONING_TIMEOUT,
                system=(
                    "Summarize the following email in a few plain spoken sentences "
                    "suitable for text-to-speech (no markdown, no bullet points). "
                    "Cover who it's from, the gist, and anything requiring action."
                ),
                messages=[{"role": "user", "content": f"Subject: {email['subject']}\n\n{email['body']}"}],
            )
            return extract_response_text(response) or _truncate_for_prompt(email["body"], 500)
        except Exception as exc:
            logger.warning(f"summarize_email() failed ({exc}).")
            return _truncate_for_prompt(email["body"], 500)

    def answer_email_question(self, question: str, email: dict) -> str:
        """
        Answer a specific factual question about ONE already-fetched email
        (who sent it, the subject, what they asked for, whether it mentions X)
        in a short spoken sentence or two - WITHOUT re-reading the whole email.
        Realtime Haiku tier: the boss is waiting on a quick answer. Reuses the
        already-in-hand sender/subject/date/body (no fetch, no search). Body is
        sanitized (defensive) and truncated. Never raises - a safe fallback
        string on any failure.
        """
        safe_body = sanitize_body_text(truncate_text(email.get("body", ""), QA_BODY_CHAR_CAP))
        email_block = (
            f"From: {sanitize_body_text(email.get('sender_name', ''))} "
            f"<{email.get('sender_email', '')}>\n"
            f"Subject: {sanitize_body_text(email.get('subject', ''))}\n"
            f"Date: {email.get('date', '')}\n"
            f"Body: {safe_body}"
        )
        try:
            response = self._client.messages.create(
                model=INTENT_MODEL_ID,
                max_tokens=QA_MAX_TOKENS,
                timeout=INTENT_TIMEOUT,
                system=(
                    "You are a voice assistant answering a spoken question about ONE "
                    "specific email (provided below). Answer in one or two short, plain "
                    "spoken sentences suitable for text-to-speech - no markdown, no "
                    "lists. Use ONLY the email's own content. For a simple metadata "
                    "question (who it's from, the subject, when it arrived), just state "
                    "it directly. If the answer isn't in the email, say so briefly."
                ),
                messages=[{"role": "user", "content": f"Question: {question}\n\nEmail:\n{email_block}"}],
            )
            return extract_response_text(response) or "I couldn't work that out from the email just now."
        except Exception as exc:
            logger.warning(f"answer_email_question() failed ({exc}).")
            return "I couldn't work that out from the email just now."

    # ── Voice: sending ────────────────────────────────────────────────────────
    #
    # A bad send cannot be undone, so every model call on this path is built to
    # fail toward NOT sending: classify_send() falls back to an empty parse,
    # classify_confirmation() falls back to "unclear" (never "yes"), and
    # parse_spoken_email_address() returns None rather than a guess.

    @staticmethod
    def _empty_send_parse() -> dict:
        return {
            "recipient": None, "message": None, "subject_hint": None,
            "account_hint": None, "is_reply": False,
        }

    def classify_send(self, text: str, last_read: dict | None = None) -> dict:
        """
        Extract who to email and what to say from a spoken send request. One
        Haiku call, fully defaulted, never raises.

        Deliberately SEPARATE from classify_intent rather than folded into its
        field set: that prompt is a heavily-tuned artifact of the reading build,
        and its "exactly these keys" contract is load-bearing. Keeping send
        extraction here means a change to it can never regress reading. This
        mirrors the existing classify_scope() precedent and only ever runs after
        classify_intent has already returned send_email, so it costs nothing on
        a read turn.

        `last_read` is the email currently in focus (email_ctx["last_read"] -
        the same referent "who sent it?" resolves against). Naming it in the
        prompt is what lets "reply to it and say I'll review it tonight" come
        back as is_reply=True with the message extracted and no recipient.
        """
        fallback = self._empty_send_parse()
        text = text.strip()
        if not text:
            return fallback

        account_labels = [a["label"] for a in self._store.list_safe()]
        accounts_desc = ", ".join(account_labels) if account_labels else "(none connected)"

        reply_note = ""
        if last_read:
            reply_note = (
                "\n\nThe user is currently discussing this specific email:\n"
                f"  From: {_clean_str(last_read.get('sender_name'))} "
                f"<{_clean_str(last_read.get('sender_email'))}>\n"
                f"  Subject: \"{_truncate_for_prompt(_clean_str(last_read.get('subject')), 80)}\"\n"
                "If they are asking to reply to THAT email (\"reply to it\", \"reply and "
                "say...\", \"tell them...\", \"answer him\"), set is_reply true and leave "
                "recipient null - the sender above is already known. If they name a "
                "DIFFERENT person to email, set is_reply false and fill in recipient."
            )

        try:
            response = self._client.messages.create(
                model=INTENT_MODEL_ID,
                max_tokens=SEND_PARSE_MAX_TOKENS,
                timeout=INTENT_TIMEOUT,
                system=(
                    "The user asked a voice assistant to send an email. Extract the "
                    f"details. Connected email accounts: {accounts_desc}.{reply_note}\n\n"
                    "Respond with ONLY a JSON object (no markdown fences, no commentary) "
                    "with exactly these keys:\n"
                    "- recipient (string or null): who the email should go TO, as the user "
                    "said it - a person's name (\"Michael\", \"Sarah Chen\") or a literal "
                    "email address if they spelled one out. Null if they didn't say, or if "
                    "this is a reply to the email above.\n"
                    "- message (string or null): what the user wants the email to SAY, in "
                    "their own words (\"I want to have a meeting at 5pm\", \"I'll review it "
                    "tonight\"). This is the instruction to a drafter, not a subject line. "
                    "Null if they only said who to email and not what.\n"
                    "- subject_hint (string or null): a subject or topic ONLY if they "
                    "explicitly stated one (\"subject it 'Q3 budget'\", \"about the "
                    "budget\"). Do not invent one from the message.\n"
                    "- account_hint (string or null): an account label or provider name "
                    "ONLY if the user explicitly said which account to send from; never "
                    "fill it in just because accounts exist.\n"
                    "- is_reply (true/false): true only when replying to the email shown "
                    "above. False when starting a fresh email to someone.\n\n"
                    "Extract only what was actually said. Never invent a recipient, an "
                    "email address, or message content the user did not express."
                ),
                messages=[{"role": "user", "content": text}],
            )
            parsed = parse_json_object(extract_response_text(response))
            if not isinstance(parsed, dict):
                return fallback

            result = self._empty_send_parse()
            for key in ("recipient", "message", "subject_hint", "account_hint"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    result[key] = value.strip()
            result["is_reply"] = bool(parsed.get("is_reply", False))
            return result
        except Exception as exc:
            logger.warning(f"classify_send() failed ({exc}) - no details extracted.")
            return fallback

    def classify_confirmation(self, answer_text: str) -> str:
        """
        Interpret the boss's spoken answer to "should I send it?" (and to the
        spelled-back-address check). Returns one of CONFIRM_YES / CONFIRM_NO /
        CONFIRM_REVISE / CONFIRM_UNCLEAR.

        Modeled on memory.classify_same_or_new, including the short-circuit on
        empty input with no API call. The asymmetry is the whole point and is
        stronger here than there: only an unambiguous, unqualified affirmative
        returns "yes". Silence, garble, a hedge, and "yes, but…" all resolve to
        something that does NOT send. Any exception resolves to "unclear".
        """
        if not answer_text.strip():
            return CONFIRM_UNCLEAR  # silence never authorizes a send; no API call

        try:
            response = self._client.messages.create(
                model=INTENT_MODEL_ID,
                max_tokens=CONFIRM_MAX_TOKENS,
                timeout=INTENT_TIMEOUT,
                system=(
                    "A voice assistant just read a drafted email aloud and asked the user "
                    "whether to send it. Read the user's reply and answer with ONLY one "
                    "lowercase word:\n"
                    "- \"yes\" ONLY if the reply is an unambiguous, unqualified approval to "
                    "send it right now (\"yes\", \"yeah send it\", \"go ahead\", \"sounds "
                    "good, send\", \"perfect\").\n"
                    "- \"no\" if they clearly do not want it sent (\"no\", \"cancel\", "
                    "\"never mind\", \"don't send it\", \"scrap it\").\n"
                    "- \"revise\" if they approve of sending in principle but want the "
                    "email CHANGED first, or ask for any edit to its wording, subject, "
                    "recipient, or details (\"yes but make it shorter\", \"change the time "
                    "to six\", \"say it more formally\", \"add that I'll bring the deck\").\n"
                    "- \"unclear\" for ANYTHING else: hesitation (\"uh\", \"hmm\", \"wait\", "
                    "\"hold on\"), a contradictory or hedged reply (\"yeah, no\", \"sure I "
                    "guess\", \"I mean, maybe\"), a question back, an unrelated remark, or "
                    "anything you are not fully confident about.\n\n"
                    "Sending an email cannot be undone. When in any doubt, answer "
                    "\"unclear\" rather than \"yes\". Output only that one word."
                ),
                messages=[{"role": "user", "content": answer_text}],
            )
            verdict = extract_response_text(response).strip().lower()
            for candidate in (CONFIRM_REVISE, CONFIRM_YES, CONFIRM_NO):
                if verdict.startswith(candidate):
                    return candidate
            return CONFIRM_UNCLEAR
        except Exception as exc:
            logger.warning(f"classify_confirmation() failed ({exc}) - defaulting to 'unclear'.")
            return CONFIRM_UNCLEAR

    def parse_spoken_email_address(self, text: str) -> str | None:
        """
        Turn a spoken address ("feras at codex dot com", "f e r a s at codex
        dot com") into feras@codex.com. Returns None - never a guess - when the
        result doesn't validate, so the caller re-asks instead of mailing a
        stranger.

        Tries a plain regex on the raw transcript first: STT often produces the
        address verbatim, and that path costs nothing and cannot hallucinate.
        """
        text = _clean_str(text)
        if not text:
            return None

        literal = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        if literal and is_valid_email(literal.group(0)):
            return literal.group(0).lower()

        try:
            response = self._client.messages.create(
                model=INTENT_MODEL_ID,
                max_tokens=ADDRESS_PARSE_MAX_TOKENS,
                timeout=INTENT_TIMEOUT,
                system=(
                    "The user spoke an email address aloud to a voice assistant and it was "
                    "transcribed imperfectly. Reconstruct the address.\n"
                    "- \"at\" means @, \"dot\"/\"point\" means a period, \"underscore\" "
                    "means _, \"dash\"/\"hyphen\" means -, \"plus\" means +.\n"
                    "- Letters may be spelled out one at a time (\"f e r a s\" or \"f-e-r-a-s\" "
                    "-> feras); join them with no separator.\n"
                    "- Ignore surrounding words like \"my email is\" or \"it's\".\n"
                    "- Lowercase the whole address.\n\n"
                    "Output ONLY the address, nothing else. If you cannot reconstruct a "
                    "single complete, plausible email address from the text, output exactly "
                    "NONE. Never invent a domain or a name that was not spoken."
                ),
                messages=[{"role": "user", "content": text}],
            )
            candidate = extract_response_text(response).strip().strip("<>\"' ").lower()
            # The model is only ever trusted through this gate.
            return candidate if is_valid_email(candidate) else None
        except Exception as exc:
            logger.warning(f"parse_spoken_email_address() failed ({exc}).")
            return None

    def draft_email(
        self,
        message: str,
        recipient_name: str,
        subject_hint: str | None = None,
        reply_to: dict | None = None,
        prior_draft: dict | None = None,
        revision: str | None = None,
    ) -> dict:
        """
        Draft a short, appropriately professional email from what the boss
        said - never a verbatim transcript of it. Returns
        {"subject": str, "body": str}.

        Sonnet tier (EXTRACTION_MODEL_ID), matching how reason_over_emails and
        summarize_email chose their model: this is heavier writing work, not a
        realtime classification. max_retries=0 like its siblings, since the boss
        is waiting; the caller speaks a filler line first.

        `subject_hint` is a subject the boss explicitly dictated; absent one,
        the model writes its own. `reply_to` is the email being replied to
        (subject is forced to "Re: <original>", so a hint is ignored there).
        `prior_draft` + `revision` drive the revise loop - the model edits the
        existing draft rather than starting over, so "make it shorter" doesn't
        lose the details from three turns ago.

        Never raises: on failure it returns a plain-but-honest draft built from
        the boss's own words, which the confirmation step will read back anyway.
        """
        fallback_subject = _clean_str(reply_to and reply_to.get("subject")) or "Quick note"
        if reply_to:
            fallback_subject = reply_subject(fallback_subject)
        elif _clean_str(subject_hint):
            fallback_subject = _clean_str(subject_hint)
        fallback = {"subject": fallback_subject, "body": _clean_str(message)}

        context_parts = [f"Recipient's name: {recipient_name or '(unknown)'}"]
        if _clean_str(subject_hint) and not reply_to:
            context_parts.append(
                f"Use exactly this subject line: \"{_clean_str(subject_hint)}\""
            )
        if reply_to:
            context_parts.append(
                "This is a REPLY to the following email. Address the sender by name, "
                "respond to what they actually wrote, and keep the subject as "
                f"\"{reply_subject(_clean_str(reply_to.get('subject')))}\".\n"
                f"  From: {sanitize_body_text(_clean_str(reply_to.get('sender_name')))}\n"
                f"  Subject: {sanitize_body_text(_clean_str(reply_to.get('subject')))}\n"
                f"  Body: {sanitize_body_text(_truncate_for_prompt(_clean_str(reply_to.get('body')), DRAFT_REPLY_BODY_CAP))}"
            )
        if prior_draft and revision:
            context_parts.append(
                "You previously drafted this email:\n"
                f"  Subject: {_clean_str(prior_draft.get('subject'))}\n"
                f"  Body: {_clean_str(prior_draft.get('body'))}\n"
                "The user wants it CHANGED as follows. Apply only that change and keep "
                f"everything else intact - do not start over:\n  \"{_clean_str(revision)}\""
            )
        else:
            context_parts.append(f"What the user wants to say: \"{_clean_str(message)}\"")

        try:
            response = self._reasoning_client.messages.create(
                model=EXTRACTION_MODEL_ID,
                max_tokens=DRAFT_MAX_TOKENS,
                timeout=REASONING_TIMEOUT,
                system=(
                    "You draft short emails on behalf of a busy professional, from a "
                    "one-line spoken instruction. Turn the instruction into a real email - "
                    "never a transcript of what was said.\n\n"
                    "Rules:\n"
                    "- Keep it SHORT: a greeting, one to three sentences, and a sign-off. "
                    "It will be read aloud in full for approval before sending.\n"
                    "- Tone: warm but professional. Do not be effusive or padded.\n"
                    "- Write only what the user actually conveyed. Never invent facts, "
                    "dates, times, commitments, names, or details they did not give you.\n"
                    "- Sign off as the sender without inventing a name (\"Best,\" on its "
                    "own line is fine). Never write a placeholder like [Your Name].\n"
                    "- The subject must be a real subject line: short, specific, no "
                    "\"Subject:\" prefix.\n"
                    "- Plain text only. No markdown, no bullet points, no links.\n\n"
                    "Respond with ONLY a JSON object (no markdown fences, no commentary) "
                    "with exactly these keys: subject (string), body (string)."
                ),
                messages=[{"role": "user", "content": "\n\n".join(context_parts)}],
            )
            parsed = parse_json_object(extract_response_text(response))
            if not isinstance(parsed, dict):
                return fallback

            subject = _clean_str(parsed.get("subject")) or fallback["subject"]
            body = _clean_str(parsed.get("body"))
            if not body:
                return fallback
            if reply_to:
                subject = reply_subject(_clean_str(reply_to.get("subject")))
            return {"subject": subject, "body": body}
        except Exception as exc:
            logger.warning(f"draft_email() failed ({exc}) - falling back to a literal draft.")
            return fallback

    def get_account_send_identity(self, account_id: str) -> str:
        """The address an account sends as (its From header). Reads credentials,
        so like get()/list_all() this is transport-path only - never an API
        response. Returns "" for an unknown account."""
        account = self._store.get(account_id)
        if not account:
            return ""
        sender = PROVIDER_SENDERS.get(account["provider"])
        if sender is None:
            return ""
        return sender.sender_address(account["credentials"])

    def send_email(self, account_id: str, to_addr: str, subject: str, body: str) -> None:
        """
        Send one email. Raises on any failure - the caller speaks the outcome
        aloud, so a swallowed error would be indistinguishable from success.

        Stateless, exactly like the deep-search path: it never reads or writes
        the poll cache, seen_ids, unannounced_ids, last_error, or announcement
        state, and holds no manager lock across the network call (store.get()
        locks internally and returns a deep copy). data/email_cache.json must be
        byte-identical across a send.

        Logs the recipient and subject; never the body, password, or token.
        """
        to_addr = _clean_str(to_addr)
        if not is_valid_email(to_addr):
            raise ValueError(f"Refusing to send to a malformed address: '{to_addr}'")

        account = self._store.get(account_id)
        if not account:
            raise ValueError(f"No such email account: {account_id}")
        sender = PROVIDER_SENDERS.get(account["provider"])
        if sender is None:
            raise ValueError(f"Account '{account['label']}' cannot send email.")

        sender.send(account, to_addr, _clean_str(subject), body)
        logger.success(
            f"Email sent from '{account['label']}' to {to_addr} — subject '{subject}'."
        )

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Dashboard payload: unread counts + recent metadata per account. No
        bodies, no credentials."""
        safe_accounts = self._store.list_safe()
        with self._cache_lock:
            accounts_summary = []
            total_unread = 0
            for account in safe_accounts:
                entry = self._cache["accounts"].get(account["id"], self._empty_account_cache())
                recent = [
                    {
                        "subject": e["subject"],
                        "sender_name": e["sender_name"],
                        "sender_email": e["sender_email"],
                        "date": e["date"],
                        "unread": e["unread"],
                        "snippet": e["snippet"],
                    }
                    for e in entry.get("emails", [])[:10]
                ]
                accounts_summary.append({
                    "id": account["id"],
                    "label": account["label"],
                    "provider": account["provider"],
                    "unread_count": entry.get("unread_count", 0),
                    "last_poll": entry.get("last_poll"),
                    "last_error": entry.get("last_error"),
                    "recent": recent,
                })
                total_unread += entry.get("unread_count", 0)
            return {"accounts": accounts_summary, "total_unread": total_unread}


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    manager = EmailManager()
    manager.initialize()

    print(f"\nAccounts connected: {manager.list_accounts_safe()}")

    print("\nRunning a synchronous poll cycle (refresh_now)…")
    manager.refresh_now()

    summary = manager.get_summary()
    print(f"\nSummary:\n{json.dumps(summary, indent=2)}")

    print("\nClaiming any pending announcement…")
    first_claim = manager.claim_announcement()
    print(f"First claim: {first_claim!r}")
    second_claim = manager.claim_announcement()
    print(f"Second claim (should be None - already claimed): {second_claim!r}")
    assert second_claim is None

    if config.ANTHROPIC_API_KEY:
        print("\nClassifying a sample utterance…")
        print(manager.classify_intent("do I have any new emails?"))
    else:
        print("\nSkipping classify_intent smoke - ANTHROPIC_API_KEY not set.")
