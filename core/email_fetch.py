# ─────────────────────────────────────────────────────────────────────────────
#  core/email_fetch.py – Unified email fetching layer
#  One normalized email shape regardless of provider. IMAP (Hostinger, any
#  standard host) goes through imapclient; Gmail goes through the Gmail API
#  directly (OAuth is already required for Gmail, so the native API avoids
#  IMAP's XOAUTH2 SASL handshake and gives richer metadata for free).
#
#  Normalized email dict shape (both fetchers return exactly this):
#      {
#          "id": "<account_id>:<provider_message_id>",
#          "account_id": str, "account_label": str,
#          "subject": str, "sender_name": str, "sender_email": str,
#          "date": str,        # ISO 8601, normalized to config.TIMEZONE
#          "unread": bool,
#          "snippet": str,     # short plain-text preview
#          "body": str,        # plain-text body, HTML converted, truncated
#      }
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import base64
import email
import os
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from email import policy as email_policy
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime
from html.parser import HTMLParser
from typing import Protocol

# Ensure the project root is on sys.path so `import config` works when this
# file is run directly (e.g. `python core/email_fetch.py`).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from imapclient import IMAPClient
from loguru import logger

import config
from core.email_accounts import PROVIDER_GMAIL, PROVIDER_IMAP

# ── Tuning constants ──────────────────────────────────────────────────────────
FETCH_WINDOW_DAYS: int = 2
FETCH_MAX_MESSAGES: int = 25
BODY_CHAR_LIMIT: int = 5000
SNIPPET_CHAR_LIMIT: int = 200
SOCKET_TIMEOUT_S: float = 15.0  # bounds both poller cycles and voice-blocking "check now"


# ── Small shared helpers ──────────────────────────────────────────────────────

# Invisible characters that decode cleanly from valid UTF-8 (so
# errors="replace" on the base64 decode never touches them) but must never
# reach TTS - each is a real Unicode codepoint, not an encoding error.
# Category "Cf" (format) covers the whole zero-width / BOM / word-joiner /
# soft-hyphen / bidi-control family in one shot. An earlier explicit-list
# version of this kept missing characters (e.g. U+034F COMBINING GRAPHEME
# JOINER, which is category "Mn", not "Cf" - the classic Discord/email
# invisible that got read aloud twice), so this is now category-based plus a
# small explicit set for the non-Cf stragglers. Blanket "Cf" is safe here
# precisely because this guards the SPOKEN path: directional/format marks are
# never vocalized, and dropping them from the stored snippet is cosmetic.
_EXTRA_INVISIBLE_CHARS: frozenset[str] = frozenset({
    "͏",  # U+034F COMBINING GRAPHEME JOINER (category Mn, combining class 0)
    "᠎",  # U+180E MONGOLIAN VOWEL SEPARATOR (historically zero-width)
})
# C0/C1 control characters are stripped except these, which carry real
# meaning in body text (paragraph breaks / indentation).
_KEEP_CONTROL_CHARS: frozenset[str] = frozenset({"\n", "\t"})


def sanitize_body_text(text: str) -> str:
    """
    Strip zero-width/format characters and stray control characters from
    email text before it can ever reach TTS. Applied both at fetch time (on
    the decoded/HTML-converted body, before truncate_text/_make_snippet) and
    again at read time in the assistant right before an email is spoken - the
    latter so a body cached before this fix (or before its char set was
    broadened) still gets cleaned when actually consumed by TTS, i.e. so
    sanitization can never be bypassed by stale cached data. Idempotent;
    never raises; "" on empty input.
    """
    if not text:
        return ""
    return "".join(
        ch for ch in text
        if ch not in _EXTRA_INVISIBLE_CHARS
        and unicodedata.category(ch) != "Cf"
        and not (unicodedata.category(ch) == "Cc" and ch not in _KEEP_CONTROL_CHARS)
    )


def _rstrip_combining(text: str) -> str:
    """Drop trailing combining marks (accents/diacritics with no base
    character left after a hard truncation cut) so a sliced body doesn't end
    on an orphaned mark right before the appended ellipsis."""
    while text and unicodedata.combining(text[-1]):
        text = text[:-1]
    return text


def truncate_text(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return _rstrip_combining(text[:limit].rstrip()) + "…"


def _make_snippet(body: str, limit: int = SNIPPET_CHAR_LIMIT) -> str:
    collapsed = " ".join(body.split())
    return truncate_text(collapsed, limit)


def _decode_mime_header(value: str | None) -> str:
    """
    Decode an RFC 2047 MIME-encoded header value (e.g. Gmail API raw headers,
    which arrive NOT auto-decoded) into a plain unicode string. Safe to call
    on an already-decoded string too (no encoded words -> returned unchanged).
    Never raises - falls back to the raw value.
    """
    if not value:
        return ""
    try:
        parts = decode_header(value)
        decoded = "".join(
            part.decode(encoding or "utf-8", errors="replace") if isinstance(part, bytes) else part
            for part, encoding in parts
        )
        return decoded.strip()
    except Exception:
        return value.strip()


def _normalize_date(primary: datetime | None, fallback: datetime | None) -> str:
    """
    Normalize an email timestamp to an ISO 8601 string in config.TIMEZONE.
    Naive datetimes are assumed UTC (IMAP INTERNALDATE / Gmail internalDate
    are both effectively UTC-anchored; a missing/unparseable Date: header is
    the only realistic source of a naive datetime here). Falls back to
    `fallback`, then to "now", if `primary` is None.
    """
    target = primary or fallback or datetime.now(timezone.utc)
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    return target.astimezone(config.TIMEZONE).isoformat()


class _HTMLTextExtractor(HTMLParser):
    """Collects visible text from HTML, turning block-level tags into
    newlines and dropping script/style content entirely."""

    _BLOCK_TAGS = frozenset({"br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"})
    _SKIP_TAGS = frozenset({"script", "style", "head", "title"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)  # auto-unescapes entities in handle_data
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_startendtag(self, tag: str, attrs) -> None:
        if tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)


def html_to_text(html_content: str) -> str:
    """
    Best-effort HTML -> plain text for HTML-only email bodies (never speak
    raw HTML aloud). Never raises - falls back to "" on any parser failure.
    """
    if not html_content:
        return ""
    try:
        parser = _HTMLTextExtractor()
        parser.feed(html_content)
        text = parser.get_text()
    except Exception:
        return ""
    # Collapse runs of blank/whitespace-only lines down to a single blank line.
    lines = [line.strip() for line in text.splitlines()]
    collapsed: list[str] = []
    for line in lines:
        if line or (collapsed and collapsed[-1] != ""):
            collapsed.append(line)
    return "\n".join(collapsed).strip()


# ── Fetcher contract ──────────────────────────────────────────────────────────

class EmailFetcher(Protocol):
    def fetch_recent(
        self, account: dict, days: int = FETCH_WINDOW_DAYS, max_n: int = FETCH_MAX_MESSAGES
    ) -> dict:
        """Returns {"emails": [normalized dict, ...newest first], "unread_count": int,
        "uidvalidity": int | None}. Raises on connection/auth failure - callers
        (the poller) are responsible for catching and logging per account."""
        ...

    def search(
        self,
        account: dict,
        days: int,
        max_n: int,
        sender: str | None = None,
        subject: str | None = None,
        unread_only: bool = False,
    ) -> list[dict]:
        """
        On-demand historical search ("any email from two weeks ago?") over an
        arbitrary lookback window. Filters are pushed server-side wherever the
        provider supports it, so the max_n cap can't swallow the match.
        Returns normalized dicts, newest first. Raises on connection/auth
        failure - callers catch per account. Must never mutate mailbox state
        (nothing may get marked read).
        """
        ...

    def test_connection(self, credentials: dict) -> None:
        """Raises if the given credentials cannot connect/authenticate. Used to
        validate an account before it's stored."""
        ...


# ── IMAP (Hostinger, any standard host) ──────────────────────────────────────

def _extract_body_imap(msg) -> str:
    """Prefer text/plain, fall back to text/html -> text. `msg` is an
    email.message.EmailMessage (parsed with policy=email.policy.default)."""
    try:
        part = msg.get_body(preferencelist=("plain", "html"))
    except Exception:
        part = None
    if part is None:
        return ""
    try:
        content = part.get_content()
    except Exception:
        return ""
    if part.get_content_type() == "text/html":
        return html_to_text(content)
    return content or ""


class ImapFetcher:
    """
    IMAP fetcher via imapclient (chosen over raw imaplib: typed parsed
    responses, correct UID-vs-sequence-number handling, transparent literal/
    encoding handling - raw imaplib hands back undocumented byte-tuple soup).
    Opens a fresh connection per call; polling happens every few minutes, so
    a persistent connection isn't worth the idle-timeout/reconnect complexity.
    """

    def fetch_recent(
        self, account: dict, days: int = FETCH_WINDOW_DAYS, max_n: int = FETCH_MAX_MESSAGES
    ) -> dict:
        creds = account["credentials"]
        account_id = account["id"]
        account_label = account["label"]

        with IMAPClient(
            creds["host"],
            port=creds.get("port", 993),
            ssl=creds.get("use_ssl", True),
            timeout=SOCKET_TIMEOUT_S,
            use_uid=True,
        ) as client:
            client.login(creds["username"], creds["password"])
            # readonly=True is load-bearing: fetching BODY[] must never set \Seen.
            select_info = client.select_folder("INBOX", readonly=True)
            uidvalidity = select_info.get(b"UIDVALIDITY")

            since_date = (datetime.now(timezone.utc) - timedelta(days=days)).date()
            uids = client.search(["SINCE", since_date])
            unread_count = len(client.search(["UNSEEN"]))

            # UIDs ascend with delivery order - cap to the newest max_n so a
            # huge inbox's first baseline poll doesn't fetch thousands of msgs.
            uids = sorted(uids)[-max_n:] if uids else []

            emails: list[dict] = []
            if uids:
                response = client.fetch(uids, ["FLAGS", "INTERNALDATE", "BODY.PEEK[]"])
                for uid, data in response.items():
                    try:
                        emails.append(self._parse_message(data, uid, account_id, account_label))
                    except Exception as exc:
                        logger.warning(
                            f"Skipping unparseable IMAP message uid={uid} on '{account_label}': {exc}"
                        )

        emails.sort(key=lambda e: e["date"], reverse=True)
        return {"emails": emails, "unread_count": unread_count, "uidvalidity": uidvalidity}

    def search(
        self,
        account: dict,
        days: int,
        max_n: int,
        sender: str | None = None,
        subject: str | None = None,
        unread_only: bool = False,
    ) -> list[dict]:
        creds = account["credentials"]
        account_id = account["id"]
        account_label = account["label"]

        since_date = (datetime.now(timezone.utc) - timedelta(days=days)).date()
        criteria: list = ["SINCE", since_date]
        if sender:
            criteria.extend(["FROM", sender])
        if subject:
            criteria.extend(["SUBJECT", subject])
        if unread_only:
            criteria.append("UNSEEN")

        with IMAPClient(
            creds["host"],
            port=creds.get("port", 993),
            ssl=creds.get("use_ssl", True),
            timeout=SOCKET_TIMEOUT_S,
            use_uid=True,
        ) as client:
            client.login(creds["username"], creds["password"])
            # readonly=True is load-bearing: fetching BODY[] must never set \Seen.
            client.select_folder("INBOX", readonly=True)

            local_filter = False
            try:
                # Non-ASCII FROM/SUBJECT values (e.g. Arabic names) need an
                # explicit charset on the SEARCH command.
                needs_charset = any(v and not v.isascii() for v in (sender, subject))
                uids = client.search(criteria, charset="UTF-8" if needs_charset else None)
            except Exception as exc:
                # Some servers reject CHARSET searches (or specific criteria).
                # Fall back to a SINCE-only search and filter locally on the
                # parsed results. Limitation: the local filter only sees the
                # newest max_n messages in the window, so a match deeper in a
                # busy inbox can be missed on such servers.
                logger.warning(
                    f"IMAP filtered search failed on '{account_label}' ({exc}) - "
                    f"retrying unfiltered and filtering locally."
                )
                fallback_criteria: list = ["SINCE", since_date]
                if unread_only:
                    fallback_criteria.append("UNSEEN")
                uids = client.search(fallback_criteria)
                local_filter = True

            uids = sorted(uids)[-max_n:] if uids else []

            emails: list[dict] = []
            if uids:
                response = client.fetch(uids, ["FLAGS", "INTERNALDATE", "BODY.PEEK[]"])
                for uid, data in response.items():
                    try:
                        emails.append(self._parse_message(data, uid, account_id, account_label))
                    except Exception as exc:
                        logger.warning(
                            f"Skipping unparseable IMAP message uid={uid} on '{account_label}': {exc}"
                        )

        if local_filter:
            if sender:
                sender_lower = sender.lower()
                emails = [
                    e for e in emails
                    if sender_lower in e["sender_name"].lower() or sender_lower in e["sender_email"].lower()
                ]
            if subject:
                subject_lower = subject.lower()
                emails = [e for e in emails if subject_lower in e["subject"].lower()]

        emails.sort(key=lambda e: e["date"], reverse=True)
        return emails

    @staticmethod
    def _parse_message(data: dict, uid: int, account_id: str, account_label: str) -> dict:
        raw = data[b"BODY[]"]
        flags = data.get(b"FLAGS", ())
        internaldate = data.get(b"INTERNALDATE")

        msg = email.message_from_bytes(raw, policy=email_policy.default)

        subject = _decode_mime_header(msg.get("Subject", ""))
        sender_name, sender_email = parseaddr(msg.get("From", ""))
        sender_name = _decode_mime_header(sender_name) or sender_email

        try:
            date_header = parsedate_to_datetime(msg.get("Date"))
        except Exception:
            date_header = None

        body = truncate_text(sanitize_body_text(_extract_body_imap(msg)), BODY_CHAR_LIMIT)

        return {
            "id": f"{account_id}:{uid}",
            "account_id": account_id,
            "account_label": account_label,
            "subject": subject or "(no subject)",
            "sender_name": sender_name or sender_email or "(unknown sender)",
            "sender_email": sender_email,
            "date": _normalize_date(date_header, internaldate),
            "unread": b"\\Seen" not in flags,
            "snippet": _make_snippet(body),
            "body": body,
        }

    def test_connection(self, credentials: dict) -> None:
        """Raises on any connection/login failure - used to validate an
        account before it's stored."""
        with IMAPClient(
            credentials["host"],
            port=credentials.get("port", 993),
            ssl=credentials.get("use_ssl", True),
            timeout=SOCKET_TIMEOUT_S,
            use_uid=True,
        ) as client:
            client.login(credentials["username"], credentials["password"])
            client.select_folder("INBOX", readonly=True)


# ── Gmail (via the Gmail API, not IMAP-over-OAuth) ───────────────────────────

def _decode_gmail_body(data_b64url: str) -> str:
    try:
        padded = data_b64url + "=" * (-len(data_b64url) % 4)
        raw = base64.urlsafe_b64decode(padded)
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _walk_gmail_parts(payload: dict) -> tuple[str, str]:
    """
    Depth-first search of a Gmail payload's part tree. Returns (first
    text/plain body found, first text/html body found) - either "" if absent.
    """
    mime_type = payload.get("mimeType", "")
    body_data = (payload.get("body") or {}).get("data")

    plain = html_text = ""
    if body_data and mime_type == "text/plain":
        plain = _decode_gmail_body(body_data)
    elif body_data and mime_type == "text/html":
        html_text = _decode_gmail_body(body_data)

    for part in payload.get("parts") or []:
        child_plain, child_html = _walk_gmail_parts(part)
        plain = plain or child_plain
        html_text = html_text or child_html

    return plain, html_text


def _extract_body_gmail(payload: dict) -> str:
    plain, html_text = _walk_gmail_parts(payload)
    if plain:
        return plain
    if html_text:
        return html_to_text(html_text)
    return ""


class GmailFetcher:
    """
    Gmail fetcher via the Gmail API (not IMAP-over-OAuth) - OAuth is already
    required for Gmail, so the native API avoids IMAP's XOAUTH2 SASL handshake
    and gives unread state / labels directly instead of IMAP flag juggling.
    """

    def fetch_recent(
        self, account: dict, days: int = FETCH_WINDOW_DAYS, max_n: int = FETCH_MAX_MESSAGES
    ) -> dict:
        creds = account["credentials"]
        account_id = account["id"]
        account_label = account["label"]
        service = self._build_service(creds)

        list_response = (
            service.users()
            .messages()
            .list(userId="me", q=f"in:inbox newer_than:{days}d", maxResults=max_n)
            .execute()
        )
        message_refs = list_response.get("messages", [])

        emails: list[dict] = []
        for ref in message_refs:
            try:
                full = (
                    service.users()
                    .messages()
                    .get(userId="me", id=ref["id"], format="full")
                    .execute()
                )
                emails.append(self._parse_message(full, account_id, account_label))
            except Exception as exc:
                logger.warning(
                    f"Skipping unparseable Gmail message id={ref.get('id')} on '{account_label}': {exc}"
                )

        try:
            inbox_label = service.users().labels().get(userId="me", id="INBOX").execute()
            unread_count = int(inbox_label.get("messagesUnread", 0))
        except Exception as exc:
            logger.warning(f"Could not read INBOX unread count for '{account_label}': {exc}")
            unread_count = sum(1 for e in emails if e["unread"])

        emails.sort(key=lambda e: e["date"], reverse=True)
        return {"emails": emails, "unread_count": unread_count, "uidvalidity": None}

    def search(
        self,
        account: dict,
        days: int,
        max_n: int,
        sender: str | None = None,
        subject: str | None = None,
        unread_only: bool = False,
    ) -> list[dict]:
        creds = account["credentials"]
        account_id = account["id"]
        account_label = account["label"]
        service = self._build_service(creds)

        # Filters ride in the Gmail query so the maxResults cap can't swallow
        # the match. Note Gmail's subject: matches whole tokens, not arbitrary
        # substrings - acceptable for spoken topic cues. Embedded quotes are
        # stripped so a value can't break out of its quoted term.
        q_parts = [f"in:inbox newer_than:{days}d"]
        if sender:
            cleaned = sender.replace('"', "")
            q_parts.append(f'from:"{cleaned}"')
        if subject:
            cleaned = subject.replace('"', "")
            q_parts.append(f'subject:"{cleaned}"')
        if unread_only:
            q_parts.append("is:unread")

        list_response = (
            service.users()
            .messages()
            .list(userId="me", q=" ".join(q_parts), maxResults=max_n)
            .execute()
        )
        message_refs = list_response.get("messages", [])

        emails: list[dict] = []
        for ref in message_refs:
            try:
                full = (
                    service.users()
                    .messages()
                    .get(userId="me", id=ref["id"], format="full")
                    .execute()
                )
                emails.append(self._parse_message(full, account_id, account_label))
            except Exception as exc:
                logger.warning(
                    f"Skipping unparseable Gmail message id={ref.get('id')} on '{account_label}': {exc}"
                )

        emails.sort(key=lambda e: e["date"], reverse=True)
        return emails

    @staticmethod
    def _build_service(creds: dict):
        if not config.GOOGLE_OAUTH_CLIENT_ID or not config.GOOGLE_OAUTH_CLIENT_SECRET:
            raise ValueError(
                "GOOGLE_OAUTH_CLIENT_ID/GOOGLE_OAUTH_CLIENT_SECRET are not configured - "
                "cannot refresh a Gmail access token."
            )
        credentials = Credentials(
            token=None,
            refresh_token=creds["refresh_token"],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=config.GOOGLE_OAUTH_CLIENT_ID,
            client_secret=config.GOOGLE_OAUTH_CLIENT_SECRET,
        )
        # cache_discovery=False: avoids googleapiclient's on-disk discovery
        # cache, which warns/misbehaves on some Python versions and buys
        # nothing for a locally-run assistant.
        return build("gmail", "v1", credentials=credentials, cache_discovery=False)

    @staticmethod
    def _parse_message(full: dict, account_id: str, account_label: str) -> dict:
        payload = full.get("payload", {})
        headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}

        subject = _decode_mime_header(headers.get("subject", ""))
        sender_name, sender_email = parseaddr(headers.get("from", ""))
        sender_name = _decode_mime_header(sender_name) or sender_email

        internal_dt = None
        internal_date_ms = full.get("internalDate")
        if internal_date_ms:
            try:
                internal_dt = datetime.fromtimestamp(int(internal_date_ms) / 1000, tz=timezone.utc)
            except (ValueError, OSError, OverflowError):
                internal_dt = None

        body = truncate_text(sanitize_body_text(_extract_body_gmail(payload)), BODY_CHAR_LIMIT)
        if not body:
            body = sanitize_body_text(_decode_mime_header(full.get("snippet", "")))

        return {
            "id": f"{account_id}:{full.get('id')}",
            "account_id": account_id,
            "account_label": account_label,
            "subject": subject or "(no subject)",
            "sender_name": sender_name or sender_email or "(unknown sender)",
            "sender_email": sender_email,
            "date": _normalize_date(internal_dt, None),
            "unread": "UNREAD" in (full.get("labelIds") or []),
            "snippet": _make_snippet(body),
            "body": body,
        }

    def test_connection(self, credentials: dict) -> None:
        """Raises on any auth/refresh failure - used to validate an account
        before it's stored, and reused when re-checking a stored account."""
        service = self._build_service(credentials)
        service.users().getProfile(userId="me").execute()


# ── Registry (extensibility point for future provider types) ────────────────

PROVIDER_FETCHERS: dict[str, EmailFetcher] = {
    PROVIDER_IMAP: ImapFetcher(),
    PROVIDER_GMAIL: GmailFetcher(),
}


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    print("\n--- html_to_text ---")
    sample_html = (
        "<html><body><p>Hi &amp; welcome!</p><br><div>Second line</div>"
        "<script>evil()</script></body></html>"
    )
    text = html_to_text(sample_html)
    print(repr(text))
    assert "evil()" not in text
    assert "Hi & welcome!" in text
    print("OK: entities decoded, script content stripped.")

    print("\n--- date normalization ---")
    naive = datetime(2026, 7, 7, 12, 0, 0)
    normalized = _normalize_date(naive, None)
    print(f"Naive -> {normalized}")
    assert normalized
    print(f"No date at all -> {_normalize_date(None, None)} (falls back to now)")

    print("\n--- optional live IMAP smoke (set SMOKE_IMAP_HOST to enable) ---")
    smoke_host = os.environ.get("SMOKE_IMAP_HOST")
    if smoke_host:
        smoke_creds = {
            "host": smoke_host,
            "port": int(os.environ.get("SMOKE_IMAP_PORT", "993")),
            "username": os.environ.get("SMOKE_IMAP_USER", ""),
            "password": os.environ.get("SMOKE_IMAP_PASS", ""),
            "use_ssl": True,
        }
        fetcher = ImapFetcher()
        fetcher.test_connection(smoke_creds)
        print("IMAP test_connection OK.")
        result = fetcher.fetch_recent({"id": "smoke", "label": "Smoke", "credentials": smoke_creds})
        print(f"Fetched {len(result['emails'])} email(s), unread_count={result['unread_count']}")
        for e in result["emails"][:5]:
            status = "UNREAD" if e["unread"] else "read"
            print(f"  - [{status}] {e['subject']!r} from {e['sender_name']} ({e['date']})")

        print("\n--- deep search smoke (14 days) ---")
        found = fetcher.search(
            {"id": "smoke", "label": "Smoke", "credentials": smoke_creds}, days=14, max_n=30
        )
        print(f"Deep search found {len(found)} email(s) in the last 14 days:")
        for e in found[:5]:
            status = "UNREAD" if e["unread"] else "read"
            print(f"  - [{status}] {e['subject']!r} from {e['sender_name']} ({e['date']})")
    else:
        print("Skipped (no SMOKE_IMAP_HOST/PORT/USER/PASS set).")
