# ─────────────────────────────────────────────────────────────────────────────
#  core/email_send.py – Unified email sending layer
#  The write-side counterpart to core/email_fetch.py, kept in its own module
#  precisely so that file's read-only invariant stays legible: nothing in
#  email_fetch.py may mutate a mailbox (select_folder(readonly=True) is
#  load-bearing there), and nothing here may fetch.
#
#  IMAP-provider accounts send over SMTP; Gmail-OAuth accounts send through the
#  Gmail API using the same refresh token and the gmail.send scope granted at
#  connect time. Both providers are reached through PROVIDER_SENDERS, mirroring
#  email_fetch.PROVIDER_FETCHERS - a new provider is a new entry, not a rewrite.
#
#  Deliberately NOT supported: CC/BCC, attachments, multiple recipients, and
#  reply threading (In-Reply-To/References/threadId). A reply is a plain new
#  message to the original sender with a "Re: " subject. All future work.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import base64
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from typing import Protocol

# Ensure the project root is on sys.path so `import config` works when this
# file is run directly (e.g. `python core/email_send.py`).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from loguru import logger

from core.email_accounts import PROVIDER_GMAIL, PROVIDER_IMAP
from core.email_fetch import build_gmail_service

# ── Tuning constants ──────────────────────────────────────────────────────────
# Module-level rather than env vars, per house convention: these only ever
# change when a provider changes, not per deployment.
SMTP_DEFAULT_SSL_PORT: int = 465      # implicit TLS (SMTPS)
SMTP_DEFAULT_STARTTLS_PORT: int = 587  # explicit TLS
SMTP_TIMEOUT_S: float = 20.0           # the boss is waiting; fail fast, don't hang the turn

# Subject prefix for replies. Compared case-insensitively before prepending, so
# replying to an already-"Re: " subject doesn't stack them.
REPLY_SUBJECT_PREFIX: str = "Re: "


def derive_smtp_host(imap_host: str) -> str:
    """
    Best-effort outgoing host for an account that only stored an incoming one.
    Virtually every provider that serves `imap.<domain>` also serves
    `smtp.<domain>` (Hostinger included), so this covers the accounts that were
    added before sending existed. Only a leading "imap." is rewritten; anything
    else is returned unchanged and will simply be tried as-is.

    An explicit credentials["smtp_host"] always wins over this guess.
    """
    host = (imap_host or "").strip()
    if host.lower().startswith("imap."):
        return "smtp." + host[len("imap."):]
    return host


def reply_subject(original_subject: str) -> str:
    """"Re: " the original subject, without stacking the prefix."""
    subject = (original_subject or "").strip()
    if not subject:
        return REPLY_SUBJECT_PREFIX.strip()
    if subject.lower().startswith("re:"):
        return subject
    return f"{REPLY_SUBJECT_PREFIX}{subject}"


def build_message(from_addr: str, to_addr: str, subject: str, body: str) -> EmailMessage:
    """
    The one place a MIME message is constructed. Plain-text UTF-8 only;
    set_content() picks the transfer encoding and charset, so a body with
    non-ASCII (or a very long line) is encoded correctly rather than mangled
    or silently rejected by the server.
    """
    message = EmailMessage()
    message["From"] = from_addr
    message["To"] = to_addr
    message["Subject"] = subject
    message.set_content(body)
    return message


# ── Sender contract ───────────────────────────────────────────────────────────

class EmailSender(Protocol):
    def send(self, account: dict, to_addr: str, subject: str, body: str) -> None:
        """
        Send one plain-text message from `account` (a full record WITH
        credentials, as returned by EmailAccountStore.get()).

        Raises on any authentication, connection, or delivery failure - never
        returns a bool, never swallows. The caller speaks the failure aloud, so
        a silent no-op would be indistinguishable from success to the boss.

        Must not read or write the poll cache, seen_ids, or announcement state.
        """
        ...

    def sender_address(self, credentials: dict) -> str:
        """The address this account sends as (the From header)."""
        ...

    def test_connection(self, credentials: dict) -> None:
        """Authenticate against the outgoing server without sending anything.
        Raises on failure. Used to validate explicitly-supplied SMTP settings
        before they are stored."""
        ...


# ── IMAP-provider accounts: SMTP ──────────────────────────────────────────────

class SmtpSender:
    """
    Sends via SMTP using the account's IMAP username/password - the same
    credential pair works for both on every host this supports (Hostinger,
    standard cPanel-style mail). Host/port come from the optional smtp_* keys
    when present and are derived from the IMAP host otherwise.
    """

    def sender_address(self, credentials: dict) -> str:
        return credentials["username"].strip()

    @staticmethod
    def _resolve_endpoint(credentials: dict) -> tuple[str, int, bool]:
        """(host, port, use_ssl). An explicitly stored value always beats a
        derived one; the port default follows whichever TLS mode is in use."""
        use_ssl = bool(credentials.get("smtp_use_ssl", True))
        host = credentials.get("smtp_host") or derive_smtp_host(credentials["host"])
        port = credentials.get("smtp_port") or (
            SMTP_DEFAULT_SSL_PORT if use_ssl else SMTP_DEFAULT_STARTTLS_PORT
        )
        return host, int(port), use_ssl

    def _connect(self, credentials: dict) -> smtplib.SMTP:
        host, port, use_ssl = self._resolve_endpoint(credentials)
        if use_ssl:
            client = smtplib.SMTP_SSL(
                host, port, timeout=SMTP_TIMEOUT_S, context=ssl.create_default_context()
            )
        else:
            client = smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT_S)
            client.starttls(context=ssl.create_default_context())
        client.login(credentials["username"], credentials["password"])
        return client

    def send(self, account: dict, to_addr: str, subject: str, body: str) -> None:
        credentials = account["credentials"]
        message = build_message(self.sender_address(credentials), to_addr, subject, body)
        with self._connect(credentials) as client:
            client.send_message(message)

    def test_connection(self, credentials: dict) -> None:
        with self._connect(credentials) as client:
            client.noop()


# ── Gmail-OAuth accounts: Gmail API ───────────────────────────────────────────

class GmailSender:
    """
    Sends through users().messages().send(). The gmail.send scope has been
    requested since the OAuth flow was first written, so every already-
    connected account can send without re-consent.
    """

    def sender_address(self, credentials: dict) -> str:
        return credentials["email_address"].strip()

    def send(self, account: dict, to_addr: str, subject: str, body: str) -> None:
        credentials = account["credentials"]
        message = build_message(self.sender_address(credentials), to_addr, subject, body)
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
        service = build_gmail_service(credentials)
        service.users().messages().send(userId="me", body={"raw": raw}).execute()

    def test_connection(self, credentials: dict) -> None:
        service = build_gmail_service(credentials)
        service.users().getProfile(userId="me").execute()


# ── Registry (extensibility point, mirrors PROVIDER_FETCHERS) ────────────────

PROVIDER_SENDERS: dict[str, EmailSender] = {
    PROVIDER_IMAP: SmtpSender(),
    PROVIDER_GMAIL: GmailSender(),
}


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Offline assertions always run. A live send happens only when all of
    # SMOKE_SMTP_HOST / _USER / _PASS / _TO are set, and goes to _TO only.
    failures = 0

    def check(label: str, condition: bool) -> None:
        global failures
        if not condition:
            failures += 1
            print(f"  FAIL  {label}")
        else:
            print(f"  ok    {label}")

    print("-- derive_smtp_host --")
    check("imap. prefix is rewritten", derive_smtp_host("imap.hostinger.com") == "smtp.hostinger.com")
    check("case-insensitive prefix", derive_smtp_host("IMAP.hostinger.com") == "smtp.hostinger.com")
    check("non-imap host untouched", derive_smtp_host("mail.example.com") == "mail.example.com")
    check("an smtp. host is untouched", derive_smtp_host("smtp.example.com") == "smtp.example.com")
    check("'imap' inside the domain is not rewritten",
          derive_smtp_host("mail.imap-host.com") == "mail.imap-host.com")
    check("empty host", derive_smtp_host("") == "")

    print("\n-- reply_subject --")
    check("prefixes a plain subject", reply_subject("Monthly KPI Report") == "Re: Monthly KPI Report")
    check("does not stack Re:", reply_subject("Re: Monthly KPI") == "Re: Monthly KPI")
    check("case-insensitive stack guard", reply_subject("RE: Budget") == "RE: Budget")
    check("empty subject", reply_subject("") == "Re:")

    print("\n-- _resolve_endpoint --")
    resolve = SmtpSender._resolve_endpoint
    check("derives host + ssl port from imap creds",
          resolve({"host": "imap.hostinger.com"}) == ("smtp.hostinger.com", 465, True))
    check("explicit smtp_host wins over derivation",
          resolve({"host": "imap.a.com", "smtp_host": "out.b.com"}) == ("out.b.com", 465, True))
    check("explicit port wins",
          resolve({"host": "imap.a.com", "smtp_port": 2525}) == ("smtp.a.com", 2525, True))
    check("starttls mode defaults to 587",
          resolve({"host": "imap.a.com", "smtp_use_ssl": False}) == ("smtp.a.com", 587, False))

    print("\n-- build_message --")
    msg = build_message("me@codex.com", "you@acme.com", "Meeting at 5pm", "Hi,\n\nLet's meet at 5pm.\n")
    check("From header", msg["From"] == "me@codex.com")
    check("To header", msg["To"] == "you@acme.com")
    check("Subject header", msg["Subject"] == "Meeting at 5pm")
    check("body round-trips", msg.get_content().strip() == "Hi,\n\nLet's meet at 5pm.")
    check("content type is text/plain", msg.get_content_type() == "text/plain")
    check("serializes to bytes", isinstance(msg.as_bytes(), bytes))

    unicode_msg = build_message("me@codex.com", "you@acme.com", "Café", "Naïve résumé — 5pm\n")
    check("non-ascii body round-trips", "résumé" in unicode_msg.get_content())
    check("non-ascii subject is encodable", isinstance(unicode_msg.as_bytes(), bytes))
    check("base64url-encodes for the Gmail API",
          isinstance(base64.urlsafe_b64encode(unicode_msg.as_bytes()).decode("ascii"), str))

    smoke = {k: os.environ.get(f"SMOKE_SMTP_{k}") for k in ("HOST", "USER", "PASS", "TO")}
    if all(smoke.values()):
        print(f"\n-- live SMTP send to {smoke['TO']} --")
        try:
            SmtpSender().send(
                {"credentials": {
                    "host": smoke["HOST"], "username": smoke["USER"], "password": smoke["PASS"],
                }},
                smoke["TO"], "Jarvis SMTP smoke test", "This is a test send from core/email_send.py.",
            )
            check("live send succeeded", True)
        except Exception as exc:
            check(f"live send succeeded (got {type(exc).__name__}: {exc})", False)
    else:
        print("\n(skipping live send — set SMOKE_SMTP_HOST/_USER/_PASS/_TO to enable)")

    print(f"\n{'FAILED' if failures else 'PASSED'} — {failures} failure(s)")
    sys.exit(1 if failures else 0)
