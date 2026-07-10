# ─────────────────────────────────────────────
#  api/server.py – FastAPI application factory
#  Exposes Jarvis pipeline and WebSocket events
# ─────────────────────────────────────────────

import asyncio
import html
import secrets
import smtplib
import threading
from typing import Dict, Any, Optional, Set
from contextlib import asynccontextmanager
from urllib.parse import urlparse, parse_qs

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from imapclient.exceptions import IMAPClientError
from loguru import logger
from pydantic import BaseModel, Field

from core.assistant import JarvisAssistant
from core.email_manager import OAUTH_STATE_TTL_S
from core.memory import is_valid_email
import config

# Global singleton assistant
assistant = JarvisAssistant()
# Exactly one EmailManager must exist per process - bind to the assistant's
# shared instance rather than constructing a second one (two instances would
# double-poll and race the cache file; see core/email_manager.py's module
# docstring).
email_manager = assistant.get_email_manager()

# ── API token auth ────────────────────────────────────────────────────────────
# /start, /stop and /events carry control of the assistant and the live
# transcription stream — anything on this machine (including any web page in a
# browser: CORS does not apply to WebSockets) could reach them otherwise.
# Auth fails CLOSED: no configured token means every protected request is
# rejected, not silently allowed.

if not config.JARVIS_API_TOKEN:
    logger.warning(
        "JARVIS_API_TOKEN is not set — /start, /stop, /events, and all "
        "/email/* endpoints (except the OAuth callback, which is protected "
        "by a one-time state nonce instead) will reject all requests until "
        "it is added to .env (see .env.example)."
    )


def _token_matches(presented: Optional[str]) -> bool:
    """Constant-time token comparison; False when unset/missing (fail closed)."""
    if not config.JARVIS_API_TOKEN or not presented:
        return False
    return secrets.compare_digest(
        presented.encode("utf-8"), config.JARVIS_API_TOKEN.encode("utf-8")
    )


async def require_token(authorization: Optional[str] = Header(default=None)) -> None:
    """Dependency guarding control endpoints: `Authorization: Bearer <token>`."""
    presented: Optional[str] = None
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[len("bearer "):].strip()
    if not _token_matches(presented):
        raise HTTPException(status_code=401, detail="Missing or invalid API token.")

# A set of active websocket connections for broadcasting
active_websockets: Set[WebSocket] = set()


async def broadcast_events():
    """Background task to read from assistant queue and broadcast to all websockets."""
    q = assistant.get_event_queue()
    while True:
        try:
            # Wait for an event in a thread so we don't block the asyncio event loop
            event = await asyncio.to_thread(q.get)
            
            # Broadcast to all connected clients
            dead_sockets = set()
            for ws in active_websockets:
                try:
                    await ws.send_json(event)
                except Exception:
                    dead_sockets.add(ws)
            
            # Clean up disconnected sockets
            for ws in dead_sockets:
                active_websockets.discard(ws)
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Event broadcast error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the email account store/cache, start its poller (both
    # calls are idempotent - assistant.run(), if also active, may have
    # already done this; see core/email_manager.py's module docstring), then
    # start the background event broadcaster task.
    await asyncio.to_thread(email_manager.initialize)
    email_manager.start_polling()
    task = asyncio.create_task(broadcast_events())
    yield
    # Shutdown: stop the email poller (assistant.stop() deliberately does
    # NOT do this - see EmailManager.stop_polling()'s docstring) and cancel
    # the broadcaster task.
    email_manager.stop_polling()
    task.cancel()


app = FastAPI(
    title="Jarvis API",
    description="Local backend server for the Jarvis voice AI assistant.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for the local Electron frontend. Origins stay wildcard because the
# Electron app's origin scheme (file:// → "null", or app://) isn't fixed;
# the bearer token is the actual protection (and CORS can't cover WebSockets
# anyway). Credentials are OFF — we use no cookies, and wildcard+credentials
# is the most permissive combination possible.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
async def health_check() -> Dict[str, str]:
    """Simple liveness probe for the Electron frontend."""
    return {"status": "ok", "service": "jarvis"}


@app.get("/status", tags=["state"])
async def get_status() -> Dict[str, Any]:
    """Returns the current assistant state and last component failure (if any)."""
    return {
        "running": assistant.is_active(),
        "state": assistant.get_state().value,
        "error": assistant.get_last_error(),
    }


@app.post("/start", tags=["control"], dependencies=[Depends(require_token)])
async def start_assistant() -> Dict[str, str]:
    """Starts the assistant pipeline in a background thread."""
    if assistant.is_active():
        return {"status": "already running"}
    
    logger.info("Starting Jarvis Assistant from API request...")
    threading.Thread(target=assistant.run, name="jarvis-main-loop", daemon=True).start()
    return {"status": "started"}


@app.post("/stop", tags=["control"], dependencies=[Depends(require_token)])
async def stop_assistant() -> Dict[str, str]:
    """Stops the assistant cleanly."""
    if not assistant.is_active():
        return {"status": "already stopped"}
    
    logger.info("Stopping Jarvis Assistant from API request...")
    assistant.stop()
    return {"status": "stopped"}


@app.websocket("/events")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint that receives real-time state transitions from the assistant.
    Requires the API token as a query param (ws://.../events?token=...) — browsers
    can't set headers on WebSocket handshakes, so it rides in the URL.
    """
    # Authenticate BEFORE accepting: a bad/missing token never gets a socket.
    if not _token_matches(websocket.query_params.get("token")):
        await websocket.close(code=1008)  # 1008 = policy violation
        return

    await websocket.accept()
    active_websockets.add(websocket)
    logger.info("WebSocket client connected.")
    
    try:
        while True:
            # Keep the connection alive and wait for client to disconnect
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    finally:
        active_websockets.discard(websocket)


# ── Email ──────────────────────────────────────────────────────────────────
# All endpoints below require the same JARVIS_API_TOKEN as /start and /stop,
# EXCEPT the OAuth callback: Google's redirect is a plain browser GET that
# cannot carry an Authorization header. That one is instead protected by a
# one-time state nonce minted by /oauth-url (which IS behind require_token),
# so with JARVIS_API_TOKEN unset, no nonce can ever be minted and the whole
# OAuth chain still fails closed. Every blocking call (IMAP validation,
# OAuth token exchange, account deletion) runs via asyncio.to_thread so it
# never stalls the WebSocket event broadcaster.

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    # Used by core/email_send.py's GmailSender. Requested from the very first
    # version of this flow, so every already-connected account can send
    # without re-consenting.
    "https://www.googleapis.com/auth/gmail.send",
]

_OAUTH_RESULT_HTML = """<!doctype html>
<html><head><title>Jarvis - Gmail</title></head>
<body style="font-family: sans-serif; text-align: center; padding-top: 4rem;">
<h2>{message}</h2>
</body></html>"""


class ImapAccountRequest(BaseModel):
    label: str = Field(min_length=1, max_length=60)
    host: str = Field(min_length=1)
    port: int = Field(default=993, ge=1, le=65535)
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)
    use_ssl: bool = True
    # Outgoing (SMTP) settings, all optional. Left unset, the outgoing host is
    # derived from `host` at send time (imap.x.com -> smtp.x.com:465), so an
    # account added for reading needs no changes to start sending. When
    # smtp_host IS given, the SMTP login is validated live before storing.
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = Field(default=None, ge=1, le=65535)
    smtp_use_ssl: bool = True


class ContactEmailRequest(BaseModel):
    """A recipient's address, typed into the dashboard instead of spoken."""
    email: str = Field(min_length=3, max_length=254)


def _sanitize_imap_error(exc: Exception) -> str:
    """
    Maps a test_connection() failure to a user-safe message - never echoes
    the raw exception text (which can include IMAP server banners) back in
    an API response or a log line.
    """
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return "The outgoing mail server rejected the username and password."
    if isinstance(exc, smtplib.SMTPException):
        return "Could not connect with the given SMTP settings."
    if isinstance(exc, IMAPClientError):
        return "Login failed — check the username and password."
    if isinstance(exc, OSError):  # socket timeouts, DNS failures, TLS errors, refused connections
        return "Could not connect to the server — check the host, port, and SSL setting."
    return "Could not connect with the given IMAP settings."


def _gmail_redirect_uri() -> str:
    return f"{config.GOOGLE_OAUTH_REDIRECT_BASE}/email/accounts/gmail/oauth-callback"


def _build_oauth_flow() -> Flow:
    redirect_uri = _gmail_redirect_uri()
    client_config = {
        "web": {
            "client_id": config.GOOGLE_OAUTH_CLIENT_ID,
            "client_secret": config.GOOGLE_OAUTH_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }
    flow = Flow.from_client_config(client_config, scopes=GMAIL_SCOPES)
    flow.redirect_uri = redirect_uri
    return flow


def _authorization_url_has_scopes(url: str) -> bool:
    """
    True only if the generated consent URL actually carries a non-empty
    `scope` query param containing every scope in GMAIL_SCOPES. Guards against
    ever handing Google (or the frontend) a URL that would fail with "Missing
    required parameter: scope" - a silent, confusing failure mode that a
    dependency upgrade or refactor could otherwise reintroduce.
    """
    scope_values = parse_qs(urlparse(url).query).get("scope")
    if not scope_values:
        return False
    return all(scope in scope_values[0] for scope in GMAIL_SCOPES)


@app.post("/email/accounts/imap", tags=["email"], dependencies=[Depends(require_token)])
async def add_imap_account(request: ImapAccountRequest) -> Dict[str, Any]:
    """Adds a Hostinger-style (or any standard) IMAP account. Validates the
    connection live before storing - never persists unreachable credentials.
    An explicitly-supplied smtp_host is validated too; a derived one is not
    (see EmailManager.add_imap_account)."""
    try:
        account = await asyncio.to_thread(
            email_manager.add_imap_account,
            request.label, request.host, request.port,
            request.username, request.password, request.use_ssl,
            request.smtp_host, request.smtp_port, request.smtp_use_ssl,
        )
    except Exception as exc:
        logger.warning(f"IMAP account validation failed for host '{request.host}': {type(exc).__name__}")
        raise HTTPException(status_code=400, detail=_sanitize_imap_error(exc))
    return {"account": account}


# ── Typed recipient addresses ────────────────────────────────────────────────
# When the voice send flow needs an address it doesn't have, it asks aloud and
# opens a pending request. The boss can answer by speaking it (the default) or,
# because email addresses are hard for STT to get right, by typing it here.
# This path is strictly optional: the voice flow never waits on it, and works
# identically if the dashboard is closed.

@app.get("/email/pending-contact", tags=["email"], dependencies=[Depends(require_token)])
async def get_pending_contact() -> Dict[str, Any]:
    """Whether Jarvis is currently waiting on a contact's email address, and
    for whom. Lets the dashboard show an input box in response to the
    `contact_email_requested` WebSocket event (or by polling)."""
    return {"pending": email_manager.get_contact_request()}


@app.post("/email/pending-contact", tags=["email"], status_code=202,
          dependencies=[Depends(require_token)])
async def submit_pending_contact(request: ContactEmailRequest) -> Dict[str, str]:
    """
    Fulfil the open address request with a typed address. The voice loop claims
    it on its next round. Submitting again before it's claimed replaces the
    value, so a typo caught in time does the right thing.

    400 when the address is malformed; 409 when nothing is pending (or it
    expired). Shape is checked first so the two are never confused by a request
    that expires between a check and a submit. Note this never sends anything
    by itself - the drafted email is still read back in full and explicitly
    confirmed out loud.
    """
    if not is_valid_email(request.email):
        raise HTTPException(status_code=400, detail="That doesn't look like a valid email address.")
    if not email_manager.submit_contact_email(request.email):
        raise HTTPException(
            status_code=409, detail="Jarvis isn't waiting for a contact's email address."
        )
    return {"status": "accepted"}


@app.get("/email/accounts/gmail/oauth-url", tags=["email"], dependencies=[Depends(require_token)])
async def gmail_oauth_url() -> Dict[str, Any]:
    """
    Returns the Google consent URL to open in a browser. Minting the state
    nonce here - behind require_token - is what lets the callback (which
    can't carry a bearer header) effectively inherit this endpoint's auth.
    """
    if not config.GOOGLE_OAUTH_CLIENT_ID or not config.GOOGLE_OAUTH_CLIENT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Gmail OAuth is not configured — set GOOGLE_OAUTH_CLIENT_ID/"
            "GOOGLE_OAUTH_CLIENT_SECRET in .env.",
        )
    flow = _build_oauth_flow()
    state = email_manager.mint_oauth_state()
    # prompt=consent is load-bearing: Google omits refresh_token on repeat
    # consents without it, and a refresh token is the whole point.
    auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent", state=state)
    # Fail loud and local if the URL somehow lacks its scopes, rather than
    # letting Google reject it downstream with a confusing "Missing required
    # parameter: scope". A no-op on correct output (the normal case).
    if not _authorization_url_has_scopes(auth_url):
        logger.error("Generated Gmail OAuth URL is missing scopes — refusing to return it.")
        raise HTTPException(
            status_code=500,
            detail="Failed to build a valid Google consent URL (scopes missing). "
            "Check the google-auth-oauthlib / oauthlib install.",
        )
    return {"url": auth_url, "expires_in": int(OAUTH_STATE_TTL_S)}


@app.get("/email/accounts/gmail/oauth-callback", tags=["email"])
async def gmail_oauth_callback(state: str = "", code: str = "", error: str = "") -> HTMLResponse:
    """
    Handles Google's OAuth redirect. Deliberately NOT behind require_token -
    see the "Email" section header comment above for why the state nonce
    takes its place. Never logs or renders the authorization code or any
    token; the only values ever interpolated into the response HTML are our
    own fixed strings (plus the boss's own, HTML-escaped, Gmail address).
    """
    if not email_manager.consume_oauth_state(state):
        return HTMLResponse(
            _OAUTH_RESULT_HTML.format(
                message="This link is invalid or has expired. Please try connecting your Gmail account again."
            ),
            status_code=403,
        )

    if error:
        logger.info("Gmail OAuth: consent was cancelled or denied.")
        return HTMLResponse(
            _OAUTH_RESULT_HTML.format(
                message="Connection cancelled — you can try again anytime from the settings dashboard."
            )
        )

    if not code:
        return HTMLResponse(
            _OAUTH_RESULT_HTML.format(message="No authorization code was received. Please try again."),
            status_code=400,
        )

    def _exchange_and_store() -> str:
        flow = _build_oauth_flow()
        flow.fetch_token(code=code)
        credentials = flow.credentials
        if not credentials.refresh_token:
            return (
                "Google didn't grant a long-lived connection this time. Please remove "
                "Jarvis's access at myaccount.google.com/permissions and try connecting again."
            )
        service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
        profile = service.users().getProfile(userId="me").execute()
        email_address = profile.get("emailAddress", "")
        email_manager.add_gmail_account(email_address or "Gmail", credentials.refresh_token, email_address)
        return f"Gmail account connected — {html.escape(email_address)}. You can close this tab."

    try:
        message = await asyncio.to_thread(_exchange_and_store)
    except Exception as exc:
        # Log the full error detail (type + message) for the operator: a bare
        # exception name is useless for diagnosing setup issues. Google's
        # HttpError message here is setup guidance (e.g. "Gmail API has not
        # been used in project N before or it is disabled … enable it by
        # visiting <link>"), not a secret — it carries no auth code or token.
        logger.warning(f"Gmail OAuth exchange failed: {type(exc).__name__}: {exc}")
        return HTMLResponse(
            _OAUTH_RESULT_HTML.format(message="Something went wrong connecting your Gmail account. Please try again."),
            status_code=400,
        )

    return HTMLResponse(_OAUTH_RESULT_HTML.format(message=message))


@app.get("/email/accounts", tags=["email"], dependencies=[Depends(require_token)])
async def list_email_accounts() -> Dict[str, Any]:
    """Lists connected accounts - label and provider only, never credentials."""
    return {"accounts": email_manager.list_accounts_safe()}


@app.delete("/email/accounts/{account_id}", tags=["email"], dependencies=[Depends(require_token)])
async def delete_email_account(account_id: str) -> Dict[str, str]:
    removed = await asyncio.to_thread(email_manager.delete_account, account_id)
    if not removed:
        raise HTTPException(status_code=404, detail="No email account with that id.")
    return {"status": "deleted"}


@app.get("/email/summary", tags=["email"], dependencies=[Depends(require_token)])
async def email_summary() -> Dict[str, Any]:
    """Dashboard payload: unread counts + recent subjects/senders per account."""
    return email_manager.get_summary()
