# Jarvis API Reference

The Jarvis backend exposes a FastAPI server that the frontend can use to control the assistant and receive real-time state updates.

## Authentication (BREAKING CHANGE)

`POST /start`, `POST /stop`, the `/events` WebSocket, and all `/email/*` endpoints (except the Gmail OAuth callback, which is protected differently — see below) now require a shared secret token. Requests without it are rejected — there is no unauthenticated fallback (an unset token means those endpoints reject **everything**).

**Setup (once):** generate a token and put it in the backend's `.env`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
# .env:
# JARVIS_API_TOKEN=<the generated value>
```

The frontend must be configured with the same value and present it as follows:

| Surface | How to send the token | On failure |
|---|---|---|
| `POST /start`, `POST /stop` | HTTP header `Authorization: Bearer <token>` | `401 {"detail": "Missing or invalid API token."}` |
| `WS /events` | Query param: `ws://localhost:8000/events?token=<token>` (browsers can't set headers on WebSocket handshakes) | Handshake rejected (close code 1008 / HTTP 403) |
| All `/email/*` endpoints, except the one below | HTTP header `Authorization: Bearer <token>` | `401 {"detail": "Missing or invalid API token."}` |
| `GET /email/accounts/gmail/oauth-callback` | Not bearer-protected — Google's redirect is a plain browser GET that can't carry the header. Instead requires the one-time `state` query param minted by the (bearer-protected) `GET /email/accounts/gmail/oauth-url` — so an unset token still blocks the whole OAuth flow. | `403` (HTML page: invalid, reused, or expired link) |
| `GET /health`, `GET /status` | No token required | — |

## REST Endpoints

### 1. Health Check
- **URL:** `GET /health`
- **Description:** Simple liveness probe to verify the server is running.
- **Example Response:**
  ```json
  {
    "status": "ok",
    "service": "jarvis"
  }
  ```

### 2. Status
- **URL:** `GET /status`
- **Description:** Returns whether the assistant pipeline is running and its current state.
- **Note:** `running` is `true` for the entire pipeline lifetime — from the moment startup begins (model loading, first-run onboarding) until shutdown fully completes — not only while the wake-word loop is listening. A `POST /start` sent immediately after `POST /stop` may therefore return `"already running"` while the previous run winds down; poll `/status` until `running` is `false`, then retry.
- **Note:** `error` is `null` when healthy. When a component has died unrecoverably (currently: the wake-word microphone failed to open, or died mid-stream), it holds a short human-readable reason — the process is up but the assistant cannot hear "Hey Jarvis" until the problem is fixed. It clears automatically the next time the detector starts successfully.
- **Example Response:**
  ```json
  {
    "running": true,
    "state": "idle",
    "error": null
  }
  ```

### 3. Start Assistant
- **URL:** `POST /start`
- **Auth:** Requires `Authorization: Bearer <token>` (see [Authentication](#authentication-breaking-change)); returns `401` otherwise.
- **Description:** Starts the assistant pipeline in a background thread.
- **Example Response:**
  ```json
  {
    "status": "started"
  }
  ```

### 4. Stop Assistant
- **URL:** `POST /stop`
- **Auth:** Requires `Authorization: Bearer <token>` (see [Authentication](#authentication-breaking-change)); returns `401` otherwise.
- **Description:** Stops the assistant pipeline cleanly.
- **Example Response:**
  ```json
  {
    "status": "stopped"
  }
  ```

## Email Endpoints

Connects and manages email accounts (Hostinger-style IMAP and Gmail via OAuth), and answers dashboard/settings-UI queries. All of these are backed by a background poller (default every `EMAIL_POLL_INTERVAL_S` = 240s) that also drives Jarvis's proactive spoken "you've got new mail" nudge — see the [WebSocket Events](#websocket-events) note below.

**Scope of this API.** These eight endpoints cover *account management*, a *dashboard summary*, and one optional assist for the voice send flow. Everything else about email — listing, reading, searching history, summarizing, "anything need my attention?", and **sending** — happens **by voice**, and has no REST equivalent.

**Sending.** Jarvis can send a new email, and reply to one it just read, by voice. There is deliberately **no endpoint that sends an email**: the only way a message leaves the machine is for the drafted email to be read aloud in full and then explicitly approved out loud. CC/BCC, attachments, multiple recipients, and reply threading (`In-Reply-To`/`References`) are not implemented. The one send-adjacent endpoint, `POST /email/pending-contact`, supplies a *recipient's address* when Jarvis asks for one — it never sends anything by itself.

**The poller is independent of the voice pipeline.** It starts with the server and keeps running across `POST /stop`, so `GET /email/summary` stays fresh even while the assistant is stopped. Only shutting the server down stops it.

### 1. Add IMAP Account
- **URL:** `POST /email/accounts/imap`
- **Auth:** Requires `Authorization: Bearer <token>` (see [Authentication](#authentication-breaking-change)); returns `401` otherwise.
- **Description:** Adds a Hostinger-style (or any standard) IMAP account. The IMAP connection is validated live before anything is stored — on failure, nothing is saved.
- **Outgoing mail (optional).** The `smtp_*` fields are optional. Left out, the outgoing host is derived from `host` at send time (`imap.hostinger.com` → `smtp.hostinger.com`, port 465, implicit TLS), so an account added before sending existed can send with no changes and no re-add. When `smtp_host` **is** supplied, the SMTP login is also validated live before storing; when it isn't, SMTP is never probed here, so adding a read-only mailbox can't fail on an outgoing server it will never use.
- **Request Body:**
  ```json
  {
    "label": "Hostinger Support",
    "host": "imap.hostinger.com",
    "port": 993,
    "username": "support@example.com",
    "password": "the-mailbox-password",
    "use_ssl": true,

    "smtp_host": "smtp.hostinger.com",
    "smtp_port": 465,
    "smtp_use_ssl": true
  }
  ```
- **Example Response:**
  ```json
  {
    "account": {
      "id": "3f9a1c2b4e5d4f6a8b9c0d1e2f3a4b5c",
      "label": "Hostinger Support",
      "provider": "imap",
      "created_at": "2026-07-07T09:15:00+03:00"
    }
  }
  ```
- **Error Response (400):** `{"detail": "Login failed — check the username and password."}` — the raw server error is never echoed back or logged.

### 2. Gmail — Get OAuth Consent URL
- **URL:** `GET /email/accounts/gmail/oauth-url`
- **Auth:** Requires `Authorization: Bearer <token>`; returns `401` otherwise.
- **Description:** Returns a Google consent URL to open in a browser (requests `gmail.readonly` and `gmail.send` scopes — both are now used: `gmail.send` backs the voice send flow, and because it was requested from the first version of this endpoint, already-connected accounts never needed to re-consent). Mints a short-lived, single-use `state` nonce that the callback below requires.
- **Example Response:**
  ```json
  {
    "url": "https://accounts.google.com/o/oauth2/auth?...&state=...",
    "expires_in": 600
  }
  ```
- **Error Responses:**
  - `503` — `GOOGLE_OAUTH_CLIENT_ID`/`GOOGLE_OAUTH_CLIENT_SECRET` aren't configured in the backend `.env`. Surface this as "Gmail sign-in isn't set up on this machine"; it is not a user error and retrying won't help.
  - `500` — the server built a consent URL that was missing its scopes and refused to hand it out (a broken `google-auth-oauthlib`/`oauthlib` install). Never expected in a healthy deployment; treat as a backend bug, not a retry.

- **Notes for the frontend — how to use the returned `url` (this trips people up):**
  - **Open it verbatim, don't rebuild it.** It's ~400 characters. Truncating, re-encoding, or manually re-adding query params produces Google's confusing `Missing required parameter: scope` / `invalid client_id` errors. Hand the whole string to the browser (`shell.openExternal(url)` in Electron).
  - **The `state` nonce is in-memory, single-use, and expires in `expires_in` seconds (600).** It lives in the running server process. That means: a URL is only completable against the same backend process that minted it; **restarting the backend invalidates every pending consent URL**; and a URL can be completed exactly once. So fetch a fresh `url` at the moment the boss clicks "Connect Gmail" — never cache one, never persist one across a backend restart, and don't reuse one after a failed attempt (fetch a new one). A stale/reused/expired nonce is what produces the `403` page on the callback.
  - This nonce is also the callback's **only** protection (Google's redirect is a plain browser GET that can't carry the bearer header). If `JARVIS_API_TOKEN` is unset, no nonce can be minted, so the entire Gmail flow correctly fails closed.

### 3. Gmail — OAuth Callback
- **URL:** `GET /email/accounts/gmail/oauth-callback`
- **Auth:** Not bearer-protected — see the [Authentication](#authentication-breaking-change) table above.
- **Description:** Google redirects the boss's browser here after consent, with `code`/`state` query params (or `error` on cancellation). Exchanges the code for a refresh token, looks up the account's email address, and stores the account. Returns a small human-readable HTML page — there's nothing for a frontend to parse; the boss just closes the tab.
- **Note:** If Google doesn't return a refresh token (can happen on a repeat consent without revoking prior access first), nothing is stored and the page explains how to remove Jarvis's access at `myaccount.google.com/permissions` and try again.
- **Note:** The `403` page means the `state` nonce was invalid, already used, or older than 10 minutes — see the bullets above. The fix is always "fetch a fresh consent URL", never "retry this link".
- **Backend setup gotcha:** the **Gmail API must be enabled** in the Google Cloud project that owns the OAuth client (`console.cloud.google.com` → APIs & Services → Enable APIs → Gmail API). Consent will succeed and the callback will then fail on the token exchange / profile lookup with a `400`; the server log line reads `Gmail OAuth exchange failed: HttpError: ...`. Nothing is stored.
- **Note:** A newly connected account is usable **immediately, with no backend restart** — the voice assistant and `GET /email/summary` both see it, and its mail lands in the cache within a few seconds (the poller is woken on connect).

### 4. List Email Accounts
- **URL:** `GET /email/accounts`
- **Auth:** Requires `Authorization: Bearer <token>`; returns `401` otherwise.
- **Description:** Lists connected accounts — label and provider only. Credentials/tokens are never included in this or any other response.
- **Example Response:**
  ```json
  {
    "accounts": [
      {
        "id": "3f9a1c2b4e5d4f6a8b9c0d1e2f3a4b5c",
        "label": "Hostinger Support",
        "provider": "imap",
        "created_at": "2026-07-07T09:15:00+03:00"
      },
      {
        "id": "7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f",
        "label": "boss@gmail.com",
        "provider": "gmail_oauth",
        "created_at": "2026-07-07T09:20:00+03:00"
      }
    ]
  }
  ```

### 5. Delete Email Account
- **URL:** `DELETE /email/accounts/{account_id}`
- **Auth:** Requires `Authorization: Bearer <token>`; returns `401` otherwise.
- **Description:** Removes a connected account and its cached emails. Returns `404` if the id doesn't exist.
- **Example Response:**
  ```json
  {
    "status": "deleted"
  }
  ```

### 6. Email Summary (Dashboard)
- **URL:** `GET /email/summary`
- **Auth:** Requires `Authorization: Bearer <token>`; returns `401` otherwise.
- **Description:** Unread counts and recent subjects/senders per connected account. Answered from the local cache populated by the background poller — not a live fetch, so it returns instantly and is safe to poll (every 30–60s is plenty; the backend refreshes every `EMAIL_POLL_INTERVAL_S` = 240s by default).
- **Important — `unread_count` and `recent` do not describe the same set of emails.** Don't build UI that implies they do (e.g. "2,922 unread" above a list of 4 items reading as "here they are"):
  - `unread_count` is the **provider's total unread count** for the whole inbox (Gmail's `INBOX` unread label). It can be in the thousands for a neglected mailbox.
  - `recent` is at most **10 items drawn from the local cache**, and the cache only ever holds the **last 2 days**, up to 25 messages per account — unread *and* read.
  - So `len(recent)` is unrelated to `unread_count`, `recent` contains already-read mail, and most unread mail is not in `recent` at all. A big count with a short list is correct, not a sync bug. Label them separately (e.g. "2,922 unread total" vs. "Last 2 days").
  - There is **no full-inbox sync** and no progress to display; nothing is ever "still importing". Older mail isn't cached — it's fetched live, on demand, by voice search only. There is no REST endpoint for searching older mail.
- **Per-account fields:** `last_poll` is `null` until that account's first poll completes (a few seconds after it's added). `last_error` is `null` when healthy and otherwise holds a short human-readable reason that account's last poll failed (bad password, host unreachable, revoked token) — the other accounts keep working, so surface it per-row rather than as a global banner.
- **Example Response:**
  ```json
  {
    "accounts": [
      {
        "id": "3f9a1c2b4e5d4f6a8b9c0d1e2f3a4b5c",
        "label": "Hostinger Support",
        "provider": "imap",
        "unread_count": 2,
        "last_poll": "2026-07-07T09:40:00+03:00",
        "last_error": null,
        "recent": [
          {
            "subject": "Ticket #4521: Refund request",
            "sender_name": "Jane Doe",
            "sender_email": "jane@example.com",
            "date": "2026-07-07T09:35:00+03:00",
            "unread": true,
            "snippet": "Hi, I'd like to request a refund for..."
          }
        ]
      }
    ],
    "total_unread": 2
  }
  ```

### 7. Pending Contact Address (Read)
- **URL:** `GET /email/pending-contact`
- **Auth:** Requires `Authorization: Bearer <token>`; returns `401` otherwise.
- **Description:** Whether Jarvis is currently waiting for a recipient's email address, and whose. When the voice send flow needs an address it doesn't have in memory, it asks aloud **and** opens this request, so the dashboard can offer a text box instead — email addresses are hard for speech-to-text to get right.
- **Optional by design.** The voice flow never waits on this endpoint. If the dashboard is closed, the boss simply speaks the address and Jarvis spells it back for confirmation. At most one request is open at a time, it expires after 180 seconds, and it is always closed when the conversation ends.
- **Example Response (waiting):**
  ```json
  {"pending": {"name": "Michael", "expires_in": 142}}
  ```
- **Example Response (not waiting):**
  ```json
  {"pending": null}
  ```

### 8. Pending Contact Address (Submit)
- **URL:** `POST /email/pending-contact`
- **Auth:** Requires `Authorization: Bearer <token>`; returns `401` otherwise.
- **Description:** Fulfils the open address request with a typed address. The voice loop picks it up on its next round and uses it verbatim — no spoken read-back is needed, because it wasn't transcribed. Submitting again before it's claimed replaces the value, so a typo caught in time does the right thing.
- **This does not send anything.** The drafted email is still read aloud in full and must be explicitly approved out loud before it leaves the machine. A confirmed address is also saved to that person's entry in memory, so the next email to them needs no spelling.
- **Request Body:**
  ```json
  {"email": "michael@codex.com"}
  ```
- **Example Response (202):**
  ```json
  {"status": "accepted"}
  ```
- **Error Responses:**
  - `400 {"detail": "That doesn't look like a valid email address."}` — malformed. Checked before anything else, so a request that expires mid-flight can never be reported as a bad address.
  - `409 {"detail": "Jarvis isn't waiting for a contact's email address."}` — nothing pending, or the request expired.

## WebSocket Events

- **URL:** `ws://localhost:8000/events?token=<token>`
- **Auth:** The API token must be passed as the `token` query param; a missing or wrong token rejects the handshake (close code 1008 / HTTP 403).
- **Connection:** Connect to this endpoint to receive real-time events broadcast by the assistant.

### Event Types

All events are broadcast as JSON objects containing an `"event"` field and any associated payload data.

| Event | Description | Payload Example |
|---|---|---|
| `wake_word_detected` | Fired when the wake word is detected. | `{"event": "wake_word_detected"}` |
| `listening_started` | Fired when the microphone starts recording. | `{"event": "listening_started"}` |
| `transcription` | Fired when speech-to-text finishes processing. | `{"event": "transcription", "text": "Hello Jarvis"}` |
| `llm_response` | Fired when Claude finishes generating a reply. | `{"event": "llm_response", "text": "Hello! How can I help you today?"}` |
| `speaking_started` | Fired when text-to-speech audio starts playing. | `{"event": "speaking_started"}` |
| `speaking_ended` | Fired when text-to-speech audio finishes playing. | `{"event": "speaking_ended"}` |
| `idle` | Fired when the conversation cycle resets. | `{"event": "idle"}` |
| `error` | Fired when a component dies unrecoverably (currently: the wake-word microphone failed to open or died mid-stream). The same string is available via `GET /status` as `error`. | `{"event": "error", "message": "wake word detector: microphone failed to open: ..."}` |
| `contact_email_requested` | Fired when the voice send flow needs a recipient's email address and has opened a pending request. Cue to show a text box wired to `POST /email/pending-contact`. | `{"event": "contact_email_requested", "name": "Michael"}` |
| `contact_email_resolved` | Fired when that request ends. `source` is `"voice"` (spoken and confirmed), `"typed"` (submitted from the dashboard), or `"cancelled"` (never supplied — nothing was sent). Cue to hide the text box. | `{"event": "contact_email_resolved", "source": "typed"}` |

**Note — sending email:** there are no events for drafting, confirming, or sending. The whole exchange — "here's what I've got…", "should I send it?", "sent to Michael" — flows through the ordinary `llm_response` → `speaking_started` → `speaking_ended` → `listening_started` → `transcription` sequence, so a transcript view shows it with no special handling. The two `contact_email_*` events above are the only additions, and they exist solely so the dashboard can offer typing as an alternative to speaking an address.

**Note — proactive email announcements:** no new event types were added for these. When the background email poller finds new mail and Jarvis is idle, it speaks a nudge (e.g. "Hey, you've got 3 new emails…") using the same `llm_response` → `speaking_started` → `speaking_ended` sequence as a normal reply, followed immediately by the usual `listening_started`/`transcription`/... cycle for whatever the boss says next — exactly as if "Hey Jarvis" had just been said. The only difference a frontend can key off is that this `llm_response` has **no preceding `wake_word_detected`** event. Dashboards should poll `GET /email/summary` for email state rather than relying on WebSocket events.

### Quick Start: Connecting via JavaScript

The following snippet demonstrates how a frontend application (like Electron) can connect to the WebSocket and handle events.

```javascript
const JARVIS_API_TOKEN = "<value of JARVIS_API_TOKEN from the backend .env>";
const ws = new WebSocket(`ws://localhost:8000/events?token=${JARVIS_API_TOKEN}`);

ws.onopen = () => {
    console.log("Connected to Jarvis events WebSocket.");
};

ws.onmessage = (message) => {
    const data = JSON.parse(message.data);
    
    switch(data.event) {
        case "wake_word_detected":
            console.log("Wake word detected! Getting ready...");
            break;
            
        case "listening_started":
            console.log("Jarvis is listening...");
            break;
            
        case "transcription":
            console.log("You said:", data.text);
            break;
            
        case "llm_response":
            console.log("Jarvis says:", data.text);
            break;
            
        case "speaking_started":
            console.log("Jarvis is speaking...");
            break;
            
        case "speaking_ended":
            console.log("Jarvis finished speaking.");
            break;
            
        case "idle":
            console.log("Jarvis is now idle and waiting for the wake word.");
            break;
            
        case "error":
            console.error("Jarvis component failure:", data.message);
            break;
            
        default:
            console.log("Unknown event received:", data);
    }
};

ws.onclose = () => {
    console.log("Disconnected from Jarvis.");
};

ws.onerror = (error) => {
    console.error("WebSocket error:", error);
};
```
