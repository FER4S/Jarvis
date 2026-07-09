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

### 1. Add IMAP Account
- **URL:** `POST /email/accounts/imap`
- **Auth:** Requires `Authorization: Bearer <token>` (see [Authentication](#authentication-breaking-change)); returns `401` otherwise.
- **Description:** Adds a Hostinger-style (or any standard) IMAP account. The connection is validated live before anything is stored — on failure, nothing is saved.
- **Request Body:**
  ```json
  {
    "label": "Hostinger Support",
    "host": "imap.hostinger.com",
    "port": 993,
    "username": "support@example.com",
    "password": "the-mailbox-password",
    "use_ssl": true
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
- **Auth:** Requires `Authorization: Bearer <token>`; returns `401` otherwise. Returns `503` if `GOOGLE_OAUTH_CLIENT_ID`/`GOOGLE_OAUTH_CLIENT_SECRET` aren't configured.
- **Description:** Returns a Google consent URL to open in a browser (requests `gmail.readonly` and `gmail.send` scopes — send is provisioned for a future feature and unused today; Jarvis cannot send email yet). Mints a short-lived, single-use `state` nonce that the callback below requires.
- **Example Response:**
  ```json
  {
    "url": "https://accounts.google.com/o/oauth2/auth?...&state=...",
    "expires_in": 600
  }
  ```

### 3. Gmail — OAuth Callback
- **URL:** `GET /email/accounts/gmail/oauth-callback`
- **Auth:** Not bearer-protected — see the [Authentication](#authentication-breaking-change) table above.
- **Description:** Google redirects the boss's browser here after consent, with `code`/`state` query params (or `error` on cancellation). Exchanges the code for a refresh token, looks up the account's email address, and stores the account. Returns a small human-readable HTML page — there's nothing for a frontend to parse; the boss just closes the tab.
- **Note:** If Google doesn't return a refresh token (can happen on a repeat consent without revoking prior access first), nothing is stored and the page explains how to remove Jarvis's access at `myaccount.google.com/permissions` and try again.

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
- **Description:** Unread counts and recent subjects/senders per connected account. Answered from the local cache populated by the background poller — not a live fetch.
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
