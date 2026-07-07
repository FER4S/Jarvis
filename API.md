# Jarvis API Reference

The Jarvis backend exposes a FastAPI server that the frontend can use to control the assistant and receive real-time state updates.

## Authentication (BREAKING CHANGE)

`POST /start`, `POST /stop`, and the `/events` WebSocket now require a shared secret token. Requests without it are rejected — there is no unauthenticated fallback (an unset token means those endpoints reject **everything**).

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
