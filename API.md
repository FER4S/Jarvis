# Jarvis API Reference

The Jarvis backend exposes a FastAPI server that the frontend can use to control the assistant and receive real-time state updates.

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
- **Example Response:**
  ```json
  {
    "running": true,
    "state": "idle"
  }
  ```

### 3. Start Assistant
- **URL:** `POST /start`
- **Description:** Starts the assistant pipeline in a background thread.
- **Example Response:**
  ```json
  {
    "status": "started"
  }
  ```

### 4. Stop Assistant
- **URL:** `POST /stop`
- **Description:** Stops the assistant pipeline cleanly.
- **Example Response:**
  ```json
  {
    "status": "stopped"
  }
  ```

## WebSocket Events

- **URL:** `ws://localhost:8000/events`
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

### Quick Start: Connecting via JavaScript

The following snippet demonstrates how a frontend application (like Electron) can connect to the WebSocket and handle events.

```javascript
const ws = new WebSocket("ws://localhost:8000/events");

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
