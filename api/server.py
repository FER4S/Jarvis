# ─────────────────────────────────────────────
#  api/server.py – FastAPI application factory
#  Exposes Jarvis pipeline and WebSocket events
# ─────────────────────────────────────────────

import asyncio
import secrets
import threading
from typing import Dict, Any, Optional, Set
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from core.assistant import JarvisAssistant
import config

# Global singleton assistant
assistant = JarvisAssistant()

# ── API token auth ────────────────────────────────────────────────────────────
# /start, /stop and /events carry control of the assistant and the live
# transcription stream — anything on this machine (including any web page in a
# browser: CORS does not apply to WebSockets) could reach them otherwise.
# Auth fails CLOSED: no configured token means every protected request is
# rejected, not silently allowed.

if not config.JARVIS_API_TOKEN:
    logger.warning(
        "JARVIS_API_TOKEN is not set — /start, /stop and /events will reject "
        "all requests until it is added to .env (see .env.example)."
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
    # Startup: Start the background broadcaster task
    task = asyncio.create_task(broadcast_events())
    yield
    # Shutdown: Cancel the broadcaster task
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
