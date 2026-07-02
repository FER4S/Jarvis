# ─────────────────────────────────────────────
#  api/server.py – FastAPI application factory
#  Exposes Jarvis pipeline and WebSocket events
# ─────────────────────────────────────────────

import asyncio
import threading
from typing import Dict, Any, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from core.assistant import JarvisAssistant

# Global singleton assistant
assistant = JarvisAssistant()

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

# Allow CORS for local Electron frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
async def health_check() -> Dict[str, str]:
    """Simple liveness probe for the Electron frontend."""
    return {"status": "ok", "service": "jarvis"}


@app.get("/status", tags=["state"])
async def get_status() -> Dict[str, Any]:
    """Returns the current assistant state."""
    return {
        "running": getattr(assistant, "_running", False),
        "state": assistant.get_state().value
    }


@app.post("/start", tags=["control"])
async def start_assistant() -> Dict[str, str]:
    """Starts the assistant pipeline in a background thread."""
    if getattr(assistant, "_running", False):
        return {"status": "already running"}
    
    logger.info("Starting Jarvis Assistant from API request...")
    threading.Thread(target=assistant.run, name="jarvis-main-loop", daemon=True).start()
    return {"status": "started"}


@app.post("/stop", tags=["control"])
async def stop_assistant() -> Dict[str, str]:
    """Stops the assistant cleanly."""
    if not getattr(assistant, "_running", False):
        return {"status": "already stopped"}
    
    logger.info("Stopping Jarvis Assistant from API request...")
    assistant.stop()
    return {"status": "stopped"}


@app.websocket("/events")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint that receives real-time state transitions from the assistant.
    """
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
