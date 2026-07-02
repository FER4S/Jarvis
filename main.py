# ─────────────────────────────────────────────
#  main.py – Jarvis entry point
# ─────────────────────────────────────────────

import sys
import threading
import uvicorn
from loguru import logger

import config
from api.server import app, assistant  # FastAPI app and global assistant


def configure_logger() -> None:
    """Set up Loguru with the level and optional file sink from config."""
    logger.remove()  # Remove the default stderr handler
    logger.add(
        sys.stderr,
        level=config.LOG_LEVEL,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
    )
    if config.LOG_FILE:
        logger.add(
            config.LOG_FILE,
            level=config.LOG_LEVEL,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )


def main() -> None:
    configure_logger()

    logger.info("─" * 50)
    logger.success("✅  Jarvis is ready.")
    logger.info(f"    Model  : {config.CLAUDE_MODEL}")
    logger.info(f"    STT    : faster-whisper ({config.WHISPER_MODEL_SIZE} / {config.WHISPER_DEVICE})")
    logger.info(f"    TTS    : Kokoro-82M ({config.TTS_VOICE})")
    logger.info(f"    Server : http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    logger.info("─" * 50)

    # Start the voice pipeline automatically on launch in a background thread
    logger.info("Starting Jarvis Assistant background thread...")
    threading.Thread(target=assistant.run, name="jarvis-main-loop", daemon=True).start()

    uvicorn.run(
        app,
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=config.SERVER_RELOAD,
        log_level=config.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
