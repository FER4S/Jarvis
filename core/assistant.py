# ─────────────────────────────────────────────────────────────────────────────
#  core/assistant.py – End-to-end Jarvis voice loop
#  Orchestrates wake word → STT → LLM → TTS with follow-up support
#
#  Conversation model:
#    • "Hey Jarvis"  →  fresh conversation (LLM history reset)
#    • After Jarvis speaks  →  keep listening for follow-up (no wake word needed)
#    • User stays silent for FOLLOWUP_SILENCE_TIMEOUT s  →  end conversation,
#      go back to sleep and wait for the next "Hey Jarvis"
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys
import threading
import queue
from enum import Enum

# Ensure the project root is on sys.path so `import config` works when this
# file is run directly (e.g. `python core/assistant.py`).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from loguru import logger

from core.llm import LLMEngine
from core.stt import STTEngine
from core.tts import TTSEngine
from core.wake_word import WakeWordDetector
import config

# ── Tuning constants ──────────────────────────────────────────────────────────

# How long (seconds) to wait for the user to START speaking on a follow-up turn
# before giving up and ending the conversation.  First turn uses STT's default.
FOLLOWUP_SILENCE_TIMEOUT: float = 5.0


class AssistantState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class JarvisAssistant:
    """
    Orchestrates the full Jarvis voice pipeline:

        WakeWordDetector  →  STTEngine  →  LLMEngine  →  TTSEngine
                               ↑                              |
                               └──────── follow-up loop ──────┘

    Call run() to start the assistant.  It blocks until Ctrl-C.
    """

    def __init__(self) -> None:
        # ── Instantiate all engines ───────────────────────────────────────────
        logger.info("Initialising Jarvis engines…")

        self._stt = STTEngine(
            model_size=config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
            language=config.WHISPER_LANGUAGE,
            mic_device_index=config.MIC_DEVICE_INDEX,
        )

        self._llm = LLMEngine()

        self._tts = TTSEngine(
            voice=config.TTS_VOICE,
            speed=config.TTS_SPEED,
        )

        # The wake word detector callback just signals the main thread — it must
        # not do any heavy work itself to avoid blocking the detector's internal
        # thread.
        self._wake_event = threading.Event()
        self._detector = WakeWordDetector(
            callback=self._on_wake_word,
            model_name=config.WAKE_WORD_MODEL,
            threshold=config.WAKE_WORD_THRESHOLD,
            device_index=config.MIC_DEVICE_INDEX,
        )

        self._running = False
        self._state = AssistantState.IDLE
        self._event_queue: queue.Queue[dict] = queue.Queue()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_state(self) -> AssistantState:
        return self._state

    def get_event_queue(self) -> queue.Queue[dict]:
        return self._event_queue

    def _set_state(self, new_state: AssistantState) -> None:
        self._state = new_state

    def _emit_event(self, event_type: str, **kwargs) -> None:
        event = {"event": event_type}
        event.update(kwargs)
        self._event_queue.put(event)

    def run(self) -> None:
        """
        Load all models, start the wake word detector, and block forever.
        Handles Ctrl-C gracefully.
        """
        # Load Whisper and Kokoro models once at startup
        logger.info("Loading Whisper model (first-time may download weights)…")
        self._stt.load_model()
        logger.success("Whisper model ready.")

        logger.info("Loading Kokoro TTS model…")
        self._tts.initialize()
        logger.success("Kokoro TTS ready.")

        self._running = True
        self._wake_event.clear()
        self._detector.start()

        logger.success("─" * 50)
        logger.success("Jarvis is ready. Say 'Hey Jarvis' to begin.")
        logger.success("─" * 50)

        try:
            while self._running:
                # Block until the wake word fires
                self._wake_event.wait()
                self._wake_event.clear()

                if not self._running:
                    break

                # Run the conversation in its own thread, separate from the
                # detector's internal callback thread, to avoid any deadlock.
                conversation_thread = threading.Thread(
                    target=self._run_conversation,
                    name="jarvis-conversation",
                    daemon=True,
                )
                conversation_thread.start()
                conversation_thread.join()  # wait before listening for next wake word

        except KeyboardInterrupt:
            logger.info("\nCtrl-C received — shutting down Jarvis.")
        finally:
            self.stop()

    # ── Wake word callback ────────────────────────────────────────────────────

    def _on_wake_word(self) -> None:
        """
        Called by WakeWordDetector from its internal detector thread.
        Must be non-blocking — just signals the main loop.
        """
        self._emit_event("wake_word_detected")
        self._wake_event.set()

    # ── Conversation loop ─────────────────────────────────────────────────────

    def _run_conversation(self) -> None:
        """
        Full conversation lifecycle: STT → LLM → TTS, with follow-up support.
        Runs in its own thread.  Always restarts the wake detector when done.
        """
        try:
            # ── 1. Stop detector to free the mic ─────────────────────────────
            logger.info("Wake word detected! Stopping wake detector to free mic…")
            self._detector.stop()

            # ── 2. Fresh conversation ─────────────────────────────────────────
            self._llm.reset_history()
            logger.success("=" * 50)
            logger.success("New conversation started.")
            logger.success("=" * 50)

            # Greet the user once per new conversation, before any STT
            logger.info("[Greeting] Speaking opening line…")
            self._tts.speak("Hey, how can I help you today?")

            first_turn = True

            while True:
                # ── 3a. Listen and transcribe ─────────────────────────────────
                self._set_state(AssistantState.LISTENING)
                self._emit_event("listening_started")
                
                if first_turn:
                    logger.info("[Listening] Speak now…")
                    text = self._stt.listen_and_transcribe()
                    first_turn = False
                else:
                    logger.info(
                        f"[Listening for follow-up] "
                        f"Speak within {FOLLOWUP_SILENCE_TIMEOUT:.0f}s or stay silent to end…"
                    )
                    text = self._stt.listen_and_transcribe(
                        initial_silence_timeout=FOLLOWUP_SILENCE_TIMEOUT
                    )

                # ── 3b. Nothing transcribed → end conversation ────────────────
                if not text.strip():
                    logger.info("[Conversation ended] No speech detected.")
                    break

                logger.info(f"[Transcribed] '{text}'")
                self._emit_event("transcription", text=text)

                # ── 3c. Get Claude's response ─────────────────────────────────
                self._set_state(AssistantState.THINKING)
                logger.info("[Thinking] Sending to Claude…")
                reply = self._llm.get_response(text)
                self._emit_event("llm_response", text=reply)

                # ── 3d. Speak the response ────────────────────────────────────
                self._set_state(AssistantState.SPEAKING)
                self._emit_event("speaking_started")
                logger.info(f"[Speaking] '{reply[:80]}{'…' if len(reply) > 80 else ''}'")
                self._tts.speak(reply)
                self._emit_event("speaking_ended")

                # Loop back to listen for follow-up

        except Exception as exc:
            logger.exception(f"Unexpected error in conversation loop: {exc}")

        finally:
            # ── 4. Always restart the wake detector ───────────────────────────
            self._set_state(AssistantState.IDLE)
            self._emit_event("idle")
            self._wake_event.clear()  # discard any stale wake events
            logger.success("─" * 50)
            logger.success("Ready. Say 'Hey Jarvis' to begin a new conversation.")
            logger.success("─" * 50)
            self._detector.start()

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Gracefully stop all components."""
        self._running = False
        self._wake_event.set()  # unblock wait() if sleeping

        logger.info("Stopping wake detector…")
        self._detector.stop()

        logger.info("Shutting down TTS engine…")
        self._tts.shutdown()

        logger.success("Jarvis shut down cleanly. Goodbye.")


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pretty console logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "{message}"
        ),
    )

    assistant = JarvisAssistant()
    assistant.run()
