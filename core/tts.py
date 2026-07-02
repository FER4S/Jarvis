# ─────────────────────────────────────────────────────────────────────────────
#  core/tts.py – Text-to-speech engine for Jarvis
#  Uses Kokoro-82M (local, GPU-accelerated) via the kokoro Python package.
#
#  Pipeline position:  Claude text  →  TTSEngine.speak()  →  Speaker
#
#  Why Kokoro instead of edge-tts:
#    edge-tts has ~1.7s unavoidable network latency (Microsoft server round-trip).
#    Kokoro runs fully locally on the RTX 5060 and generates audio in ~0.3s.
#
#  Flow:
#    1. initialize() — loads the Kokoro-82M model onto GPU (once at startup)
#    2. speak(text)  — runs inference → gets torch.Tensor audio at 24 kHz
#                    → plays directly via sounddevice (no temp file, no disk I/O)
#
#  Voice chosen: am_michael
#    Best-graded American English male voice in Kokoro-82M (B quality target,
#    H hours of training data, C+ overall grade). Neutral, professional tone
#    fitting for an AI assistant. Tied with am_fenrir and am_puck on metrics;
#    am_michael was chosen for its natural assistant-like vocal character.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys
import time
from typing import Optional

# Ensure the project root is on sys.path so `import config` works when this
# file is run directly (e.g. `python core/tts.py`).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import sounddevice as sd
import torch
from kokoro import KPipeline
from loguru import logger

import config

# ── Constants ─────────────────────────────────────────────────────────────────

# Kokoro-82M always outputs at 24 kHz
KOKORO_SAMPLE_RATE: int = 24_000


class TTSEngine:
    """
    Converts text to speech using Kokoro-82M and plays it via sounddevice.

    Call initialize() once at startup to load the model onto GPU.
    Then call speak(text) for each utterance.

    Usage:
        engine = TTSEngine()
        engine.initialize()
        engine.speak("Good morning. How can I help you?")
        engine.shutdown()
    """

    def __init__(
        self,
        voice: str = config.TTS_VOICE,
        speed: float = config.TTS_SPEED,
    ) -> None:
        """
        Args:
            voice: Kokoro voice ID (e.g. 'am_michael').
            speed: Speech rate multiplier (1.0 = normal, 1.1 = slightly faster).
        """
        self._voice = voice
        self._speed = speed
        self._pipeline: Optional[KPipeline] = None
        self._active_device: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Load the Kokoro-82M model onto GPU (or CPU as fallback).
        Call this once at startup — model weights are downloaded automatically
        from Hugging Face on the first run and cached locally after that.
        """
        # Detect device
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            f"Loading Kokoro-82M on {target_device.upper()} "
            f"(voice='{self._voice}', speed={self._speed})…"
        )
        t0 = time.monotonic()

        try:
            # lang_code='a' = American English
            self._pipeline = KPipeline(lang_code="a", device=target_device)
            self._active_device = target_device
            logger.success(
                f"Kokoro-82M ready on {self._active_device.upper()} "
                f"in {time.monotonic() - t0:.1f}s."
            )

        except Exception as exc:
            if target_device != "cpu":
                logger.warning(
                    f"GPU load failed ({exc}). Falling back to CPU…"
                )
                self._pipeline = KPipeline(lang_code="a", device="cpu")
                self._active_device = "cpu"
                logger.success(
                    f"Kokoro-82M ready on CPU (fallback) "
                    f"in {time.monotonic() - t0:.1f}s."
                )
            else:
                logger.error(f"Failed to load Kokoro-82M: {exc}")
                raise

    def speak(self, text: str) -> None:
        """
        Synthesise `text` to speech and play it through the default speaker.
        Blocks until playback is complete.

        Args:
            text: Plain text to speak. Avoid markdown or special symbols.
        """
        if not text or not text.strip():
            logger.warning("speak() called with empty text — skipping.")
            return

        if self._pipeline is None:
            raise RuntimeError(
                "TTSEngine not initialized. Call initialize() first."
            )

        logger.info(f"Speaking: '{text[:80]}{'…' if len(text) > 80 else ''}'")
        t_start = time.monotonic()

        try:
            # ── Inference ─────────────────────────────────────────────────────
            # KPipeline.__call__ returns a generator of Result dataclasses.
            # Each Result.audio is a torch.Tensor (float32, 24 kHz).
            # For most short utterances there is only one segment.
            segments: list[np.ndarray] = []

            for result in self._pipeline(
                text,
                voice=self._voice,
                speed=self._speed,
            ):
                if result.audio is not None:
                    # Move to CPU and convert to float32 numpy for sounddevice
                    segments.append(result.audio.cpu().numpy().astype(np.float32))

            if not segments:
                logger.warning("Kokoro produced no audio for the given text.")
                return

            audio = np.concatenate(segments) if len(segments) > 1 else segments[0]

            t_generated = time.monotonic()
            duration_s = len(audio) / KOKORO_SAMPLE_RATE
            logger.debug(
                f"Inference: {t_generated - t_start:.3f}s → "
                f"{duration_s:.1f}s of audio "
                f"({self._active_device.upper()})"
            )

            # ── Playback ──────────────────────────────────────────────────────
            sd.play(audio, samplerate=KOKORO_SAMPLE_RATE)
            sd.wait()  # block until speaker is done

            logger.debug(f"Total speak() latency: {t_generated - t_start:.3f}s")

        except Exception as exc:
            logger.error(f"TTS error: {exc}")
            sd.stop()  # ensure sounddevice isn't left in a broken state

    def shutdown(self) -> None:
        """Stop any ongoing playback and release resources."""
        sd.stop()
        self._pipeline = None
        logger.debug("TTSEngine shut down.")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    engine = TTSEngine()
    engine.initialize()

    test_lines = [
        "Hey, how can I help you today?",
        "The meeting is scheduled for 3 PM. Shall I send the calendar invite?",
    ]

    for line in test_lines:
        print(f"\nSpeaking: {line}")
        t = time.monotonic()
        engine.speak(line)
        print(f"Done in {time.monotonic() - t:.3f}s total")

    engine.shutdown()
    print("\nDone.")
