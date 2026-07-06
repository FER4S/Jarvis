# ─────────────────────────────────────────────────────────────────────────────
#  core/wake_word.py – Wake word listener
#  ─────────────────────────────────────────────────────────────────────────────
#  openwakeword ships 6 built-in models:
#    alexa, hey_mycroft, hey_jarvis, hey_rhasspy, timer, weather
#
#  "hey_jarvis" is the only model that contains the word "Jarvis", making it
#  the closest phonetic match to any "Jarvis"-style wake phrase. The phrase
#  "Hey Jarvis" is two syllables shorter than "Wake up Jarvis" but in
#  practice both trigger the same model because the neural net learned the
#  acoustic pattern around the word *Jarvis*. A custom openwakeword model
#  trained on "Wake up Jarvis" audio can be swapped in later via config.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys

# Ensure the project root is on sys.path so `import config` works when this
# file is run directly (e.g. `python core/wake_word.py`) instead of as a module.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import queue
import threading
from typing import Callable, Optional

import numpy as np
import pyaudio
from loguru import logger
from openwakeword.model import Model

import config

# ── Audio constants required by openwakeword ──────────────────────────────────
SAMPLE_RATE: int = 16_000          # Hz — openwakeword requires 16 kHz audio
CHANNELS: int = 1                  # mono
SAMPLE_WIDTH: int = 2              # bytes (16-bit PCM = paInt16)
CHUNK_SAMPLES: int = 1_280         # ~80 ms per chunk at 16 kHz (oww recommendation)
FORMAT: int = pyaudio.paInt16


class WakeWordDetector:
    """
    Continuously listens to the microphone and fires a callback whenever
    the wake word is detected.

    Usage:
        def on_wake():
            print("Jarvis woke up!")

        detector = WakeWordDetector(callback=on_wake)
        detector.start()          # non-blocking — runs listener in background thread
        ...
        detector.stop()           # graceful shutdown
    """

    def __init__(
        self,
        callback: Optional[Callable[[], None]] = None,
        model_name: str = "hey_jarvis",
        threshold: float = 0.5,
        device_index: Optional[int] = None,
        inference_framework: str = "onnx",
    ) -> None:
        """
        Args:
            callback:           Function called (no arguments) on each detection.
            model_name:         openwakeword built-in model name.  Defaults to the
                                value in config, falling back to "hey_jarvis".
            threshold:          Confidence score [0.0 – 1.0] above which a detection
                                is reported.  0.5 is a good starting point; lower
                                values increase sensitivity but also false positives.
            device_index:       PyAudio device index for the microphone.  None = OS
                                default.
            inference_framework: "onnx" (default, uses onnxruntime already installed)
                                 or "tflite" (requires tflite-runtime).
        """
        self._callback: Callable[[], None] = callback or self._default_callback
        self._model_name: str = getattr(config, "WAKE_WORD_MODEL", model_name)
        self._threshold: float = getattr(config, "WAKE_WORD_THRESHOLD", threshold)
        self._device_index: Optional[int] = (
            device_index if device_index is not None
            else getattr(config, "MIC_DEVICE_INDEX", None)
        )
        self._inference_framework: str = inference_framework

        self._audio_queue: queue.Queue[bytes] = queue.Queue()
        self._running: threading.Event = threading.Event()
        # Serializes stop(): two threads tearing down the same PyAudio
        # stream/instance concurrently crashes natively (access violation in
        # PortAudio), so the running-check + teardown must be atomic.
        self._stop_lock: threading.Lock = threading.Lock()
        self._listener_thread: Optional[threading.Thread] = None
        self._detector_thread: Optional[threading.Thread] = None

        # PyAudio handle — opened on start()
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None

        # openwakeword model — loaded on start()
        self._oww: Optional[Model] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Load the model and begin listening in background threads."""
        if self._running.is_set():
            logger.warning("WakeWordDetector is already running.")
            return

        logger.info(f"Loading openwakeword model: '{self._model_name}' (framework={self._inference_framework})")
        self._oww = Model(
            wakeword_models=[self._model_name],
            inference_framework=self._inference_framework,
        )
        logger.success(f"Model loaded. Threshold = {self._threshold}")

        self._pa = pyaudio.PyAudio()
        self._running.set()

        # Thread 1 – reads raw PCM from the mic into the queue
        self._listener_thread = threading.Thread(
            target=self._mic_listener,
            name="jarvis-mic-listener",
            daemon=True,
        )
        # Thread 2 – pops chunks off the queue and runs inference
        self._detector_thread = threading.Thread(
            target=self._detector_loop,
            name="jarvis-wake-detector",
            daemon=True,
        )

        self._listener_thread.start()
        self._detector_thread.start()
        logger.info("Wake word detector is listening… (say 'Hey Jarvis')")

    def stop(self) -> None:
        """Signal both threads to stop and clean up all resources.

        Safe to call even if the detector is already stopped, and safe to call
        from multiple threads at once (the whole teardown holds _stop_lock, so
        a concurrent second caller waits, then sees the detector already
        stopped and returns).  Fully releases the microphone and drains the
        audio queue so the next start() begins with a clean slate.
        """
        with self._stop_lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        """Actual stop logic. Caller must already hold self._stop_lock."""
        if not self._running.is_set():
            # Still drain the queue in case a previous stop left a sentinel
            self._drain_queue()
            return

        logger.info("Stopping wake word detector…")
        self._running.clear()

        # Unblock the detector loop if it is waiting on an empty queue
        self._audio_queue.put(b"")

        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=3)
        if self._detector_thread and self._detector_thread.is_alive():
            self._detector_thread.join(timeout=3)

        # ── Release the audio stream ──────────────────────────────────────────
        # Do NOT gate this on is_active() — on Windows, an idle-but-open
        # blocking stream can report is_active()==False yet still hold the
        # device, preventing the next pa.open() from succeeding.
        if self._stream is not None:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

        # ── Drain the queue ───────────────────────────────────────────────────
        # The listener may have put one more audio chunk after the sentinel was
        # inserted (a benign race).  If that chunk ends up AFTER the sentinel,
        # the next detector loop would read the sentinel first and exit
        # immediately, leaving the detector silently dead.  Draining here
        # guarantees the next start() always gets a clean queue.
        self._drain_queue()

        logger.info("Wake word detector stopped.")

    def _drain_queue(self) -> None:
        """Remove all items from the audio queue (including any stale sentinel)."""
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break


    # ── Internal threads ──────────────────────────────────────────────────────

    def _mic_listener(self) -> None:
        """
        Opens the PyAudio input stream and continuously reads 80 ms chunks,
        pushing raw bytes onto the audio queue.
        """
        try:
            self._stream = self._pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=self._device_index,
                frames_per_buffer=CHUNK_SAMPLES,
            )
        except OSError as exc:
            logger.error(f"Failed to open microphone: {exc}")
            self._running.clear()
            return

        logger.debug(f"Microphone stream opened (device_index={self._device_index}, "
                     f"rate={SAMPLE_RATE}, chunk={CHUNK_SAMPLES})")

        while self._running.is_set():
            try:
                raw = self._stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
                self._audio_queue.put(raw)
            except OSError as exc:
                logger.warning(f"Mic read error: {exc}")

    def _detector_loop(self) -> None:
        """
        Consumes audio chunks from the queue, runs openwakeword inference on
        each chunk, and fires the callback when the confidence exceeds the
        threshold.
        """
        while self._running.is_set():
            try:
                raw = self._audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Sentinel value from stop() — exit loop
            if raw == b"":
                break

            # Convert raw bytes → int16 numpy array expected by openwakeword
            audio_chunk = np.frombuffer(raw, dtype=np.int16)

            # Run inference; returns {model_name: score, ...}
            predictions: dict[str, float] = self._oww.predict(audio_chunk)

            for model_name, score in predictions.items():
                if score >= self._threshold:
                    logger.success(
                        f"🎙️  Wake word detected! "
                        f"[model={model_name}, score={score:.3f}, threshold={self._threshold}]"
                    )
                    print("Wake word detected!")
                    try:
                        self._callback()
                    except Exception as exc:
                        logger.error(f"Callback raised an exception: {exc}")

                    # Reset the model's internal context to avoid repeated
                    # triggers from the same utterance
                    self._oww.reset()

    # ── Default callback (placeholder) ───────────────────────────────────────

    @staticmethod
    def _default_callback() -> None:
        """Default callback for standalone testing."""
        logger.info("(Placeholder) STT pipeline would start here.")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import signal
    import time
    from loguru import logger

    # Pretty console logging for standalone testing
    logger.remove()
    logger.add(sys.stderr, level="DEBUG", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

    def on_wake_word():
        """Test callback — just prints a confirmation."""
        logger.info("Callback fired. In production, STT would start here.")

    # ── Verification: repeated stop/start cycles (no mic needed) ─────────────
    # Confirms Thread objects are recreated, queue is drained, and no
    # exceptions are raised across multiple cycles.
    logger.info("Running stop/start cycle verification (no mic required)…")
    detector = WakeWordDetector(callback=on_wake_word, threshold=0.5)

    CYCLES = 5
    for i in range(1, CYCLES + 1):
        detector.start()

        # Give the threads a moment to actually launch
        time.sleep(0.3)

        assert detector._listener_thread is not None and detector._listener_thread.is_alive(), \
            f"Cycle {i}: listener thread is not alive after start()"
        assert detector._detector_thread is not None and detector._detector_thread.is_alive(), \
            f"Cycle {i}: detector thread is not alive after start()"

        detector.stop()

        assert not detector._listener_thread.is_alive(), \
            f"Cycle {i}: listener thread still alive after stop()"
        assert not detector._detector_thread.is_alive(), \
            f"Cycle {i}: detector thread still alive after stop()"
        assert detector._audio_queue.empty(), \
            f"Cycle {i}: audio queue is not empty after stop()"

        logger.success(f"Cycle {i}/{CYCLES} passed.")

    logger.success("All stop/start cycles passed. Proceeding to live test.")

    # ── Live detection loop ───────────────────────────────────────────────────
    detector2 = WakeWordDetector(callback=on_wake_word, threshold=0.5)

    def handle_sigint(sig, frame):
        print()
        detector2.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    detector2.start()

    # Block the main thread so the daemon threads can keep running
    try:
        signal.pause()   # Unix-only; falls through on Windows
    except AttributeError:
        # Windows fallback — just loop
        while True:
            time.sleep(0.5)

