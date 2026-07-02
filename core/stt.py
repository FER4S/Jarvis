# ─────────────────────────────────────────────────────────────────────────────
#  core/stt.py – Speech-to-text engine for Jarvis
# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline:
#    1. listen_and_transcribe() opens the mic
#    2. Calibrates ambient noise for ~0.3 s
#    3. Records in chunks, tracking RMS energy per chunk
#    4. Stops when silence lasts >= SILENCE_DURATION or MAX_DURATION is hit
#    5. Converts the PCM buffer to float32 and passes it to Whisper
#    6. Returns the transcribed text as a plain string
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys

# Ensure the project root is on sys.path so `import config` works when this
# file is run directly (e.g. `python core/stt.py`).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import math
import time
from typing import Optional

import numpy as np
import pyaudio
from loguru import logger

# ── Fix for Windows CUDA DLLs ─────────────────────────────────────────────────
# Windows Python 3.8+ no longer uses PATH for DLL resolution. The nvidia-* pip
# packages place their DLLs in site-packages/nvidia/*/bin. We must add them manually.
if os.name == "nt":
    from pathlib import Path
    _site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    for _dll_path in _site_packages.glob("nvidia/*/bin"):
        if _dll_path.is_dir():
            try:
                os.add_dll_directory(str(_dll_path))
                logger.info(f"Added CUDA DLL path: {_dll_path}")
            except Exception as e:
                logger.error(f"Failed to add DLL path {_dll_path}: {e}")
# ──────────────────────────────────────────────────────────────────────────────

from faster_whisper import WhisperModel

import config

# ── Audio constants ───────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000       # Hz — Whisper requires 16 kHz
CHANNELS: int = 1               # mono
FORMAT: int = pyaudio.paInt16   # 16-bit PCM
CHUNK_SAMPLES: int = 512        # ~32 ms per chunk at 16 kHz (fine-grained for silence detection)

# ── Recording limits ──────────────────────────────────────────────────────────
MAX_DURATION_S: float = 30.0    # hard cap — stop recording after this many seconds
SILENCE_DURATION_S: float = 2 # stop after this many consecutive seconds of silence

# ── Silence detection ─────────────────────────────────────────────────────────
CALIBRATION_S: float = 0.3      # seconds of audio used to measure ambient noise floor
SILENCE_MULTIPLIER: float = 1.5 # threshold = ambient_rms * this value


class STTEngine:
    """
    Wraps faster-whisper for on-demand speech-to-text.

    Typical usage (called by the pipeline after wake word fires):

        engine = STTEngine()
        engine.load_model()
        text = engine.listen_and_transcribe()
        print(text)
    """

    def __init__(
        self,
        model_size: str = config.WHISPER_MODEL_SIZE,
        device: str = config.WHISPER_DEVICE,
        compute_type: str = config.WHISPER_COMPUTE_TYPE,
        language: str = config.WHISPER_LANGUAGE,
        mic_device_index: Optional[int] = config.MIC_DEVICE_INDEX,
    ) -> None:
        """
        Args:
            model_size:       Whisper model size (tiny/base/small/medium/large-v3).
            device:           "cuda" for GPU or "cpu". "auto" lets faster-whisper decide.
            compute_type:     "float16" (GPU), "int8_float16" (GPU, less VRAM),
                              "int8" (CPU). Defaults to "default" which lets the
                              library pick based on device.
            language:         BCP-47 language code, e.g. "en". None = auto-detect.
            mic_device_index: PyAudio device index. None = OS default mic.
        """
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._mic_device_index = mic_device_index
        self._model: Optional[WhisperModel] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """
        Load the Whisper model onto the requested device.
        Falls back to CPU automatically if CUDA is unavailable.
        """
        logger.info(
            f"Loading Whisper model '{self._model_size}' "
            f"on {self._device} ({self._compute_type})…"
        )
        try:
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            self._active_device = self._device
            logger.success(
                f"Whisper '{self._model_size}' loaded. "
                f"Active device: {self._active_device.upper()} | "
                f"compute_type: {self._compute_type}"
            )
        except Exception as exc:
            # If CUDA is unavailable at load time, fall back to CPU transparently
            if self._device != "cpu":
                logger.warning(
                    f"Failed to load on {self._device} ({exc}). "
                    f"Falling back to CPU with int8 compute."
                )
                self._model = WhisperModel(
                    self._model_size,
                    device="cpu",
                    compute_type="int8",
                )
                self._active_device = "cpu"
                logger.success(f"Whisper '{self._model_size}' loaded on CPU (fallback).")
            else:
                raise

    def listen_and_transcribe(
        self,
        initial_silence_timeout: float = SILENCE_DURATION_S,
    ) -> str:
        """
        Open the microphone, record until the user stops speaking, then
        transcribe the audio with Whisper.

        Args:
            initial_silence_timeout: How long (seconds) to wait for the user to
                START speaking before giving up entirely.  Once the first speech
                is detected, the existing trailing-silence cutoff takes over.
                Default = SILENCE_DURATION_S so normal first-turn behaviour is
                unchanged.  Pass a larger value (e.g. 4-5 s) for follow-up turns
                so the user has a natural pause to decide whether to keep talking.

        Returns:
            Transcribed text as a plain stripped string.
            Returns an empty string if nothing was captured or understood.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info("Listening…")
        audio_data = self._record_until_silence(
            initial_silence_timeout=initial_silence_timeout
        )

        if audio_data is None or len(audio_data) == 0:
            logger.warning("No audio captured.")
            return ""

        duration_s = len(audio_data) / SAMPLE_RATE
        logger.info(f"Recorded {duration_s:.1f}s of audio. Transcribing…")

        return self._transcribe(audio_data)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _record_until_silence(
        self,
        initial_silence_timeout: float = SILENCE_DURATION_S,
    ) -> Optional[np.ndarray]:
        """
        Open the mic and record until silence is detected or the max duration
        is reached.

        Args:
            initial_silence_timeout: Seconds to wait for the user to START
                speaking.  If no speech is detected within this window the
                recording is aborted and an empty result is returned.  Once
                the first speech chunk is seen, trailing-silence detection
                takes over using the global SILENCE_DURATION_S constant.

        Returns a float32 numpy array (values in [-1.0, 1.0]) at SAMPLE_RATE,
        or None on error.
        """
        pa = pyaudio.PyAudio()
        stream = None

        try:
            stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=self._mic_device_index,
                frames_per_buffer=CHUNK_SAMPLES,
            )

            # ── Step 1: calibrate ambient noise ──────────────────────────────
            silence_threshold = self._calibrate_noise(stream)
            logger.debug(f"Silence threshold set to RMS={silence_threshold:.1f}")

            # ── Step 2: record chunks until silence or max duration ───────────
            frames: list[bytes] = []
            silent_chunks = 0
            speech_started = False                          # has the user begun talking?
            pre_speech_start = time.monotonic()            # clock for initial timeout
            max_chunks = int(MAX_DURATION_S * SAMPLE_RATE / CHUNK_SAMPLES)
            silence_chunks_needed = int(SILENCE_DURATION_S * SAMPLE_RATE / CHUNK_SAMPLES)

            logger.debug(
                f"Recording (max={MAX_DURATION_S}s, "
                f"initial_timeout={initial_silence_timeout}s, "
                f"trailing_silence={SILENCE_DURATION_S}s)…"
            )

            for _ in range(max_chunks):
                chunk = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
                frames.append(chunk)

                rms = self._compute_rms(chunk)

                if not speech_started:
                    # ── Waiting for user to start speaking ────────────────
                    if rms >= silence_threshold:
                        # First speech chunk detected — switch to trailing-silence mode
                        speech_started = True
                        silent_chunks = 0
                    elif time.monotonic() - pre_speech_start >= initial_silence_timeout:
                        # Timeout: user never started speaking
                        logger.debug(
                            f"No speech within {initial_silence_timeout:.1f}s — aborting."
                        )
                        frames.clear()   # discard the captured silence
                        break
                else:
                    # ── Trailing-silence detection (original logic) ────────
                    if rms < silence_threshold:
                        silent_chunks += 1
                    else:
                        silent_chunks = 0  # reset on any speech

                    if silent_chunks >= silence_chunks_needed:
                        logger.debug("Trailing silence detected — stopping recording.")
                        break

        except OSError as exc:
            logger.error(f"Microphone error: {exc}")
            return None
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            pa.terminate()

        if not frames:
            return None

        # ── Step 3: convert raw PCM bytes → float32 numpy array ──────────────
        raw_bytes = b"".join(frames)
        int16_audio = np.frombuffer(raw_bytes, dtype=np.int16)
        float32_audio = int16_audio.astype(np.float32) / 32768.0  # normalise to [-1, 1]
        return float32_audio

    def _calibrate_noise(self, stream: pyaudio.Stream) -> float:
        """
        Read a short burst of audio to measure the ambient RMS noise floor,
        then return a threshold slightly above it.
        """
        calibration_chunks = int(CALIBRATION_S * SAMPLE_RATE / CHUNK_SAMPLES)
        rms_values: list[float] = []

        for _ in range(max(calibration_chunks, 5)):
            chunk = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
            rms_values.append(self._compute_rms(chunk))

        ambient_rms = sum(rms_values) / len(rms_values)
        threshold = max(ambient_rms * SILENCE_MULTIPLIER, 50.0)  # 50 = sane minimum
        return threshold

    @staticmethod
    def _compute_rms(chunk: bytes) -> float:
        """
        Compute the Root Mean Square energy of a raw int16 PCM chunk.
        Higher values = louder audio.
        """
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        if len(samples) == 0:
            return 0.0
        return math.sqrt(np.mean(samples ** 2))

    def _transcribe(self, audio: np.ndarray) -> str:
        """
        Run Whisper inference on a float32 audio array.

        Returns the full transcription as a single stripped string.
        If a CUDA runtime library is missing (e.g. cublas64_12.dll), the model
        is transparently reloaded on CPU and inference is retried once.
        """
        def _run(model: WhisperModel) -> tuple:
            return model.transcribe(
                audio,
                language=self._language or None,  # None = auto-detect
                beam_size=5,
                vad_filter=True,           # skip non-speech segments automatically
                vad_parameters={"min_silence_duration_ms": 500},
            )

        try:
            segments, info = _run(self._model)
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except RuntimeError as exc:
            # cublas64_12.dll / CUDA runtime not found — fall back to CPU
            err = str(exc).lower()
            if any(kw in err for kw in ("cublas", "cuda", "cudnn", "dll")):
                logger.warning(
                    f"CUDA inference error: {exc}\n"
                    f"Reloading model on CPU and retrying…"
                )
                self._model = WhisperModel(
                    self._model_size,
                    device="cpu",
                    compute_type="int8",
                )
                self._active_device = "cpu"
                logger.success("Model reloaded on CPU.")
                segments, info = _run(self._model)
                text = " ".join(seg.text.strip() for seg in segments).strip()
            else:
                raise

        logger.debug(
            f"Detected language: '{info.language}' "
            f"(probability={info.language_probability:.2f})"
        )
        logger.success(f"Transcribed: '{text}'")
        return text


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import signal

    # Pretty console logging for standalone testing
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    engine = STTEngine(
        model_size="base",   # fast & accurate enough for testing
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )

    engine.load_model()

    print("\n--- Speak after the prompt, then stay silent to stop ---\n")

    try:
        result = engine.listen_and_transcribe()
        if result:
            print(f"\nYou said: {result}\n")
        else:
            print("\n(Nothing transcribed)\n")
    except KeyboardInterrupt:
        print("\nCancelled.")
