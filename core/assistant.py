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

import anthropic
from loguru import logger

from core.llm import LLMEngine
from core.memory import (
    MemoryManager,
    STRUCTURE_KIND_FACT,
    extract_response_text,
    parse_json_object,
)
from core.stt import STTEngine
from core.tts import TTSEngine
from core.wake_word import WakeWordDetector
import config

# ── Tuning constants ──────────────────────────────────────────────────────────

# How long (seconds) to wait for the user to START speaking on a follow-up turn
# before giving up and ending the conversation.  First turn uses STT's default.
FOLLOWUP_SILENCE_TIMEOUT: float = 5.0

# How long (seconds) to wait for the user to answer an onboarding question —
# longer than FOLLOWUP_SILENCE_TIMEOUT since these are substantive, open-ended
# answers rather than quick follow-ups.
ONBOARDING_SILENCE_TIMEOUT: float = 8.0

# One-time first-run onboarding questions, asked in order.
ONBOARDING_QUESTIONS: list[str] = [
    "What's your name?",
    "What's your role at CodeX?",
    "Who are the key people you work with regularly?",
    "What are your main priorities right now?",
    "Is there anything else you'd like me to know about you, your work style, or your preferences?",
]

# Trigger phrases that mark an explicit "remember this" voice command.
MEMORY_TRIGGER_PHRASES: tuple[str, ...] = ("remember that", "don't forget")

# Filler tokens allowed before a trigger phrase at the start of an utterance
# ("Please remember that…", "Jarvis, remember that…"). Any other leading word
# ("Do you remember that…") means the boss is talking ABOUT remembering, not
# issuing the command.
_TRIGGER_PREFIX_TOKENS: frozenset[str] = frozenset({
    "jarvis", "please", "hey", "ok", "okay", "oh", "and", "also", "so", "now", "um", "uh",
})


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

        self._memory = MemoryManager()

        # The wake word detector callback just signals the main thread — it must
        # not do any heavy work itself to avoid blocking the detector's internal
        # thread.
        self._wake_event = threading.Event()
        self._detector = WakeWordDetector(
            callback=self._on_wake_word,
            model_name=config.WAKE_WORD_MODEL,
            threshold=config.WAKE_WORD_THRESHOLD,
            device_index=config.MIC_DEVICE_INDEX,
            on_failure=self._on_detector_failure,
        )

        # Last unrecoverable component failure (e.g. dead mic), surfaced via
        # GET /status and the "error" WebSocket event so the frontend can tell
        # "process up but deaf" apart from healthy. Cleared on each detector
        # (re)start; None = healthy.
        self._last_error: str | None = None

        self._running = False
        self._state = AssistantState.IDLE
        self._event_queue: queue.Queue[dict] = queue.Queue()

        # Lifecycle guard: _active is True from run() entry to exit — covering
        # model loading and onboarding, unlike _running which only covers the
        # live wake loop. run() test-and-sets it under _lifecycle_lock, so two
        # concurrent run() bodies are impossible no matter who spawns the
        # thread (main.py auto-start, POST /start, standalone __main__).
        self._active = False
        self._stop_requested = False
        self._lifecycle_lock = threading.Lock()
        # Serializes stop()'s engine teardown — two threads tearing down
        # PyAudio/sounddevice at once (e.g. POST /stop racing run()'s own
        # shutdown) can crash the process natively.
        self._shutdown_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_state(self) -> AssistantState:
        return self._state

    def is_active(self) -> bool:
        """
        True from run() entry to exit — includes model loading and onboarding,
        not just the live wake loop (which is what _running tracks).
        """
        return self._active

    def get_last_error(self) -> str | None:
        """
        Last unrecoverable component failure (currently: wake word detector /
        mic death), or None when healthy. Exposed for GET /status.
        """
        return self._last_error

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
        Load all models, run first-time onboarding if needed, start the wake
        word detector, and block forever. Handles Ctrl-C gracefully.

        Idempotent across threads: if a run() is already active anywhere —
        loading models, onboarding, or in the wake loop — a second call logs
        a warning and returns immediately instead of starting a duplicate
        pipeline.
        """
        with self._lifecycle_lock:
            if self._active:
                logger.warning("run() called while already active — ignoring duplicate start.")
                return
            self._active = True
            self._stop_requested = False

        try:
            # Load Whisper and Kokoro models once at startup
            logger.info("Loading Whisper model (first-time may download weights)…")
            self._stt.load_model()
            logger.success("Whisper model ready.")

            logger.info("Loading Kokoro TTS model…")
            self._tts.initialize()
            logger.success("Kokoro TTS ready.")

            logger.info("Loading memory store…")
            self._memory.load()
            logger.success("Memory store ready.")

            if not self._memory.is_onboarding_complete():
                self._run_onboarding()

            # A stop() that arrived while models were loading (or during
            # onboarding) must win — don't bring the wake loop up at all.
            if self._stop_requested:
                logger.info("Stop was requested during startup — not starting the wake loop.")
                return

            self._running = True
            self._wake_event.clear()
            self._start_detector()

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
                # If stop() was already requested (e.g. POST /stop), the
                # requesting thread is running the teardown right now — don't
                # race it with a redundant second stop() from this thread.
                if not self._stop_requested:
                    self.stop()
        finally:
            self._active = False

    # ── First-run onboarding ──────────────────────────────────────────────────

    def _run_onboarding(self) -> None:
        """
        One-time onboarding conversation, run before the wake-word loop starts.
        Uses the assistant's already-loaded STT/TTS engines directly — no wake
        word needed to trigger it, and the wake detector hasn't started yet at
        this point (it loads its model / opens the mic in start(), not
        __init__), so there's no mic contention to resolve.

        A silent/empty answer to any question is recorded as blank and
        onboarding simply moves on — it never retries or hangs, so a mic issue
        or an away-from-desk boss can't block first-run startup forever.
        """
        logger.success("=" * 50)
        logger.success("First run detected — starting onboarding.")
        logger.success("=" * 50)

        self._tts.speak(
            "Before we get started, I'd like to get to know you better so I "
            "can serve you well."
        )

        qa_pairs: list[tuple[str, str]] = []
        for question in ONBOARDING_QUESTIONS:
            self._tts.speak(question)
            logger.info(f"[Onboarding] Asked: '{question}'")
            answer = self._stt.listen_and_transcribe(
                initial_silence_timeout=ONBOARDING_SILENCE_TIMEOUT
            )
            logger.info(f"[Onboarding] Answer: '{answer.strip() or '(no answer)'}'")
            qa_pairs.append((question, answer.strip()))

        profile = self._format_onboarding_profile(qa_pairs)
        self._memory.complete_onboarding(profile)

        self._tts.speak("Got it. I'll remember that. Let's get started.")
        logger.success("Onboarding complete.")

    @staticmethod
    def _format_onboarding_profile(qa_pairs: list[tuple[str, str]]) -> dict:
        """
        One-off Claude Haiku call to format raw onboarding Q/A pairs into a
        clean profile dict. Never raises and always returns something safe to
        pass to MemoryManager.complete_onboarding() — falls back to
        {"raw_qa": "<concatenated Q/A text>"} if the model doesn't return
        valid, well-formed JSON for any reason (timeout, auth error, bad JSON,
        wrong shape, etc.). Uses its own one-off anthropic client rather than
        LLMEngine (which manages the stateful voice conversation) or
        MemoryManager (whose public surface doesn't include profile
        formatting).
        """
        raw_qa_text = "\n".join(
            f"Q: {q}\nA: {a if a else '(no answer given)'}" for q, a in qa_pairs
        )
        fallback_profile = {"raw_qa": raw_qa_text}

        try:
            # max_retries=0: the boss is sitting through onboarding waiting on
            # this — fail fast to the raw-Q/A fallback rather than stacking
            # SDK auto-retries.
            client = anthropic.Anthropic(
                api_key=config.ANTHROPIC_API_KEY, timeout=10.0, max_retries=0
            )
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                system=(
                    "Format the following onboarding question-and-answer pairs into "
                    "a clean JSON profile object. Respond with ONLY a JSON object "
                    "(no markdown fences, no commentary), using these keys: "
                    "name, role, key_people, priorities, preferences — "
                    "each a short string summarizing the corresponding answer. If "
                    "an answer was blank/unclear, use an empty string for that key. "
                    "Do not invent information not present in the answers."
                ),
                messages=[{"role": "user", "content": raw_qa_text}],
            )
            parsed = parse_json_object(extract_response_text(response))
            return parsed if parsed is not None else fallback_profile

        except Exception as exc:
            logger.warning(
                f"Onboarding profile formatting failed ({exc}) — "
                f"falling back to raw Q/A text as the profile."
            )
            return fallback_profile

    # ── Explicit memory commands ("remember that...") ────────────────────────

    @staticmethod
    def _extract_explicit_memory_trigger(text: str) -> str | None:
        """
        Case-insensitively check whether text STARTS with an explicit-memory
        trigger phrase ("remember that" / "don't forget"), optionally preceded
        by harmless filler tokens (_TRIGGER_PREFIX_TOKENS — "Jarvis,",
        "please", "okay", …). If so, returns the substring after the trigger
        phrase (the fact content, original casing), stripped of surrounding
        punctuation. Returns "" (not None) if the trigger phrase is present
        but nothing meaningful follows it — still routed to the confirmation
        path, since the boss clearly tried to invoke the command.

        Returns None if no anchored trigger is found, so the caller proceeds
        to the normal get_response() path. That includes utterances that
        merely mention a trigger phrase mid-sentence: questions like "Do you
        remember that meeting?" must reach Claude, not the memory pipeline.
        Deliberate trade-off: a genuinely mid-sentence command ("The launch
        is Friday — don't forget it") no longer triggers either; a false
        negative degrades gracefully (Claude replies, and background
        extraction can still capture it), while a false positive hijacks the
        turn and writes garbage to memory.
        """
        lowered = text.lower()
        for trigger in MEMORY_TRIGGER_PHRASES:
            idx = lowered.find(trigger)
            if idx == -1:
                continue
            if idx > 0:
                # Everything before the trigger must be whitelisted filler —
                # any other word means the trigger is mid-sentence, not a
                # command. (Also rejects partial-word hits: "misremember
                # that…" leaves a "mis" fragment that isn't whitelisted.)
                prefix_tokens = [
                    token.strip(",.;:!?'\"-—…") for token in lowered[:idx].split()
                ]
                if not all(token in _TRIGGER_PREFIX_TOKENS for token in prefix_tokens if token):
                    continue
            return text[idx + len(trigger):].strip(" ,.:;!?").strip()
        return None

    def _speak_confirmation(self, reply: str) -> None:
        """
        Emit + speak a fixed confirmation line, reusing the same state/event
        vocabulary as a normal reply turn (llm_response stands in for Claude's
        reply; SPEAKING + speaking_started/ended around the TTS call).
        """
        self._emit_event("llm_response", text=reply)
        self._set_state(AssistantState.SPEAKING)
        self._emit_event("speaking_started")
        self._tts.speak(reply)
        self._emit_event("speaking_ended")

    def _handle_explicit_memory(self, explicit_fact: str) -> None:
        """
        Handle one explicit "remember that…" command. Facts go to the facts
        list (unchanged). Event-like commands go to the events list with dedup:
        a new event is saved silently; one resembling an existing event triggers
        a live spoken clarifying question ("same, rescheduled — or new?") whose
        answer decides update-vs-add. An unclear/silent answer saves as new.

        Reuses only the existing states (THINKING/LISTENING/SPEAKING) and event
        names; adds no new ones.
        """
        # Empty trigger ("remember that" with nothing after it): preserve the
        # original behavior — a plain confirmation, no structuring call.
        if not explicit_fact.strip():
            self._set_state(AssistantState.THINKING)
            self._speak_confirmation("Got it, I'll remember that.")
            return

        self._set_state(AssistantState.THINKING)
        structured = self._memory.structure_explicit_memory(explicit_fact)

        # ── Fact: keep the existing fact-cleanup + facts-list path unchanged ──
        if structured["kind"] == STRUCTURE_KIND_FACT:
            cleaned = self._memory.clean_fact_text(explicit_fact)
            self._memory.add_explicit_memory(cleaned)
            self._speak_confirmation("Got it, I'll remember that.")
            return

        description = structured["description"]
        date = structured["date"]
        match = structured["match_description"]

        # ── Event with no similar existing one: save as new, no question ──────
        if not match:
            self._memory.add_event(description, date)
            self._speak_confirmation("Got it — I've noted that event.")
            return

        # ── Resembles an existing event: ask a live clarifying question ───────
        self._speak_confirmation(
            "I've already got a similar event noted. Is this the same one, "
            "just rescheduled — or a new event?"
        )

        self._set_state(AssistantState.LISTENING)
        self._emit_event("listening_started")
        answer = self._stt.listen_and_transcribe(
            initial_silence_timeout=FOLLOWUP_SILENCE_TIMEOUT
        )
        if answer.strip():
            logger.info(f"[Explicit memory] Clarifying answer: '{answer}'")
            self._emit_event("transcription", text=answer)

        self._set_state(AssistantState.THINKING)
        verdict = self._memory.classify_same_or_new(answer)

        if verdict == "same" and self._memory.update_event_date(match, date, description):
            self._speak_confirmation("Got it — I've updated that event.")
        else:
            # An explicit "new" force-appends a separate entry even if the
            # description collides with the existing event — the boss said it's
            # different, so the dedup net must not silently merge it. "unclear"/
            # silent (and a stale "same" match key) add without forcing, letting
            # dedup guard against a stray duplicate on a non-answer. Either way
            # the information is recorded.
            self._memory.add_event(description, date, force_new=(verdict == "new"))
            self._speak_confirmation("Got it — I've noted that event.")

    # ── Wake word callback ────────────────────────────────────────────────────

    def _on_wake_word(self) -> None:
        """
        Called by WakeWordDetector from its internal detector thread.
        Must be non-blocking — just signals the main loop.
        """
        self._emit_event("wake_word_detected")
        self._wake_event.set()

    def _on_detector_failure(self, reason: str) -> None:
        """
        Called by WakeWordDetector (from its listener thread) when it dies
        unrecoverably — mic failed to open, or too many consecutive read
        failures. Records the failure for GET /status and broadcasts an
        "error" event so the frontend can show it instead of a healthy idle.
        """
        self._last_error = f"wake word detector: {reason}"
        self._emit_event("error", message=self._last_error)

    def _start_detector(self) -> None:
        """
        Start the wake detector with a clean error slate. If the mic is still
        dead, _on_detector_failure re-sets _last_error within milliseconds —
        so a successful restart clears the error and a failed one keeps it
        honest.
        """
        self._last_error = None
        self._detector.start()

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

            # Compute once per conversation — memory doesn't change mid-conversation.
            memory_context = self._memory.get_context_summary()

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

                # ── 3c. Explicit memory command? ("remember that...") ─────────
                # Replaces the normal get_response() step entirely for this turn.
                # Never sent to self._llm.get_response(), so it never enters
                # self._llm.history — captured directly in memory instead.
                explicit_fact = self._extract_explicit_memory_trigger(text)
                if explicit_fact is not None:
                    self._handle_explicit_memory(explicit_fact)
                    continue

                # ── 3d. Get Claude's response ─────────────────────────────────
                self._set_state(AssistantState.THINKING)
                logger.info("[Thinking] Sending to Claude…")
                reply = self._llm.get_response(text, memory_context=memory_context)
                self._emit_event("llm_response", text=reply)

                # ── 3e. Speak the response ────────────────────────────────────
                self._set_state(AssistantState.SPEAKING)
                self._emit_event("speaking_started")
                logger.info(f"[Speaking] '{reply[:80]}{'…' if len(reply) > 80 else ''}'")
                self._tts.speak(reply)
                self._emit_event("speaking_ended")

                # Loop back to listen for follow-up

        except Exception as exc:
            logger.exception(f"Unexpected error in conversation loop: {exc}")

        finally:
            # ── 4. Extract memory in the background, then always restart the
            #      wake detector. Spawned first, before anything else in this
            #      block: extract_and_save() makes a Sonnet-class API call —
            #      too slow to block "Hey Jarvis" being re-triggerable on.
            #      self._llm.history is a defensive copy per call, and capturing
            #      it here (before self._detector.start()) means no new
            #      conversation's reset_history() can possibly race with it.
            threading.Thread(
                target=self._memory.extract_and_save,
                args=(self._llm.history,),
                name="jarvis-memory-extraction",
                daemon=True,
            ).start()

            # ── 5. Always restart the wake detector ───────────────────────────
            self._set_state(AssistantState.IDLE)
            self._emit_event("idle")
            self._wake_event.clear()  # discard any stale wake events
            logger.success("─" * 50)
            logger.success("Ready. Say 'Hey Jarvis' to begin a new conversation.")
            logger.success("─" * 50)
            self._start_detector()

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Gracefully stop all components. Safe to call from multiple threads."""
        self._stop_requested = True  # honored by run() if it's still starting up
        self._running = False
        self._wake_event.set()  # unblock wait() if sleeping

        with self._shutdown_lock:
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
