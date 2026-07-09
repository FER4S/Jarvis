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

from core.email_fetch import sanitize_body_text
from core.email_manager import EmailManager, VAGUE_TIMEFRAME_DAYS_BACK_DEFAULT
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

# Keywords that gate a normal (non-announcement) turn into email-intent
# classification, so an unrelated "read me a poem" doesn't cost a Haiku call.
EMAIL_KEYWORDS: frozenset[str] = frozenset({
    "email", "emails", "e-mail", "mail", "inbox", "unread",
    "gmail", "hostinger", "message", "messages", "read",
})

# How many unresolved clarify-and-re-search rounds a "find this email"
# request gets before Jarvis offers to just show everything matching what's
# known so far, instead of asking indefinitely.
MAX_SEARCH_REFINEMENT_ROUNDS: int = 3


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

        self._email = EmailManager(on_new_mail=self._on_new_mail)

        # The wake word detector callback just signals the main thread — it must
        # not do any heavy work itself to avoid blocking the detector's internal
        # thread.
        self._wake_event = threading.Event()
        # Set (before _wake_event) by _on_wake_word, cleared by the dispatch
        # loop when consumed - lets the loop tell "woken by a wake word" apart
        # from "woken by a new-mail poke" without racing _wake_event itself.
        self._wake_word_pending = False
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

    def get_email_manager(self) -> EmailManager:
        """
        Exposes the single shared EmailManager instance. Exactly one must
        exist per process - api/server.py binds its endpoints to this same
        object rather than constructing its own (see core/email_manager.py's
        module docstring for why).
        """
        return self._email

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

            logger.info("Loading email account store…")
            self._email.initialize()
            logger.success("Email manager ready.")

            if not self._memory.is_onboarding_complete():
                self._run_onboarding()

            # A stop() that arrived while models were loading (or during
            # onboarding) must win — don't bring the wake loop up at all.
            if self._stop_requested:
                logger.info("Stop was requested during startup — not starting the wake loop.")
                return

            self._running = True
            self._wake_word_pending = False
            self._wake_event.clear()
            self._start_detector()
            self._email.start_polling()

            logger.success("─" * 50)
            logger.success("Jarvis is ready. Say 'Hey Jarvis' to begin.")
            logger.success("─" * 50)

            try:
                while self._running:
                    # Wake word takes priority over a pending email
                    # announcement. If both are pending, "Hey Jarvis" is
                    # served first; a still-relevant announcement survives
                    # (claim_announcement re-applies its staleness filter)
                    # to be claimed on a later pass of this loop.
                    if self._wake_word_pending:
                        self._wake_word_pending = False
                        if not self._running:
                            break
                        self._spawn_conversation()
                        continue

                    announcement = self._email.claim_announcement()
                    if announcement is not None:
                        if not self._running:
                            break
                        self._spawn_conversation(announcement=announcement)
                        continue

                    # Nothing pending - block until the wake word fires or a
                    # new-mail poke wakes us to re-check for an announcement.
                    self._wake_event.wait()
                    self._wake_event.clear()
                    if not self._running:
                        break

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

    def _spawn_conversation(self, announcement: str | None = None) -> None:
        """
        Run one conversation in its own thread - separate from the
        detector's internal callback thread, to avoid any deadlock - and
        block the dispatch loop until it finishes (conversations are
        strictly serial; see run()'s loop above).
        """
        conversation_thread = threading.Thread(
            target=self._run_conversation,
            kwargs={"announcement": announcement},
            name="jarvis-conversation",
            daemon=True,
        )
        conversation_thread.start()
        conversation_thread.join()

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

    def _ask_and_listen(self, question: str) -> str:
        """
        Speak a clarifying question, then listen for the answer using the
        follow-up timeout. Returns the transcribed answer ("" if silent).
        Shared by the explicit-memory clarify dialogue and the email-query
        clarify dialogues - same speak/listen/emit pattern either way.
        """
        self._speak_confirmation(question)
        self._set_state(AssistantState.LISTENING)
        self._emit_event("listening_started")
        answer = self._stt.listen_and_transcribe(initial_silence_timeout=FOLLOWUP_SILENCE_TIMEOUT)
        if answer.strip():
            logger.info(f"[Clarifying answer] '{answer}'")
            self._emit_event("transcription", text=answer)
        return answer

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
        answer = self._ask_and_listen(
            "I've already got a similar event noted. Is this the same one, "
            "just rescheduled — or a new event?"
        )

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

    # ── Email queries ─────────────────────────────────────────────────────────

    def _maybe_handle_email_query(self, text: str, email_ctx: dict) -> bool:
        """
        Intercepts one turn if it's an email-related request. Returns True if
        handled (caller should `continue` the turn loop without touching
        self._llm) - mirrors _handle_explicit_memory's role for "remember
        that…" commands.

        Gate: an EMAIL_KEYWORDS hit, OR live conversation email context (a
        just-announced/just-listed batch, or an in-flight criteria search).
        Natural follow-ups carry no email keyword ("yes, please", "the last
        one about Jarvis", "maybe three weeks ago"), so once emails have been
        surfaced OR a "find this email" search is awaiting more detail, every
        remaining turn is classified — unrelated turns fall through via
        not_email at the cost of one fast Haiku call. Without the
        pending_search check, an answer to Jarvis's own re-ask ("roughly two
        weeks ago") would carry no keyword and no last_listed/announced batch
        either, and would wrongly fall through to plain Claude instead of
        continuing the search. Classification runs before the "no accounts
        connected" check, so a keyword false-positive ("read me a poem") with
        zero accounts still correctly falls through to Claude. A classifier
        failure/timeout always resolves to "not_email" (see EmailManager.
        classify_intent), so this never hijacks a turn on error.
        """
        context_emails = email_ctx.get("last_listed") or email_ctx.get("announced") or []
        has_pending_search = email_ctx.get("pending_search") is not None
        # An email that was just read is "in focus": a keyword-less follow-up
        # question about it ("who sent it?", "what did they ask for?") must
        # open the gate too, even when nothing is in last_listed (a direct
        # single-match read sets only last_read).
        has_last_read = email_ctx.get("last_read") is not None
        lowered = text.lower()
        keyword_hit = any(keyword in lowered for keyword in EMAIL_KEYWORDS)
        if not (keyword_hit or context_emails or has_pending_search or has_last_read):
            return False

        self._set_state(AssistantState.THINKING)
        parsed = self._email.classify_intent(
            text, context_emails=context_emails, active_search=email_ctx.get("pending_search")
        )
        intent = parsed["intent"]

        if intent == "not_email":
            # pending_search is deliberately NOT cleared here - a stray
            # unrelated aside mid-search goes to Claude but the search stays
            # alive and resumes on the next email turn (resumable, no-trap).
            return False

        # dismiss and send_email bypass the accounts check - their replies
        # are true whether or not any accounts are connected.
        if intent == "dismiss":
            self._handle_email_dismiss(email_ctx)
            return True

        # A topic-filtered "find the email about X (from Y)" is a search, not
        # a browse, regardless of whether the classifier labeled it
        # read_email, list_emails, or from_person - route all of them through
        # the accumulator so criteria narrow across turns. A pure browse
        # (list/from_person with no topic) and any genuinely-different intent
        # clears an in-flight search (the boss moved on).
        is_search = intent == "read_email" or (
            intent in ("list_emails", "from_person") and parsed["subject_hint"]
        )
        if not is_search:
            email_ctx["pending_search"] = None

        if intent != "send_email" and not self._email.has_accounts():
            self._speak_confirmation(
                "You don't have any email accounts connected yet — you can "
                "add one from the settings dashboard."
            )
            return True

        if intent in ("check_new", "check_now"):
            self._handle_email_counts(parsed, email_ctx, live=(intent == "check_now"))
        elif intent == "email_question":
            self._handle_email_question(parsed, email_ctx, text)
        elif intent == "read_email":
            self._handle_read_email(parsed, email_ctx)
        elif is_search:  # list_emails/from_person WITH a topic → find-a-specific-email
            self._handle_email_search(parsed, email_ctx, auto_read=False)
        elif intent in ("list_emails", "from_person"):  # pure browse, no topic
            self._handle_email_list(parsed, email_ctx)
        elif intent in ("summarize", "action_needed", "meetings_mentioned"):
            self._handle_email_reasoning(parsed, intent)
        elif intent == "send_email":
            self._handle_send_decline()
        else:
            return False  # unreachable given classify_intent's whitelist, but never hijack

        return True

    def _handle_email_dismiss(self, email_ctx: dict) -> None:
        """The boss declined a just-offered email batch ("no thanks") — brief
        acknowledgment instead of a contextless Claude reply. The announced/
        listed context is deliberately kept, so "actually, read it" still
        resolves on a later turn. Any in-flight criteria search is abandoned."""
        email_ctx["pending_search"] = None
        self._set_state(AssistantState.THINKING)
        self._speak_confirmation("Alright — just say the word if you change your mind.")

    def _handle_email_counts(self, parsed: dict, email_ctx: dict, live: bool) -> None:
        """Handles check_new (answered from cache) and check_now (live
        refresh first, for "check my email right now" style requests)."""
        if live:
            self._speak_confirmation("Let me check right now.")
            self._set_state(AssistantState.THINKING)
            self._email.refresh_now()

        overview = self._email.get_unread_overview()
        total = overview["total"]
        broken_accounts = [a["label"] for a in overview["per_account"] if a["last_error"]]

        if total == 0:
            reply = "No new emails right now."
        elif total == 1 and overview["latest"] is not None:
            e = overview["latest"]
            reply = f"You've got one unread email — from {e['sender_name']}, about \"{e['subject']}\"."
            # This email was just SPOKEN — make it the referent so a
            # follow-up "read it" resolves without a clarify round.
            email_ctx["last_listed"] = [e]
        else:
            reply = f"You've got {total} unread emails across your connected accounts."

        if broken_accounts:
            reply += f" Heads up — {', '.join(broken_accounts)} needs reconnecting."

        self._speak_confirmation(reply)

    def _handle_email_list(self, parsed: dict, email_ctx: dict) -> None:
        """
        Handles list_emails ("what came in today?") and from_person
        ("anything from Sarah?"). Fills email_ctx["last_listed"] with the
        SPOKEN subset (never more than what was said aloud) so a follow-up
        like "read the second one" resolves against what the boss actually
        heard.
        """
        sender = parsed["sender"]
        if parsed["intent"] == "from_person" and not sender:
            answer = self._ask_and_listen("From who?")
            sender = answer.strip() or None
            if not sender:
                self._set_state(AssistantState.THINKING)
                self._speak_confirmation("I didn't catch a name — never mind for now.")
                return

        self._set_state(AssistantState.THINKING)
        if self._email.needs_deep_search(parsed["timeframe"], parsed["days_back"]):
            self._speak_confirmation("Hold on — let me search back through your mail.")
            self._set_state(AssistantState.THINKING)
        emails = self._email.get_emails_for_scope(
            account_hint=parsed["account_hint"],
            sender=sender,
            subject_hint=parsed["subject_hint"],
            timeframe=parsed["timeframe"],
            days_back=parsed["days_back"],
            unread_only=parsed["unread_only"],
            limit=parsed["count"] or 10,
        )

        if not emails:
            who = f" from {sender}" if sender else ""
            email_ctx["last_listed"] = []
            self._speak_confirmation(f"I didn't find any emails{who} matching that.")
            return

        self._speak_email_batch(emails, email_ctx)

    def _speak_email_batch(self, emails: list[dict], email_ctx: dict) -> None:
        """
        Speaks a non-empty batch of emails using the shared "you've got N:
        from X about Y; ...and N more" convention, and stores the SPOKEN
        subset (never more than what was said aloud) into
        email_ctx["last_listed"] so a follow-up like "read the second one"
        resolves against what the boss actually heard. Caller handles the
        empty-result case itself (the right "nothing found" wording differs
        by caller).
        """
        email_ctx["last_listed"] = emails[:5]
        if len(emails) == 1:
            e = emails[0]
            reply = f"You've got one: from {e['sender_name']}, about \"{e['subject']}\"."
        else:
            parts = [f"from {e['sender_name']} about \"{e['subject']}\"" for e in emails[:5]]
            reply = f"You've got {len(emails)}: " + "; ".join(parts)
            if len(emails) > 5:
                reply += f", and {len(emails) - 5} more."
        self._speak_confirmation(reply)

    def _email_matches_hints(self, email: dict, sender: str | None, subject_hint: str | None) -> bool:
        """True when the email matches every provided cue (no cues = True).
        Subject comparison delegates to EmailManager's looser token-overlap
        match (a spoken topic paraphrase rarely appears as a contiguous
        substring of the real subject line)."""
        if sender:
            sender_lower = sender.lower()
            if (
                sender_lower not in email["sender_name"].lower()
                and sender_lower not in email["sender_email"].lower()
            ):
                return False
        return self._email.subject_hint_matches(subject_hint, email["subject"])

    @staticmethod
    def _merge_pending_search(prior: dict | None, parsed: dict) -> dict:
        """
        Merge one turn's classifier output into an in-flight "find this
        email" search's accumulated criteria: a new non-null field this turn
        overwrites the stored value; a field this turn left null keeps
        whatever was accumulated before. unread_only only ever strengthens
        (once asked for, stays asked for). This is the fix for repeating
        sender/topic/timeframe across turns without it ever narrowing the
        search - every criteria-driven read_email turn merges into this
        instead of re-deriving from just its own sentence.
        """
        base = dict(prior) if prior else {
            "sender": None, "subject_hint": None, "account_hint": None,
            "timeframe": None, "days_back": None, "unread_only": False, "rounds": 0,
        }
        for key in ("sender", "subject_hint", "account_hint", "timeframe", "days_back"):
            if parsed.get(key) is not None:
                base[key] = parsed[key]
        if parsed.get("unread_only"):
            base["unread_only"] = True
        return base

    @staticmethod
    def _scope_phrase(timeframe: str | None, days_back: int | None) -> str:
        """Short spoken phrase describing a time scope, e.g. 'in the last
        month'. "" when neither is set."""
        if days_back:
            if days_back >= 60:
                return f"in the last {days_back // 30} months"
            if days_back >= 14:
                return f"in the last {days_back // 7} weeks"
            return f"in the last {days_back} days"
        if timeframe == "today":
            return "today"
        if timeframe == "yesterday":
            return "yesterday"
        if timeframe == "this_week":
            return "this week"
        return ""

    def _missing_criteria_prompt(self, pending: dict) -> str:
        """
        Targeted re-ask naming what's already known (never repeats a
        question the boss already answered) and asking only for what's still
        missing. If sender AND subject are both known but nothing matched,
        asks about timing instead of repeating the same two facts again.
        """
        known = []
        if pending["sender"]:
            known.append(f"from {pending['sender']}")
        if pending["subject_hint"]:
            known.append(f"about \"{pending['subject_hint']}\"")
        scope_phrase = self._scope_phrase(pending["timeframe"], pending["days_back"])
        if scope_phrase:
            known.append(scope_phrase)

        missing = []
        if not pending["sender"]:
            missing.append("who it's from")
        if not pending["subject_hint"]:
            missing.append("what it's about")
        if not scope_phrase and not missing:
            missing.append("roughly when you got it")

        known_str = f" I've got {', '.join(known)}, but still couldn't find a match." if known else ""
        ask_str = (
            " Can you tell me " + " or ".join(missing) + "?"
            if missing else " Can you give me another detail to narrow it down?"
        )
        return f"I'm still not finding it.{known_str}{ask_str}"

    @staticmethod
    def _sounds_affirmative(text: str) -> bool:
        lowered = text.strip().lower()
        return any(
            word in lowered
            for word in ("yes", "yeah", "yep", "sure", "please do", "go ahead", "do that", "okay", "ok")
        )

    def _handle_unresolved_search_round(self, pending: dict, email_ctx: dict) -> None:
        """
        Called when a criteria-driven search round didn't resolve to exactly
        one email (zero matches, or a clarify answer that didn't pick one).
        Below MAX_SEARCH_REFINEMENT_ROUNDS, asks a targeted follow-up naming
        only what's still missing and keeps the accumulated criteria alive
        for the next turn (this IS the fix for the give-up path discarding
        everything the boss just said). At the cap, offers to just show
        everything matching what's known so far instead of asking forever.
        """
        pending["rounds"] += 1
        if pending["rounds"] <= MAX_SEARCH_REFINEMENT_ROUNDS:
            email_ctx["pending_search"] = pending
            # SPEAK the re-ask only — do NOT listen here. The turn loop's own
            # next listen captures the boss's answer as a fresh turn; the gate
            # reopens on the active pending_search and classify_intent
            # (active_search) routes it back into _handle_email_search to
            # merge + re-search. Using _ask_and_listen here would listen a
            # second time and discard that answer, leaving the turn loop to
            # open a phantom silent listen window (the double-listen bug).
            self._speak_confirmation(self._missing_criteria_prompt(pending))
            return

        # Capped fallback: relax the least reliable cue (topic/subject - the
        # one most prone to paraphrase mismatch) and default the window if
        # none was ever given, then offer to just show what's left.
        fallback_days_back = pending["days_back"] or VAGUE_TIMEFRAME_DAYS_BACK_DEFAULT
        scope_phrase = self._scope_phrase(pending["timeframe"], fallback_days_back)
        from_clause = f"from {pending['sender']}" if pending["sender"] else "everything"
        offer = f"I'm having trouble narrowing that down — want me to just list {from_clause}"
        offer += f" {scope_phrase} instead?" if scope_phrase else " instead?"

        answer = self._ask_and_listen(offer)
        self._set_state(AssistantState.THINKING)
        email_ctx["pending_search"] = None

        if not answer.strip() or not self._sounds_affirmative(answer):
            self._speak_confirmation("Alright — just say the word if you change your mind.")
            return

        emails = self._email.get_emails_for_scope(
            account_hint=pending["account_hint"], sender=pending["sender"],
            timeframe=pending["timeframe"] if not pending["days_back"] else None,
            days_back=fallback_days_back, unread_only=pending["unread_only"], limit=10,
        )
        if not emails:
            self._speak_confirmation("I still couldn't find anything matching that.")
            return
        self._speak_email_batch(emails, email_ctx)

    def _clarify_and_pick(
        self, matches: list[dict], email_ctx: dict, *, pending_search: dict | None = None
    ) -> dict | None:
        """
        Spoken disambiguation over a multi-match set. Speaks the top matches,
        listens, and resolves the answer — ordinals against the SPOKEN subset
        (the boss never heard the rest, and the classifier's numbered context
        must match what was said), sender/topic cues against the full set.
        Returns the chosen email, or None after having spoken the outcome
        itself (silence, dismissal, or an answer that didn't resolve).

        When `pending_search` is given (the criteria-driven search path, as
        opposed to disambiguating an already-known just-spoken batch), a
        round that doesn't resolve does NOT just apologize and drop
        everything the boss said - any new hints in the answer are merged
        into `pending_search` and routed to _handle_unresolved_search_round,
        which re-asks for only what's still missing (or offers the capped
        fallback) instead of discarding the answer. With the default
        `pending_search=None` (the bare-acknowledgment caller), behavior is
        byte-identical to before this existed.
        """
        spoken = matches[:3]
        email_ctx["last_listed"] = spoken
        options = "; ".join(
            f"from {e['sender_name']} about \"{e['subject']}\"" for e in spoken
        )
        answer = self._ask_and_listen(f"I found a few — {options}. Which one?")
        self._set_state(AssistantState.THINKING)

        if not answer.strip():
            if pending_search is not None:
                self._handle_unresolved_search_round(pending_search, email_ctx)
            else:
                self._speak_confirmation("Sorry, I didn't catch which one — ask me again anytime.")
            return None

        reparsed = self._email.classify_intent(answer, context_emails=spoken)
        if reparsed["intent"] == "dismiss":
            if pending_search is not None:
                email_ctx["pending_search"] = None
            self._speak_confirmation("Alright — just say the word if you change your mind.")
            return None

        if reparsed["ordinal"] and 1 <= reparsed["ordinal"] <= len(spoken):
            picked = spoken[reparsed["ordinal"] - 1]
            # Content cue beats a possibly-miscounted position (see
            # _handle_read_email's ordinal guard for the rationale).
            if not self._email_matches_hints(picked, reparsed["sender"], reparsed["subject_hint"]):
                hinted = [
                    e for e in matches
                    if self._email_matches_hints(e, reparsed["sender"], reparsed["subject_hint"])
                ]
                if hinted:
                    picked = hinted[0]
            return picked

        if reparsed["sender"] or reparsed["subject_hint"]:
            hinted = [
                e for e in matches
                if self._email_matches_hints(e, reparsed["sender"], reparsed["subject_hint"])
            ]
            if hinted:
                return hinted[0]

        if pending_search is not None:
            merged = self._merge_pending_search(pending_search, reparsed)
            self._handle_unresolved_search_round(merged, email_ctx)
        else:
            self._speak_confirmation("Sorry, I didn't catch which one — ask me again anytime.")
        return None

    def _read_email_aloud(self, target: dict, email_ctx: dict) -> None:
        """
        The single place any specific email is read aloud - so read-time
        sanitization (sanitize_body_text) lives in exactly one spot and can
        never be bypassed by a body that entered the cache before the
        sanitizer existed or before its char set was broadened. Clears the
        in-flight search (a resolved read ends the "find this email" story),
        reads short bodies verbatim, and summarizes long ones (a Sonnet
        summary is already clean prose, but sender/subject are still
        sanitized for the spoken framing).
        """
        email_ctx["pending_search"] = None
        # Remember what was just read so a follow-up question about "it" / "that
        # email" (email_question) can be answered directly without re-reading.
        email_ctx["last_read"] = target
        safe_sender = sanitize_body_text(target["sender_name"])
        safe_subject = sanitize_body_text(target["subject"])
        safe_body = sanitize_body_text(target["body"])
        self._set_state(AssistantState.THINKING)

        if len(safe_body) > 800:
            self._speak_confirmation("Give me a moment to read through it.")
            self._set_state(AssistantState.THINKING)
            summary = self._email.summarize_email(target)
            self._speak_confirmation(
                f"This one's a bit long, so here's the gist — email from "
                f"{safe_sender}, subject \"{safe_subject}\". {summary}"
            )
        else:
            self._speak_confirmation(
                f"Email from {safe_sender}, subject \"{safe_subject}\": {safe_body}"
            )

    def _handle_email_question(self, parsed: dict, email_ctx: dict, text: str) -> None:
        """
        Answer a targeted factual question about a specific email ("who sent
        it?", "what did they ask for?") directly from already-fetched content,
        instead of re-running the full read/summarize. Resolves which email
        the question is about: an ordinal against the last spoken list, else
        the last-read email ("it"/"that email"), else a single just-listed
        email. Costs one fast Haiku call, no re-fetch or search.
        """
        candidates: list[dict] = email_ctx.get("last_listed") or []
        ordinal = parsed["ordinal"]

        if ordinal and 1 <= ordinal <= len(candidates):
            target = candidates[ordinal - 1]
        elif email_ctx.get("last_read") is not None:
            target = email_ctx["last_read"]
        elif len(candidates) == 1:
            target = candidates[0]
        else:
            self._set_state(AssistantState.THINKING)
            self._speak_confirmation(
                "I'm not sure which email you mean — which one are you asking about?"
            )
            return

        self._set_state(AssistantState.THINKING)
        answer = self._email.answer_email_question(text, target)
        # The question keeps this email in focus for any further follow-up.
        email_ctx["last_read"] = target
        self._speak_confirmation(answer)

    def _handle_read_email(self, parsed: dict, email_ctx: dict) -> None:
        """
        Resolves which email to read, in priority order: an ordinal ("the
        second one") against the batch that was just SPOKEN (listed or
        announced); a bare acknowledgment ("yes, read it") against that same
        batch; then a criteria-driven search delegated to _handle_email_search
        (with auto_read=True). The ordinal and bare-ack branches resolve
        against what the boss just heard and do NOT touch pending_search - a
        fundamentally different interaction from searching by criteria.
        """
        candidates: list[dict] = email_ctx.get("last_listed") or email_ctx.get("announced") or []
        ordinal = parsed["ordinal"]
        sender = parsed["sender"]
        subject_hint = parsed["subject_hint"]
        days_back = parsed["days_back"]

        if ordinal and 1 <= ordinal <= len(candidates):
            target = candidates[ordinal - 1]
            # A content cue beats a possibly-miscounted position: "the last
            # one about Jarvis" means the newest Jarvis email, wherever the
            # classifier's ordinal landed. If the picked item doesn't match
            # the stated person/topic but another candidate does, trust the
            # cue.
            if not self._email_matches_hints(target, sender, subject_hint):
                hinted = [
                    e for e in candidates if self._email_matches_hints(e, sender, subject_hint)
                ]
                if hinted:
                    target = hinted[0]
            self._read_email_aloud(target, email_ctx)
            return

        if candidates and not sender and not subject_hint and not days_back:
            # Bare acknowledgment ("yes, please") against what was just spoken.
            if len(candidates) == 1:
                self._read_email_aloud(candidates[0], email_ctx)
            else:
                target = self._clarify_and_pick(candidates, email_ctx)
                if target is not None:
                    self._read_email_aloud(target, email_ctx)
                # else: clarify already spoke the outcome
            return

        # Criteria-driven search (auto-read the single match).
        self._handle_email_search(parsed, email_ctx, auto_read=True)

    def _handle_email_search(self, parsed: dict, email_ctx: dict, auto_read: bool) -> None:
        """
        Criteria-driven "find a specific email" search, shared by read_email
        (auto_read=True → read the single match) and topic-filtered
        list_emails/from_person (auto_read=False → list the match(es) but stay
        in the search so the boss can refine or ordinal-pick).

        Accumulates criteria across turns via email_ctx["pending_search"] so
        repeating or adding detail narrows the search instead of restarting
        each turn. Multiple matches: read-path clarifies, list-path lists and
        keeps the search alive. Zero matches: a targeted re-ask for whatever's
        still missing, up to MAX_SEARCH_REFINEMENT_ROUNDS, then a capped
        graceful fallback (see _handle_unresolved_search_round).
        """
        pending = self._merge_pending_search(email_ctx.get("pending_search"), parsed)

        if self._email.needs_deep_search(pending["timeframe"], pending["days_back"]):
            self._speak_confirmation("Hold on — let me search back through your mail.")
        self._set_state(AssistantState.THINKING)

        matches = self._email.get_emails_for_scope(
            account_hint=pending["account_hint"], sender=pending["sender"],
            subject_hint=pending["subject_hint"], timeframe=pending["timeframe"],
            days_back=pending["days_back"], unread_only=pending["unread_only"], limit=10,
        )
        if not matches and pending["subject_hint"]:
            # Lenient retry, cache AND deep alike: server-side IMAP SUBJECT /
            # Gmail subject: matching is word-order sensitive, so a correctly-
            # extracted paraphrase ("KPI monthly" vs. the real subject
            # "Monthly KPI Summary") can miss even though the topic is there.
            # Drop subject_hint from the query (sender/timeframe still narrow
            # it, so the deep-search message cap can't swallow the match) and
            # apply the looser token-overlap match locally over the broader
            # result.
            broader = self._email.get_emails_for_scope(
                account_hint=pending["account_hint"], sender=pending["sender"],
                timeframe=pending["timeframe"], days_back=pending["days_back"],
                unread_only=pending["unread_only"], limit=25,
            )
            matches = [
                e for e in broader
                if self._email.subject_hint_matches(pending["subject_hint"], e["subject"])
            ][:10]

        if len(matches) == 1:
            if auto_read:
                self._read_email_aloud(matches[0], email_ctx)
            else:
                # List intent: announce the single find, clear the search
                # (it resolved), leave it as last_listed so "read it" resolves.
                email_ctx["pending_search"] = None
                self._speak_email_batch(matches, email_ctx)
            return

        if len(matches) > 1:
            if auto_read:
                target = self._clarify_and_pick(matches, email_ctx, pending_search=pending)
                if target is not None:
                    self._read_email_aloud(target, email_ctx)
                # else: clarify routed to re-ask / fallback / dismiss already
            else:
                # List intent: speak the matches and KEEP the search alive so
                # "no, the one about X" refines and "the second one" resolves
                # against last_listed. A successful list is not a failed round.
                email_ctx["pending_search"] = pending
                self._speak_email_batch(matches, email_ctx)
            return

        # Zero matches → targeted re-ask / capped fallback.
        self._handle_unresolved_search_round(pending, email_ctx)

    def _handle_email_reasoning(self, parsed: dict, intent: str) -> None:
        """
        Handles summarize (inbox scope) / action_needed / meetings_mentioned
        - all require Claude reasoning over email content, not just metadata.
        If the boss didn't specify how far back to look, ASK rather than
        guess (per requirement); if still ambiguous after asking once,
        default to today's unread and say the assumption aloud.
        """
        timeframe = parsed["timeframe"]
        count = parsed["count"]
        days_back = parsed["days_back"]

        if not timeframe and not count and not days_back:
            answer = self._ask_and_listen(
                "Sure — how far back should I look, or how many recent emails should I check?"
            )
            self._set_state(AssistantState.THINKING)
            scope = self._email.classify_scope(answer)
            timeframe = scope["timeframe"]
            count = scope["count"]
            days_back = scope["days_back"]

        self._set_state(AssistantState.THINKING)
        if not timeframe and not count and not days_back:
            timeframe = "today"
            assumed_note = "I'll go with today's unread emails. "
        else:
            assumed_note = ""

        if self._email.needs_deep_search(timeframe, days_back):
            self._speak_confirmation("Hold on — let me search back through your mail.")
            self._set_state(AssistantState.THINKING)
        emails = self._email.get_emails_for_scope(
            account_hint=parsed["account_hint"], timeframe=timeframe,
            days_back=days_back, limit=count or 25,
        )

        if not emails:
            self._speak_confirmation(assumed_note + "I didn't find any emails matching that.")
            return

        self._speak_confirmation(assumed_note + "Give me a moment to look through them.")
        self._set_state(AssistantState.THINKING)

        if intent == "action_needed":
            question = "Is there anything in these emails that needs action or a response?"
        elif intent == "meetings_mentioned":
            question = "Are there any meetings, calls, or appointments mentioned in these emails?"
        else:
            question = "Summarize these emails."

        answer_text = self._email.reason_over_emails(question, emails)
        self._speak_confirmation(answer_text)

    def _handle_send_decline(self) -> None:
        self._set_state(AssistantState.THINKING)
        self._speak_confirmation(
            "I can't send emails yet — that's coming soon. I can read or "
            "summarize them for you in the meantime."
        )

    # ── Wake word callback ────────────────────────────────────────────────────

    def _on_wake_word(self) -> None:
        """
        Called by WakeWordDetector from its internal detector thread. Must be
        non-blocking — just signals the main loop. Sets the pending flag
        BEFORE the event, so a dispatch-loop pass woken by this event can
        never observe the event without also observing the flag.
        """
        self._emit_event("wake_word_detected")
        self._wake_word_pending = True
        self._wake_event.set()

    def _on_new_mail(self) -> None:
        """
        Called by EmailManager (from the poll thread) when a cycle finds new
        mail. Must be non-blocking, like _on_wake_word — just pokes the main
        loop; the loop discovers the pending announcement itself by calling
        claim_announcement() at the top of its next pass.
        """
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

    def _run_conversation(self, announcement: str | None = None) -> None:
        """
        Full conversation lifecycle: STT → LLM → TTS, with follow-up support.
        Runs in its own thread. Restarts the wake detector when done, unless
        a stop() happened mid-conversation (see the finally block).

        If `announcement` is given, this conversation opens with that
        proactive email nudge instead of the plain greeting - spoken via
        _speak_confirmation, so (unlike the event-silent plain greeting) it
        emits an llm_response the frontend transcript can show - and the
        first listen uses the follow-up timeout rather than STT's default,
        since the boss wasn't explicitly asked to speak.
        """
        email_ctx: dict = {"last_listed": [], "announced": [], "pending_search": None, "last_read": None}
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

            # Greet the user once per new conversation, before any STT — or,
            # if a proactive email announcement is pending, open with that.
            if announcement is not None:
                logger.info("[Announcement] Speaking proactive email nudge…")
                self._speak_confirmation(announcement)
                email_ctx["announced"] = self._email.get_announced_context()
            else:
                logger.info("[Greeting] Speaking opening line…")
                self._tts.speak("Hey, how can I help you today?")

            first_turn = True

            while True:
                # ── 3a. Listen and transcribe ─────────────────────────────────
                self._set_state(AssistantState.LISTENING)
                self._emit_event("listening_started")

                if first_turn and announcement is None:
                    logger.info("[Listening] Speak now…")
                    text = self._stt.listen_and_transcribe()
                elif first_turn:
                    logger.info(
                        f"[Listening after announcement] Speak within "
                        f"{FOLLOWUP_SILENCE_TIMEOUT:.0f}s or stay silent to end…"
                    )
                    text = self._stt.listen_and_transcribe(
                        initial_silence_timeout=FOLLOWUP_SILENCE_TIMEOUT
                    )
                else:
                    logger.info(
                        f"[Listening for follow-up] "
                        f"Speak within {FOLLOWUP_SILENCE_TIMEOUT:.0f}s or stay silent to end…"
                    )
                    text = self._stt.listen_and_transcribe(
                        initial_silence_timeout=FOLLOWUP_SILENCE_TIMEOUT
                    )

                first_turn = False

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

                # ── 3d. Email-related request? ─────────────────────────────────
                # Same rationale as the memory intercept above: handled
                # entirely here, never sent to self._llm. The gate opens on
                # an email keyword OR on live conversation context (a just-
                # announced or just-listed batch), so context-only follow-ups
                # like "yes, please" and "the last one about X" get classified.
                if self._maybe_handle_email_query(text, email_ctx):
                    continue

                # ── 3e. Get Claude's response ─────────────────────────────────
                self._set_state(AssistantState.THINKING)
                logger.info("[Thinking] Sending to Claude…")
                reply = self._llm.get_response(text, memory_context=memory_context)
                self._emit_event("llm_response", text=reply)

                # ── 3f. Speak the response ────────────────────────────────────
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

            # ── 5. Restart the wake detector — unless stop() ran while this
            #      conversation was in flight, in which case it already tore
            #      the detector down and restarting it here would resurrect
            #      the mic thread after shutdown.
            self._set_state(AssistantState.IDLE)
            self._emit_event("idle")
            self._wake_word_pending = False
            self._wake_event.clear()  # discard any stale wake events
            if self._running:
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
