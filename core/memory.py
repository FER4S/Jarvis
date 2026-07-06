# ─────────────────────────────────────────────────────────────────────────────
#  core/memory.py – Persistent memory for Jarvis
#  Local JSON store of what Jarvis knows about the boss: profile, people,
#  events, and facts. A short summary is injected into the LLM system prompt
#  each conversation; a background Sonnet call extracts new memorable content
#  from the transcript once a conversation ends.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime

# Ensure the project root is on sys.path so `import config` works when this
# file is run directly (e.g. `python core/memory.py`).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import anthropic
from loguru import logger

import config

# ── Storage location ─────────────────────────────────────────────────────────
_DATA_DIR: str = os.path.join(_PROJECT_ROOT, "data")
_MEMORY_FILE: str = os.path.join(_DATA_DIR, "memory.json")

# ── Models ────────────────────────────────────────────────────────────────────
# Heavier background-task tier (memory extraction, future briefings/summaries)
# - distinct from the realtime-voice haiku model in core/llm.py.
EXTRACTION_MODEL_ID: str = "claude-sonnet-5"
EXTRACTION_MAX_TOKENS: int = 1024
EXTRACTION_TIMEOUT: float = 15.0   # background thread - more slack than LLMEngine's 10s

# Same fast model core/llm.py uses, for the one-off explicit-memory cleanup call.
FACT_CLEANUP_MODEL_ID: str = "claude-haiku-4-5-20251001"
FACT_CLEANUP_MAX_TOKENS: int = 100
FACT_CLEANUP_TIMEOUT: float = 10.0  # runs on the conversation thread - matches LLMEngine's own timeout

# Explicit-memory structuring (fact-vs-event classify + event match) and the
# same-or-new answer classify — both realtime, reuse the Haiku tier above.
STRUCTURE_KIND_FACT: str = "fact"
STRUCTURE_KIND_EVENT: str = "event"
STRUCTURE_MAX_TOKENS: int = 300     # classify+structure JSON, larger than a 1-sentence rewrite
CLASSIFY_MAX_TOKENS: int = 10       # "same"/"new" is one word

# Cap on how many (most-recent) existing events are rendered into the prompts
# that need them as context, to bound token cost.
_EVENTS_CONTEXT_CAP: int = 40

# ── get_context_summary() token budget ───────────────────────────────────────
# No tokenizer dependency is available; ~4 chars/token is a standard
# conservative heuristic for English prose (errs toward cutting a bit early
# rather than overshooting the caller's system-prompt token cost).
_CHARS_PER_TOKEN: float = 4.0
_SUMMARY_TOKEN_BUDGET: int = 300
_SUMMARY_CHAR_BUDGET: int = int(_SUMMARY_TOKEN_BUDGET * _CHARS_PER_TOKEN)  # 1200 chars

# extract_and_save() no-ops below this many user turns - not worth an API call.
_MIN_USER_TURNS_FOR_EXTRACTION: int = 2


def parse_json_object(raw_text: str) -> dict | None:
    """
    Best-effort parse of a JSON object out of raw_text, tolerating markdown
    code fences and stray commentary around the JSON. Returns None if no
    JSON object could be located/parsed, or the parsed value isn't a dict.
    """
    text = raw_text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

    return parsed if isinstance(parsed, dict) else None


def extract_response_text(response) -> str:
    """
    Concatenate every text content block in an Anthropic API response.

    response.content[0].text is NOT reliably the reply: Claude can (and
    empirically does, observed live against claude-sonnet-5) return a
    non-text block first (e.g. a thinking block, which has no .text), making
    content[0].text None. Scanning every block and keeping only type=="text"
    ones is the robust way to pull the actual reply out of a response.
    """
    parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    return "".join(parts).strip()


def _clean_str(value) -> str:
    """
    Coerce a possibly-None/non-string value into a stripped string. Needed
    because the extraction model sometimes emits JSON `null` for an optional
    field (e.g. {"notes": null}) instead of an empty string - dict.get(key, "")
    only applies its default when the key is *missing*, not when it's present
    with a None value, so a bare `.strip()` on that would raise.
    """
    if value is None:
        return ""
    return str(value).strip()


def _today_str() -> str:
    """
    Current date, with weekday name, in config.TIMEZONE - e.g.
    "Tuesday, July 07, 2026". Gives the fact-cleanup and extraction
    prompts a real "today" so they can resolve relative time references
    (today/tomorrow/next Tuesday/etc.) into absolute dates instead of
    storing words that go stale.
    """
    return datetime.now(config.TIMEZONE).strftime("%A, %B %d, %Y")


def _events_context_block(events: list[dict]) -> str:
    """
    Render existing events as prompt context so a model can match a new
    mention against them and copy a description verbatim. One line per event:
    `- "<description>" (date: <date or "none">)`. Returns "(none)" when empty.
    Only the most recent _EVENTS_CONTEXT_CAP events are shown, to bound tokens.
    """
    recent = events[-_EVENTS_CONTEXT_CAP:]
    lines = []
    for e in recent:
        description = _clean_str(e.get("description"))
        if not description:
            continue
        date = _clean_str(e.get("date")) or "none"
        lines.append(f'- "{description}" (date: {date})')
    return "\n".join(lines) if lines else "(none)"


class MemoryManager:
    """
    Local JSON-backed memory store for Jarvis.

    Call load() once at startup, then get_context_summary() at the start of
    each conversation to feed LLMEngine.get_response(). add_explicit_memory()
    handles "Jarvis, remember that..." commands; extract_and_save() handles
    background extraction from a finished conversation's transcript.

    Usage:
        memory = MemoryManager()
        memory.load()
        summary = memory.get_context_summary()
    """

    def __init__(self, api_key: str = config.ANTHROPIC_API_KEY) -> None:
        if not api_key:
            logger.warning(
                "ANTHROPIC_API_KEY is not set. "
                "Memory extraction calls will fail until it is added to .env."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._state: dict = self._empty_state()
        self._lock = threading.Lock()

    @staticmethod
    def _empty_state() -> dict:
        return {
            "onboarding_complete": False,
            "profile": {},
            "people": [],
            "events": [],
            "facts": [],
            "last_updated": "",
        }

    # ── Storage ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load memory from disk, creating the file/directory if missing."""
        with self._lock:
            if not os.path.isdir(_DATA_DIR):
                os.makedirs(_DATA_DIR, exist_ok=True)

            if not os.path.isfile(_MEMORY_FILE):
                logger.info(f"No memory file found at {_MEMORY_FILE} - creating a new one.")
                self._state = self._empty_state()
                self._write_locked()
                return

            try:
                with open(_MEMORY_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                merged = self._empty_state()
                merged.update(loaded)
                self._state = merged
                logger.success(f"Memory loaded from {_MEMORY_FILE}.")
            except (json.JSONDecodeError, OSError) as exc:
                logger.error(
                    f"Failed to load memory file ({exc}) - starting from an empty "
                    f"in-memory state. The file on disk is left untouched until "
                    f"the next save()."
                )
                self._state = self._empty_state()

    def save(self) -> None:
        """Write the current memory state to disk."""
        with self._lock:
            self._write_locked()

    def _write_locked(self) -> None:
        """Actual write logic. Caller must already hold self._lock."""
        self._state["last_updated"] = datetime.now(config.TIMEZONE).isoformat()
        os.makedirs(_DATA_DIR, exist_ok=True)
        tmp_path = _MEMORY_FILE + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, _MEMORY_FILE)  # atomic on both Windows and POSIX
        except OSError as exc:
            logger.error(f"Failed to save memory file: {exc}")

    # ── Context summary ───────────────────────────────────────────────────────

    def get_context_summary(self) -> str:
        """
        Build a short plain-text summary of everything in memory, suitable for
        injection into an LLM system prompt. Stays under ~300 tokens (via a
        char-based heuristic). Returns "" if memory has nothing meaningful yet.

        Profile/people are included in full (bounded in practice). Events/facts
        - the two lists that grow unboundedly over months of use - are built
        recency-first: when space-constrained, the OLDEST entries are dropped,
        not the newest, since a naive "join everything then truncate the end"
        approach would silently keep ancient facts forever while dropping
        whatever was just learned.
        """
        with self._lock:
            profile = dict(self._state.get("profile") or {})
            people = list(self._state.get("people") or [])
            events = list(self._state.get("events") or [])
            facts = list(self._state.get("facts") or [])

        if not profile and not people and not events and not facts:
            return ""

        sections: list[str] = []
        for section in (self._format_profile_section(profile), self._format_people_section(people)):
            if section:
                sections.append(section)

        def _remaining_budget() -> int:
            joined_so_far = " | ".join(sections)
            separator_if_more = 3 if sections else 0  # " | "
            return _SUMMARY_CHAR_BUDGET - len(joined_so_far) - separator_if_more

        event_strs = [
            f"{e.get('description', '?')} ({e.get('date', 'date unknown')})" for e in events
        ]
        events_section = self._recency_first_section(
            "Upcoming/relevant events", event_strs, _remaining_budget()
        )
        if events_section:
            sections.append(events_section)

        facts_section = self._recency_first_section("Other known facts", facts, _remaining_budget())
        if facts_section:
            sections.append(facts_section)

        summary = " | ".join(sections)

        # Defense-in-depth backstop - the section budgeting above should
        # already guarantee this, but a hard cap costs nothing extra.
        if len(summary) > _SUMMARY_CHAR_BUDGET:
            summary = summary[:_SUMMARY_CHAR_BUDGET].rstrip() + "..."

        return summary

    @staticmethod
    def _format_profile_section(profile: dict) -> str:
        if not profile:
            return ""
        if "raw_qa" in profile and len(profile) == 1:
            return f"About the user: {profile['raw_qa']}"
        profile_bits = "; ".join(f"{k}: {v}" for k, v in profile.items() if v)
        return f"About the user: {profile_bits}" if profile_bits else ""

    @staticmethod
    def _format_people_section(people: list[dict]) -> str:
        if not people:
            return ""
        people_bits = "; ".join(
            f"{p.get('name', '?')} ({p.get('notes', '')})" if p.get("notes") else p.get("name", "?")
            for p in people
        )
        return f"Key people: {people_bits}"

    @staticmethod
    def _recency_first_section(label: str, items: list[str], budget: int) -> str:
        """
        Build "label: item; item; ..." keeping the MOST RECENT items (from the
        end of `items`) that fit within `budget` characters, dropping older
        ones first when space-constrained.
        """
        if budget <= 0 or not items:
            return ""
        kept: list[str] = []
        used = len(label) + 2  # ": "
        for item in reversed(items):
            addition = len(item) + (2 if kept else 0)  # "; " separator
            if used + addition > budget:
                break
            kept.append(item)
            used += addition
        if not kept:
            return ""
        kept.reverse()  # restore chronological order for readability
        return f"{label}: {'; '.join(kept)}"

    # ── Explicit memory ("Jarvis, remember that...") ─────────────────────────

    def add_explicit_memory(self, text: str) -> None:
        """
        Add a raw fact string to the facts list and save immediately. Used for
        "Jarvis, remember that..." voice commands. Does no LLM work itself -
        a thin, fast, always-succeeds append (see clean_fact_text() for the
        companion Haiku cleanup step assistant.py runs before calling this).
        """
        text = text.strip()
        if not text:
            logger.warning("add_explicit_memory() called with empty text - skipping.")
            return

        with self._lock:
            facts = self._state.setdefault("facts", [])
            if any(_clean_str(existing).lower() == text.lower() for existing in facts):
                logger.debug(f"Fact already present, skipping duplicate: '{text}'")
            else:
                facts.append(text)
            self._write_locked()

        logger.success(f"Explicit memory saved: '{text}'")

    def clean_fact_text(self, raw_text: str) -> str:
        """
        One-off Haiku call to normalize a raw spoken fragment (already
        stripped of its "remember that"/"don't forget" trigger phrase) into a
        clean, well-formed fact statement. Never raises - falls back to
        raw_text.strip() unchanged if the API call fails for any reason.
        """
        raw_text = raw_text.strip()
        if not raw_text:
            return raw_text
        try:
            response = self._client.messages.create(
                model=FACT_CLEANUP_MODEL_ID,
                max_tokens=FACT_CLEANUP_MAX_TOKENS,
                timeout=FACT_CLEANUP_TIMEOUT,
                system=(
                    f"Today is {_today_str()}. Rewrite the following fragment as a "
                    "single, clean, third-person factual statement suitable for a "
                    "personal memory note. Keep it concise - one sentence.\n\n"
                    "If the fragment contains a relative time/day reference (e.g. "
                    "today, tomorrow, yesterday, next Tuesday, this Friday, in two "
                    "weeks), resolve it to an absolute date using the date given "
                    "above and state that date explicitly in the rewritten sentence "
                    "instead of the relative word. Otherwise, if the fragment has no "
                    "relative time/day reference - including if it only has a bare "
                    "time-of-day like \"5pm\" with no relative day word, or if it "
                    "already names an absolute date - leave that part exactly as "
                    "stated and do not add, invent, or infer any date that is not "
                    "already implied by a relative reference in the fragment.\n\n"
                    "Return ONLY the rewritten sentence, no quotes, no preamble, no "
                    "explanation."
                ),
                messages=[{"role": "user", "content": raw_text}],
            )
            cleaned = extract_response_text(response)
            return cleaned if cleaned else raw_text
        except Exception as exc:
            logger.warning(f"clean_fact_text() failed ({exc}) - using raw text unmodified.")
            return raw_text

    def structure_explicit_memory(self, raw_text: str) -> dict:
        """
        One Haiku call that decides whether an explicit "remember that…" command
        is a general FACT or a dated EVENT; for events, produces a normalized
        {description, date} (relative dates resolved against _today_str()) and
        whether it matches an existing event.

        Returns a fully-defaulted dict (callers never KeyError):
            {"kind": "event"|"fact", "description": str, "date": str,
             "match_description": str}
        - description/date are "" for facts.
        - match_description is "" or the EXACT existing description it resembles.

        Never raises. On any failure/unparseable/wrong-shape, falls back to
        {"kind": "fact", ...} so the raw text is still saved via the always-
        succeeds fact path (no event data invented, nothing lost).
        """
        fallback = {"kind": STRUCTURE_KIND_FACT, "description": "", "date": "", "match_description": ""}

        raw_text = raw_text.strip()
        if not raw_text:
            return fallback

        # Snapshot events under the lock, then build the prompt and make the
        # network call OUTSIDE the lock (no network call ever holds self._lock).
        with self._lock:
            events_snapshot = list(self._state.get("events") or [])
        existing_descriptions_lower = {
            _clean_str(e.get("description")).lower() for e in events_snapshot
        }

        try:
            response = self._client.messages.create(
                model=FACT_CLEANUP_MODEL_ID,
                max_tokens=STRUCTURE_MAX_TOKENS,
                timeout=FACT_CLEANUP_TIMEOUT,
                system=(
                    f"Today is {_today_str()}. You are the structuring step of a "
                    "personal voice assistant's memory. The user just gave an explicit "
                    "\"remember this\" command. Decide whether it describes a dated/"
                    "scheduled EVENT (a meeting, appointment, call, deadline, trip, "
                    "reminder tied to a day/time) or a general FACT (a preference or "
                    "standing detail with no specific occurrence time).\n\n"
                    "Respond with ONLY a single JSON object (no markdown fences, no "
                    "commentary) with exactly these keys:\n"
                    "- \"kind\": either \"event\" or \"fact\".\n"
                    "- \"description\": if kind is \"event\", a short third-person "
                    "description with NO date/time words in it (e.g. \"Meeting with "
                    "Dana\", \"Dentist appointment\"); if kind is \"fact\", \"\".\n"
                    "- \"date\": if kind is \"event\", the event's date/time as stated; "
                    "if kind is \"fact\", \"\". If the event has a relative time/day "
                    "reference (today, tomorrow, next Tuesday, this Friday, in two "
                    "weeks), resolve it to an absolute ISO 8601 date (YYYY-MM-DD) using "
                    "today's date above and put that here instead of the relative "
                    "phrase; you may append a bare time-of-day (e.g. \"2026-07-14 at 5 "
                    "p.m.\"). If only a bare time-of-day was given with no day word, "
                    "keep just that time. If no date/time was given, \"\".\n"
                    "- \"match_description\": if kind is \"event\" AND it clearly refers "
                    "to the SAME real-world event as one already in the existing-events "
                    "list below (same meeting/appointment, possibly rescheduled or "
                    "reworded), copy that existing event's description string EXACTLY, "
                    "character for character. Otherwise \"\".\n\n"
                    "Existing events already in memory:\n"
                    f"{_events_context_block(events_snapshot)}\n\n"
                    "Only set match_description when you are confident it is the SAME "
                    "underlying event, not merely a similar kind of event. Do not "
                    "invent information that was not stated."
                ),
                messages=[{"role": "user", "content": raw_text}],
            )
            parsed = parse_json_object(extract_response_text(response))
            if not isinstance(parsed, dict):
                return fallback

            kind = _clean_str(parsed.get("kind")).lower()
            if kind != STRUCTURE_KIND_EVENT:
                return fallback  # whitelist: anything not "event" is a fact

            description = _clean_str(parsed.get("description"))
            if not description:
                return fallback  # event with no description → treat as a fact

            date = _clean_str(parsed.get("date"))
            match_description = _clean_str(parsed.get("match_description"))
            # Defend against a hallucinated match: only honor it if it actually
            # names an existing event (case-insensitive).
            if match_description and match_description.lower() not in existing_descriptions_lower:
                match_description = ""

            return {
                "kind": STRUCTURE_KIND_EVENT,
                "description": description,
                "date": date,
                "match_description": match_description,
            }
        except Exception as exc:
            logger.warning(f"structure_explicit_memory() failed ({exc}) - treating as a plain fact.")
            return fallback

    def add_event(self, description: str, date: str, force_new: bool = False) -> None:
        """
        Append a new event {description, date} and save immediately. Thin,
        never raises.

        By default (force_new=False), if an event with this exact
        (case-insensitive) description already exists, its date is updated in
        place instead of appending — a safety net so a stale/uncertain add
        can't create an accidental duplicate. Pass force_new=True to always
        append a separate entry regardless of collision: used when the boss
        has *explicitly* said this is a new, different event, so the dedup net
        must not silently override that answer.
        """
        description = _clean_str(description)
        if not description:
            logger.warning("add_event() called with empty description - skipping.")
            return
        date = _clean_str(date)

        with self._lock:
            events = self._state.setdefault("events", [])
            if not force_new:
                for existing in events:
                    if _clean_str(existing.get("description")).lower() == description.lower():
                        if date:
                            existing["date"] = date
                        self._write_locked()
                        logger.success(f"Event already present, updated date in place: '{description}'")
                        return
            events.append({"description": description, "date": date})
            self._write_locked()

        logger.success(
            f"Event saved{' (forced new entry)' if force_new else ''}: "
            f"'{description}' (date: '{date}')"
        )

    def update_event_date(
        self, match_description: str, new_date: str, new_description: str = ""
    ) -> bool:
        """
        Update the first event whose description case-insensitively equals
        match_description. Returns True if a match was found and updated, False
        otherwise (so the caller can fall back to add_event on a stale key).

        Date guard: only overwrite the date when new_date is non-empty (a vague
        "yeah, same one" must not blank a good stored date). Description is only
        overwritten when new_description is non-empty and actually differs.
        """
        key = _clean_str(match_description).lower()
        if not key:
            return False
        new_date = _clean_str(new_date)
        new_description = _clean_str(new_description)

        with self._lock:
            events = self._state.get("events") or []
            for existing in events:
                if _clean_str(existing.get("description")).lower() == key:
                    if new_date:
                        existing["date"] = new_date
                    if new_description and new_description.lower() != key:
                        existing["description"] = new_description
                    self._write_locked()
                    logger.success(f"Event updated: '{match_description}' -> date '{new_date}'")
                    return True

        logger.debug(f"update_event_date(): no event matching '{match_description}' - caller should add new.")
        return False

    def classify_same_or_new(self, answer_text: str) -> str:
        """
        Interpret the boss's spoken answer to the "same event, or new?"
        clarifying question. Returns "same", "new", or "unclear":
        - "same": clearly the same event (→ caller updates it in place).
        - "new": clearly a different, separate event — an EXPLICIT answer the
          caller honors by force-appending a new entry even on a description
          collision.
        - "unclear": noncommittal/ambiguous/silent, or any failure. The caller
          still records the event, but WITHOUT forcing, so the dedup safety net
          can guard against a stray duplicate on a non-answer.

        Never raises. Silence and errors both resolve to "unclear" so only a
        confident, explicit "new" ever forces a separate entry.
        """
        if not answer_text.strip():
            return "unclear"  # silence is not an explicit answer; no API call

        try:
            response = self._client.messages.create(
                model=FACT_CLEANUP_MODEL_ID,
                max_tokens=CLASSIFY_MAX_TOKENS,
                timeout=FACT_CLEANUP_TIMEOUT,
                system=(
                    "The user was asked whether something they just mentioned is the "
                    "SAME event they mentioned before (e.g. just rescheduled or "
                    "reworded) or a NEW, different event. Read their reply and answer "
                    "with ONLY one lowercase word:\n"
                    "- \"same\" if they clearly mean it is the same event,\n"
                    "- \"new\" if they clearly mean it is a different, separate event,\n"
                    "- \"unclear\" if their reply is noncommittal, ambiguous, or "
                    "doesn't answer the question.\n"
                    "Output only that one word, nothing else."
                ),
                messages=[{"role": "user", "content": answer_text}],
            )
            verdict = extract_response_text(response).strip().lower()
            if verdict.startswith("same"):
                return "same"
            if verdict.startswith("new"):
                return "new"
            return "unclear"
        except Exception as exc:
            logger.warning(f"classify_same_or_new() failed ({exc}) - defaulting to 'unclear'.")
            return "unclear"

    # ── Background extraction ─────────────────────────────────────────────────

    def extract_and_save(self, conversation: list[dict]) -> None:
        """
        Extract memorable facts/people/events from a finished conversation and
        merge them into memory. `conversation` is the list[dict] shape
        LLMEngine.history returns (role/content dicts).

        No-ops if the conversation has fewer than _MIN_USER_TURNS_FOR_EXTRACTION
        user turns. Never raises - designed to run safely from a background
        daemon thread at conversation teardown.
        """
        user_turn_count = sum(1 for turn in conversation if turn.get("role") == "user")
        if user_turn_count < _MIN_USER_TURNS_FOR_EXTRACTION:
            logger.debug(
                f"extract_and_save(): only {user_turn_count} user turn(s) - "
                f"skipping extraction (need >= {_MIN_USER_TURNS_FOR_EXTRACTION})."
            )
            return

        try:
            transcript = "\n".join(
                f"{turn.get('role', '?').upper()}: {turn.get('content', '')}"
                for turn in conversation
            )

            # Snapshot existing events under the lock so the extraction model can
            # decide update-vs-new. The authoritative re-read happens again under
            # the lock in _merge_extracted; this copy is only for the prompt.
            with self._lock:
                events_snapshot = list(self._state.get("events") or [])

            extraction_system_prompt = (
                f"Today is {_today_str()}. You analyze a conversation transcript "
                "between a voice assistant and its user, and extract any durable, "
                "memorable information about the user's life, work, or preferences "
                "that would be useful to remember in future conversations. Ignore "
                "small talk, one-off questions (e.g. what time is it), and anything "
                "not meaningfully new.\n\n"
                "Respond with ONLY a single JSON object (no markdown fences, no "
                "commentary) matching exactly this shape: an object with three keys "
                "people (list of objects with name and notes string fields), "
                "events (list of objects with description, date, and updates string "
                "fields), and facts (list of plain strings).\n\n"
                "Use empty arrays for any category with nothing to report.\n\n"
                "If a fact, event description, or event date contains a relative "
                "time/day reference (e.g. today, tomorrow, yesterday, next Tuesday, "
                "this Friday, in two weeks), resolve it to an absolute ISO 8601 date "
                "(YYYY-MM-DD) using the real today's-date given above, and write "
                "that absolute date into the text/field in place of the relative "
                "phrase. Otherwise, if there is no relative time/day reference - "
                "including a bare time-of-day like \"5pm\" with no relative day "
                "word, or a date already stated absolutely - leave it exactly as "
                "said and do not add, invent, or infer a date that is not already "
                "implied by a relative reference in the transcript. An event with no "
                "date mentioned at all should have an empty string \"\" for its date "
                "field, not a guessed or invented one. Do not invent information not "
                "present in the transcript.\n\n"
                "The user already has these events stored in memory:\n"
                f"{_events_context_block(events_snapshot)}\n"
                "For each event you output, add an \"updates\" field. If the event is "
                "the SAME real-world event as one stored above but with a changed/"
                "newly-clarified date/time (or a reworded description of that same "
                "event), set \"updates\" to that existing event's description string "
                "copied EXACTLY, character for character, and put the new date in "
                "\"date\". If it is genuinely new, set \"updates\" to \"\". Only use "
                "\"updates\" when confident it is the same underlying event, not "
                "merely a similar kind of event."
            )

            response = self._client.messages.create(
                model=EXTRACTION_MODEL_ID,
                max_tokens=EXTRACTION_MAX_TOKENS,
                timeout=EXTRACTION_TIMEOUT,
                system=extraction_system_prompt,
                messages=[{"role": "user", "content": transcript}],
            )

            raw_text = extract_response_text(response)
            extracted = self._parse_extraction_json(raw_text)
            if extracted is None:
                logger.warning(
                    "extract_and_save(): could not parse a usable JSON object from "
                    "the extraction response - discarding this extraction pass."
                )
                return

            self._merge_extracted(extracted)
            logger.success("extract_and_save(): memory updated from conversation.")

        except Exception as exc:
            logger.error(f"extract_and_save() failed, memory not updated: {exc}")

    @staticmethod
    def _parse_extraction_json(raw_text: str) -> dict | None:
        """Parse the extraction response into a dict with people/events/facts list keys."""
        parsed = parse_json_object(raw_text)
        if not isinstance(parsed, dict):
            return None

        return {
            "people": parsed.get("people") if isinstance(parsed.get("people"), list) else [],
            "events": parsed.get("events") if isinstance(parsed.get("events"), list) else [],
            "facts": parsed.get("facts") if isinstance(parsed.get("facts"), list) else [],
        }

    def _merge_extracted(self, extracted: dict) -> None:
        """Merge newly-extracted people/events/facts into self._state, deduplicating, then save."""
        with self._lock:
            existing_people = self._state.setdefault("people", [])
            existing_events = self._state.setdefault("events", [])
            existing_facts = self._state.setdefault("facts", [])

            # People: dedup by case-insensitive name match; merge notes on repeat.
            existing_names_lower = {_clean_str(p.get("name")).lower() for p in existing_people}
            for person in extracted["people"]:
                if not isinstance(person, dict):
                    continue
                name = _clean_str(person.get("name"))
                if not name:
                    continue
                name_lower = name.lower()
                if name_lower in existing_names_lower:
                    for existing in existing_people:
                        if _clean_str(existing.get("name")).lower() == name_lower:
                            new_notes = _clean_str(person.get("notes"))
                            old_notes = _clean_str(existing.get("notes"))
                            if new_notes and new_notes.lower() not in old_notes.lower():
                                existing["notes"] = f"{old_notes}; {new_notes}".strip("; ")
                            break
                else:
                    existing_people.append({"name": name, "notes": _clean_str(person.get("notes"))})
                    existing_names_lower.add(name_lower)

            # Events: update in place when the model flags an update (matched by
            # exact existing-description key), else fall back to exact-description
            # dedup, else append. Reads the authoritative existing_events list so
            # an interleaved Path-1 write is respected. Legacy events (no
            # "updates" field) are read only via .get() and never retroactively
            # merged - only extracted->existing matches act here.
            existing_descriptions_lower = {_clean_str(e.get("description")).lower() for e in existing_events}
            for event in extracted["events"]:
                if not isinstance(event, dict):
                    continue
                description = _clean_str(event.get("description"))
                if not description:
                    continue
                date = _clean_str(event.get("date"))
                updates_key = _clean_str(event.get("updates")).lower()

                target = None
                if updates_key:  # 1) explicit update by stable description key
                    for existing in existing_events:
                        if _clean_str(existing.get("description")).lower() == updates_key:
                            target = existing
                            break
                if target is None:  # 2) safety net: exact-description dedup
                    for existing in existing_events:
                        if _clean_str(existing.get("description")).lower() == description.lower():
                            target = existing
                            break

                if target is not None:
                    if date:  # don't blank a good stored date
                        target["date"] = date
                    if description.lower() != _clean_str(target.get("description")).lower():
                        target["description"] = description
                        existing_descriptions_lower.add(description.lower())
                    continue

                existing_events.append({"description": description, "date": date})  # 3) new
                existing_descriptions_lower.add(description.lower())

            # Facts: dedup by case-insensitive exact match.
            existing_facts_lower = {_clean_str(f).lower() for f in existing_facts}
            for fact in extracted["facts"]:
                if not isinstance(fact, str):
                    continue
                fact = fact.strip()
                if fact and fact.lower() not in existing_facts_lower:
                    existing_facts.append(fact)
                    existing_facts_lower.add(fact.lower())

            self._write_locked()

    # ── Onboarding ────────────────────────────────────────────────────────────

    def is_onboarding_complete(self) -> bool:
        with self._lock:
            return bool(self._state.get("onboarding_complete", False))

    def complete_onboarding(self, profile: dict) -> None:
        """Store the given profile dict, mark onboarding complete, and save."""
        with self._lock:
            self._state["profile"] = profile if isinstance(profile, dict) else {"raw_qa": str(profile)}
            self._state["onboarding_complete"] = True
            self._write_locked()
        logger.success("Onboarding complete - profile saved to memory.")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    manager = MemoryManager()
    manager.load()

    print(f"\nOnboarding complete: {manager.is_onboarding_complete()}")

    test_fact = "Prefers meetings scheduled in the afternoon, not mornings."
    print(f"\nAdding explicit memory: '{test_fact}'")
    manager.add_explicit_memory(test_fact)

    summary = manager.get_context_summary()
    print(f"\nContext summary ({len(summary)} chars):\n{summary}\n")
