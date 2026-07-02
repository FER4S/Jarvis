# ─────────────────────────────────────────────────────────────────────────────
#  core/llm.py – LLM layer for Jarvis
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys

# Ensure the project root is on sys.path so `import config` works when this
# file is run directly (e.g. `python core/llm.py`).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from typing import Optional

import anthropic
from loguru import logger

import config

# ── Constants ─────────────────────────────────────────────────────────────────

# Model: claude-haiku — fastest Claude model, ideal for real-time voice latency
MODEL_ID: str = "claude-haiku-4-5-20251001"

# Maximum tokens in Claude's reply.
# ~150 tokens ≈ 2–3 spoken sentences — keeps TTS snappy.
MAX_TOKENS: int = 150

# Request timeout in seconds. Voice assistants must feel responsive.
REQUEST_TIMEOUT: float = 10.0

# System prompt — shapes Claude's persona and response style
SYSTEM_PROMPT: str = (
    "You are Jarvis, a helpful voice assistant for the CEO of CodeX. "
    "Keep responses short and conversational since they will be spoken aloud. "
    "Avoid bullet points, markdown, or any formatting — plain sentences only."
)

# Fallback replies for each error category
_FALLBACK_TIMEOUT = "I'm sorry, that took too long. Please try again."
_FALLBACK_AUTH    = "I can't reach Claude right now — please check the API key."
_FALLBACK_NETWORK = "I couldn't connect. Please check your internet connection."
_FALLBACK_GENERIC = "Something went wrong. Please try again in a moment."


class LLMEngine:
    """
    Thin wrapper around the Anthropic Messages API.

    Maintains a rolling conversation history so Jarvis can handle follow-up
    questions within a session. Call reset_history() to start fresh.

    Usage:
        engine = LLMEngine()
        reply = engine.get_response("What's on my calendar today?")
        print(reply)
    """

    def __init__(
        self,
        api_key: str = config.ANTHROPIC_API_KEY,
        model: str = MODEL_ID,
        max_tokens: int = MAX_TOKENS,
        timeout: float = REQUEST_TIMEOUT,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        """
        Args:
            api_key:       Anthropic API key — loaded from .env via config.
            model:         Claude model identifier.
            max_tokens:    Max tokens in Claude's reply.
            timeout:       HTTP request timeout in seconds.
            system_prompt: Persona / instruction injected as the system turn.
        """
        if not api_key:
            logger.warning(
                "ANTHROPIC_API_KEY is not set. "
                "Claude calls will fail until it is added to .env."
            )

        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt

        # Conversation history: list of {"role": "user"|"assistant", "content": str}
        # Kept in memory for the duration of the session.
        self._history: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def get_response(self, user_text: str) -> str:
        """
        Send the user's transcribed speech to Claude and return its reply.

        The conversation history is maintained automatically, so follow-up
        questions work naturally within a session.

        Args:
            user_text: Transcribed text from the STT engine.

        Returns:
            Claude's plain-text reply, or a fallback string on error.
        """
        if not user_text.strip():
            logger.warning("get_response() called with empty text — skipping API call.")
            return _FALLBACK_GENERIC

        # Append the user turn to history
        self._history.append({"role": "user", "content": user_text})

        logger.info(f"Sending to Claude: '{user_text}'")

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system_prompt,
                messages=self._history,
            )

            # Extract plain text from the first content block
            reply: str = response.content[0].text.strip()

            # Append the assistant turn so follow-ups have context
            self._history.append({"role": "assistant", "content": reply})

            logger.success(f"Claude replied: '{reply}'")
            return reply

        except anthropic.APITimeoutError:
            logger.error("Claude request timed out.")
            self._history.pop()  # remove the unanswered user turn
            return _FALLBACK_TIMEOUT

        except anthropic.AuthenticationError:
            logger.error("Anthropic authentication failed — check ANTHROPIC_API_KEY in .env.")
            self._history.pop()
            return _FALLBACK_AUTH

        except anthropic.APIConnectionError:
            logger.error("No connection to Anthropic API — check internet.")
            self._history.pop()
            return _FALLBACK_NETWORK

        except anthropic.APIStatusError as exc:
            logger.error(f"Anthropic API error {exc.status_code}: {exc.message}")
            self._history.pop()
            return _FALLBACK_GENERIC

        except Exception as exc:
            logger.exception(f"Unexpected error calling Claude: {exc}")
            self._history.pop()
            return _FALLBACK_GENERIC

    def reset_history(self) -> None:
        """Clear the conversation history (e.g. on a new wake-word trigger session)."""
        self._history.clear()
        logger.debug("Conversation history cleared.")

    @property
    def history(self) -> list[dict]:
        """Read-only view of the current conversation history."""
        return list(self._history)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pretty console logging for standalone testing
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    engine = LLMEngine()

    # --- Single-turn test ---
    test_message = "What time is it in Beirut right now?"
    print(f"\nUser: {test_message}")
    reply = engine.get_response(test_message)
    print(f"Jarvis: {reply}\n")

    # --- Follow-up test (verifies history is maintained) ---
    followup = "And what about in Tokyo?"
    print(f"User: {followup}")
    reply2 = engine.get_response(followup)
    print(f"Jarvis: {reply2}\n")
