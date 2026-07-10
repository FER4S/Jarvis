# Jarvis 🎙️

A GPU-accelerated, fully local voice AI assistant for Windows, powered by Claude.

**Pipeline:** Wake word → STT (faster-whisper) → Claude API → TTS (Kokoro-82M) → Speaker

Everything runs on-device except the Claude calls themselves. A FastAPI server wraps the pipeline so a
frontend (e.g. an Electron app) can start/stop it, manage settings, and receive live state over a
WebSocket — see **[API.md](API.md)** for the full contract.

## Features

- **Voice conversation** — say "Hey Jarvis", then keep talking; follow-up turns need no wake word.
- **Persistent memory** — a one-time spoken onboarding, explicit `"remember that…"` commands, and
  background extraction of people/events/facts after each conversation.
- **Email** — connect multiple accounts (Hostinger-style IMAP + Gmail via OAuth), get proactive spoken
  new-mail announcements, and read, search, filter, and summarize mail by voice. Credentials are
  encrypted at rest with Windows DPAPI and never leave the machine.
- **Send and reply by voice** — "email Michael, tell him I want to meet at 5pm", or "reply to it and say
  I'll review it tonight". Jarvis drafts a real email from what you said, reads it back in full, and sends
  **only** on an explicit spoken "yes" — anything ambiguous asks again rather than guessing. Recipients
  resolve from memory; an unknown address is spelled back for confirmation, or can be typed into the
  dashboard. *CC/BCC, attachments, multiple recipients, and reply threading are not implemented yet.*

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/FER4S/Jarvis.git
cd Jarvis

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install PyTorch with CUDA support FIRST (~3GB download)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Configure secrets
copy .env.example .env
# Edit .env — see "Required settings" below

# 6. Run Jarvis
python main.py
```

Note: The Kokoro TTS voice model (~300MB) downloads automatically on the first run.

### Required settings (`.env`)

| Variable | Why |
|---|---|
| `ANTHROPIC_API_KEY` | The Claude API key. Nothing works without it. |
| `JARVIS_API_TOKEN` | Shared secret the frontend sends on `/start`, `/stop`, `/events`, and `/email/*`. **Not optional** — with it unset those endpoints reject every request. Generate one with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. |

Everything else has a working default. See `.env.example` for the full list.

### Optional: connecting Gmail

IMAP accounts (Hostinger and friends) need no extra setup. Gmail additionally requires a Google Cloud
OAuth client:

1. In [Google Cloud Console](https://console.cloud.google.com), create a project and **enable the Gmail
   API** (APIs & Services → Enable APIs → Gmail API). Skipping this is the single most common failure —
   consent succeeds and then the callback fails.
2. Create an OAuth 2.0 **Web application** client, and register this redirect URI:
   `http://localhost:8000/email/accounts/gmail/oauth-callback`
3. Put the client id/secret into `.env` as `GOOGLE_OAUTH_CLIENT_ID` / `GOOGLE_OAUTH_CLIENT_SECRET`.

Then connect an account through the frontend's settings UI (or the `/email/accounts/gmail/oauth-url`
endpoint). A newly connected account is live immediately — no restart.

## Project Structure

```
Jarvis/
├── main.py           # Entry point — starts the voice pipeline + FastAPI server
├── config.py         # Centralised settings (loaded from .env)
├── requirements.txt  # Python dependencies
├── .env.example      # Template for required environment variables
├── API.md            # REST + WebSocket contract (read this if you're on the frontend)
├── CLAUDE.md         # Architecture & conventions (read this if you're changing the backend)
│
├── api/              # FastAPI server & route definitions
│   ├── __init__.py
│   └── server.py
│
├── core/             # Voice pipeline modules
│   ├── __init__.py
│   ├── wake_word.py       # openwakeword listener
│   ├── stt.py             # faster-whisper transcription
│   ├── llm.py             # Claude API integration
│   ├── tts.py             # Kokoro synthesis & playback
│   ├── memory.py          # Persistent memory: onboarding, extraction, recall
│   ├── email_accounts.py  # DPAPI-encrypted account store
│   ├── email_fetch.py     # Unified IMAP + Gmail fetch/search layer
│   ├── email_manager.py   # Polling, announcements, intent routing, reasoning
│   └── assistant.py       # Pipeline orchestration & state machine
│
└── data/             # Generated at runtime, git-ignored — personal data, never committed
    ├── memory.json
    ├── email_accounts.dat   # DPAPI-encrypted; only decryptable by this Windows user
    └── email_cache.json
```

Each `core/*.py` module has a `__main__` block that runs it standalone, which is the way to exercise one
stage in isolation:

```bash
python -m core.wake_word   # mic loop, prints on wake word detection
python -m core.stt         # records one utterance and prints the transcription
python -m core.llm         # sends test messages to Claude and prints replies
python -m core.tts         # synthesizes and plays a couple of hardcoded lines
python -m core.memory      # loads data/memory.json, adds a test fact, prints the context summary
python -m core.email_manager   # runs one live poll of the connected accounts, prints the dashboard summary
```

## Hardware

- **OS:** Windows 11
- **GPU:** NVIDIA RTX 5060 (8 GB VRAM) — used for faster-whisper (CUDA)
- **RAM:** 16 GB

Both STT and TTS fall back to CPU automatically if the CUDA DLLs fail to load, so a GPU is strongly
recommended but not strictly required.
