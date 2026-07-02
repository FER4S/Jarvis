# Jarvis 🎙️

A GPU-accelerated voice AI assistant powered by Claude.

**Pipeline:** Wake word → STT (faster-whisper) → Claude API → TTS (Kokoro-82M) → Speaker

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
# Edit .env and fill in your ANTHROPIC_API_KEY

# 6. Run Jarvis
python main.py
```

Note: The Kokoro TTS voice model (~300MB) downloads automatically on the first run.

## Project Structure

```
Jarvis/
├── main.py           # Entry point — starts the FastAPI server
├── config.py         # Centralised settings (loaded from .env)
├── requirements.txt  # Python dependencies
├── .env.example      # Template for required environment variables
├── .gitignore
│
├── api/              # FastAPI server & route definitions
│   ├── __init__.py
│   └── server.py
│
└── core/             # Voice pipeline modules
    ├── __init__.py
    ├── wake_word.py  # openwakeword listener
    ├── stt.py        # faster-whisper transcription
    ├── llm.py        # Claude API integration
    ├── tts.py        # Kokoro synthesis & playback
    └── assistant.py  # Pipeline orchestration & state machine
```

## Hardware

- **OS:** Windows 11  
- **GPU:** NVIDIA RTX 5060 (8 GB VRAM) — used for faster-whisper (CUDA)  
- **RAM:** 16 GB
