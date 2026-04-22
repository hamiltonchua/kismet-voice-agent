#!/bin/bash
# Start Kismet Voice Agent with Orpheus TTS (default) — macOS Apple Silicon
# Fallbacks: uncomment Kokoro or Soprano lines below.

# Load .env if present
SCRIPT_DIR_EARLY="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR_EARLY/.env" ]; then
    set -a
    source "$SCRIPT_DIR_EARLY/.env"
    set +a
fi

# Activate conda environment (use full path so this works without conda in PATH)
CONDA_BASE=/opt/homebrew/anaconda3
eval "$($CONDA_BASE/bin/conda shell.bash hook)"
conda activate voice-agent

# Isolate from ~/.local site-packages (numpy conflicts)
export PYTHONNOUSERSITE=1

# --- TTS Model Selection ---
# Default: Orpheus (best quality, female voice)
export MLX_TTS_MODEL=mlx-community/orpheus-3b-0.1-ft-4bit
export ORPHEUS_VOICE=tara

# Fallback: Soprano (fast, single voice)
# export MLX_TTS_MODEL=mlx-community/Soprano-1.1-80M-bf16
# export SOPRANO_REF="$SCRIPT_DIR_EARLY/voices/rosamund_pike.wav"

# Fallback: Kokoro (fastest, lightest weight, 50+ voices)
# export MLX_TTS_MODEL=mlx-community/Kokoro-82M-bf16
# export MLX_TTS_VOICE=af_kore
# export KOKORO_VOICE=af_kore

# STT model (Parakeet — 4x faster than Whisper)
export MLX_STT_MODEL=${MLX_STT_MODEL:-mlx-community/parakeet-tdt-0.6b-v3}

# Feature flags (set to "false" to disable)
export WAKE_WORD_ENABLED=${WAKE_WORD_ENABLED:-true}
export SPEAKER_VERIFY=${SPEAKER_VERIFY:-true}

# Smart Turn endpoint detection (turn-taking prediction)
export SMART_TURN_ENABLED=${SMART_TURN_ENABLED:-true}
export SMART_TURN_THRESHOLD=${SMART_TURN_THRESHOLD:-0.5}
export SMART_TURN_MAX_WAIT_SEC=${SMART_TURN_MAX_WAIT_SEC:-3.0}

# LLM API key (optional — most local servers don't need auth)
# export LLM_API_KEY="${LLM_API_KEY:-}"

# Unbuffered output for logging
export PYTHONUNBUFFERED=1

# Run the server
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
exec python3 server.py "$@" > /tmp/voice-agent.log 2>&1
