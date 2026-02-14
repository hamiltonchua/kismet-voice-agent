#!/bin/bash
# Setup Kismet Voice Agent on macOS (Apple Silicon)
set -e

echo "=== Kismet Voice Agent — macOS Setup ==="

# Find conda
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
elif [ -f /opt/homebrew/anaconda3/bin/conda ]; then
    eval "$(/opt/homebrew/anaconda3/bin/conda shell.bash hook)"
elif [ -f ~/miniforge3/bin/conda ]; then
    eval "$(~/miniforge3/bin/conda shell.bash hook)"
elif [ -f ~/miniconda3/bin/conda ]; then
    eval "$(~/miniconda3/bin/conda shell.bash hook)"
else
    echo "ERROR: conda not found. Install miniforge or miniconda first."
    exit 1
fi

# Create or update conda environment
ENV_NAME="voice-agent"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda env '${ENV_NAME}' exists, activating..."
else
    echo "Creating conda env '${ENV_NAME}' with Python 3.11..."
    conda create -yn "${ENV_NAME}" python=3.11
fi

conda activate "${ENV_NAME}"

echo "Installing dependencies..."
pip install -r requirements-macos.txt

# Download openwakeword models (needed if wake word is enabled)
echo "Downloading wake word models..."
python3 -c "from openwakeword.utils import download_models; download_models()" 2>/dev/null || echo "(skipped — openwakeword not installed or download failed)"

# Generate self-signed certs if missing
if [ ! -f cert.pem ] || [ ! -f key.pem ]; then
    echo "Generating self-signed SSL certificates..."
    openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost" 2>/dev/null
fi

echo ""
echo "=== Setup complete! ==="
echo "Run: ./start-macos.sh"
echo ""
echo "Notes:"
echo "  - Wake word is OFF by default. Set WAKE_WORD_ENABLED=true to enable."
echo "  - Speaker verify is AUTO (verifies if enrolled). Enroll via the web UI."
echo "  - Models preload at startup — first launch may take 30-60s."
