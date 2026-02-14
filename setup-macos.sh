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

echo ""
echo "=== Setup complete! ==="
echo "Run: ./start-macos.sh"
