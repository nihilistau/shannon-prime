#!/usr/bin/env bash
# Shannon-Prime / Phase 2.4a WSL Ubuntu-20.04 bring-up.
#
# qai-hub-models[qwen3-4b] needs Python 3.10+ and AIMET-ONNX (Linux-only).
# This script installs everything end-to-end so qwen3_4b_v69_export.py
# can run.
#
# Run interactively (the apt step needs sudo password):
#
#     wsl -d Ubuntu-20.04
#     cd /mnt/d/F/shannon-prime-repos/shannon-prime/backends/qnn_aihub
#     bash wsl_setup.sh
#
# Idempotent: re-running skips already-installed pieces.

set -euo pipefail

echo "=== Shannon-Prime Phase 2.4a WSL setup ==="
echo "    Ubuntu version: $(lsb_release -rs)"
echo "    HOME:           $HOME"
echo

# ---- 1. Python 3.10 via deadsnakes PPA -----------------------------------

if ! command -v python3.10 &> /dev/null; then
    echo "[1/5] Installing Python 3.10 (sudo password expected)"
    sudo apt-get update -qq
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3.10-distutils
else
    echo "[1/5] Python 3.10 already installed: $(python3.10 --version)"
fi

# ---- 2. pip + venv for Python 3.10 ----------------------------------------

VENV="$HOME/.venv-sp-qnn"
if [ ! -d "$VENV" ]; then
    echo "[2/5] Creating venv at $VENV"
    python3.10 -m venv "$VENV"
else
    echo "[2/5] Venv already exists at $VENV"
fi

# Activate venv for the rest of the script
# shellcheck source=/dev/null
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

# ---- 3. qai-hub-models with qwen3-4b extras -------------------------------

if ! python -c "import qai_hub_models" 2>/dev/null; then
    echo "[3/5] Installing qai-hub-models[qwen3-4b] (this is heavy: PyTorch, transformers, etc.)"
    pip install "qai-hub-models[qwen3-4b]"
else
    echo "[3/5] qai-hub-models already installed: $(python -c 'import qai_hub_models; print(qai_hub_models.__file__)')"
fi

# ---- 4. AIMET-ONNX (the Linux-only dep that blocks Windows) ---------------

if ! python -c "import aimet_onnx" 2>/dev/null; then
    echo "[4/5] Installing aimet-onnx"
    pip install aimet-onnx || {
        echo "    aimet-onnx pip install failed. Try: pip install --extra-index-url https://download.pytorch.org/whl/cpu aimet-onnx"
        exit 1
    }
else
    echo "[4/5] aimet-onnx already installed"
fi

# ---- 5. AI Hub API token: copy from Windows side --------------------------

QAI_HUB_DIR="$HOME/.qai_hub"
WIN_QAI_HUB="/mnt/c/Users/Knack/.qai_hub/client.ini"

if [ ! -f "$QAI_HUB_DIR/client.ini" ]; then
    if [ -f "$WIN_QAI_HUB" ]; then
        echo "[5/5] Copying AI Hub API token from Windows side"
        mkdir -p "$QAI_HUB_DIR"
        cp "$WIN_QAI_HUB" "$QAI_HUB_DIR/client.ini"
    else
        echo "[5/5] No AI Hub config found at $WIN_QAI_HUB"
        echo "      Run on Windows side first: qai-hub configure --api_token <YOUR_TOKEN>"
        echo "      Then re-run this script."
        exit 1
    fi
else
    echo "[5/5] AI Hub config already at $QAI_HUB_DIR/client.ini"
fi

# ---- Sanity check ---------------------------------------------------------

echo
echo "=== Sanity check ==="
python -c "
import qai_hub as hub
import qai_hub_models
import aimet_onnx
print('  qai_hub:', hub.__name__, 'OK')
print('  qai_hub_models:', qai_hub_models.__file__)
print('  aimet_onnx:', aimet_onnx.__name__, 'OK')
print('  configured api_token:', '<set>' if open(__import__('os').path.expanduser('~/.qai_hub/client.ini')).read().strip() else '<missing>')
"

echo
echo "=== Setup done. Now run the export: ==="
echo "    source $VENV/bin/activate"
echo "    cd $(pwd)"
echo "    python3 qwen3_4b_v69_export.py \\"
echo "        --device 'Samsung Galaxy S22 Ultra 5G' \\"
echo "        --target-runtime qnn_context_binary \\"
echo "        --precision w4a16 \\"
echo "        --context-length 2048 \\"
echo "        --skip-inferencing --skip-profiling"
echo
echo "Expected: 4 PP + 4 TG context binaries, w4a16, seq=2048, V69-targeted."
echo "First run downloads ~16 GB of Qwen3-4B PyTorch weights; cached after."
echo "AI Hub compile jobs: 8+, each 10-30 min. Polling-based, can leave running."
