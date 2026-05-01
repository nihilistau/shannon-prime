#!/usr/bin/env bash
# Shannon-Prime / Phase 2.4a WSL Ubuntu-22.04 bring-up.
#
# qai-hub-models[qwen3_4b] needs Python 3.10 AND aimet-onnx 2.26.0+cpu
# from Qualcomm's GitHub releases. The PyPI `aimet-onnx` is a yanked stub
# that doesn't actually work.
#
# Verified path 2026-05-01:
#   * Ubuntu 20.04 (glibc 2.31) DOES NOT WORK — the aimet wheel is
#     manylinux_2_34, requires glibc >= 2.34. Use Ubuntu 22.04 (glibc 2.35).
#   * To bootstrap a fresh Ubuntu 22.04 WSL distro from a rootfs tarball:
#       curl -L -o ubuntu22.tar.gz \
#         https://cloud-images.ubuntu.com/wsl/jammy/current/ubuntu-jammy-wsl-amd64-ubuntu22.04lts.rootfs.tar.gz
#       wsl --import Ubuntu-22.04-sp C:\WSL\Ubuntu-22.04-sp ubuntu22.tar.gz --version 2
#     Then create a non-root user (the import comes up as root):
#       wsl -d Ubuntu-22.04-sp -- bash -c "useradd -m -s /bin/bash claude && \\
#         echo 'claude:Claude1234' | chpasswd && usermod -aG sudo claude && \\
#         echo 'claude ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/claude && \\
#         printf '[user]\ndefault=claude\n[boot]\nsystemd=false\n' > /etc/wsl.conf"
#       wsl --terminate Ubuntu-22.04-sp
#
# Then run this script:
#       wsl -d Ubuntu-22.04-sp
#       cd /mnt/d/F/shannon-prime-repos/shannon-prime/backends/qnn_aihub
#       bash wsl_setup.sh
#
# Idempotent: re-running skips already-installed pieces.

set -euo pipefail

echo "=== Shannon-Prime Phase 2.4a WSL setup (Ubuntu 22.04) ==="
echo "    Ubuntu version: $(lsb_release -rs 2>/dev/null || echo unknown)"
echo "    glibc:          $(ldd --version | head -1)"
echo "    HOME:           $HOME"
echo

# Hard guard — ubuntu 20.04 glibc 2.31 cannot load aimet-onnx 2.26.0
# (manylinux_2_34 = glibc 2.34+).
GLIBC_MAJOR=$(ldd --version | head -1 | awk '{print $NF}' | cut -d. -f1)
GLIBC_MINOR=$(ldd --version | head -1 | awk '{print $NF}' | cut -d. -f2)
if [ "$GLIBC_MAJOR" -lt 2 ] || { [ "$GLIBC_MAJOR" -eq 2 ] && [ "$GLIBC_MINOR" -lt 34 ]; }; then
    echo "ERROR: glibc < 2.34 detected. aimet-onnx 2.26.0 manylinux_2_34"
    echo "       wheel will fail to load. Use Ubuntu 22.04 instead."
    exit 1
fi

# ---- 1. Python 3.10 (system on 22.04, no PPA needed) ---------------------

if ! command -v python3.10 &> /dev/null; then
    echo "[1/5] Installing Python 3.10 (system package, sudo password expected)"
    sudo apt-get update -qq
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev \
        python3-pip git curl build-essential
else
    echo "[1/5] Python 3.10 already installed: $(python3.10 --version)"
fi

# ---- 2. Venv at $HOME/.venv-sp-qnn ----------------------------------------

VENV="$HOME/.venv-sp-qnn"
if [ ! -d "$VENV" ]; then
    echo "[2/5] Creating venv at $VENV"
    python3.10 -m venv "$VENV"
else
    echo "[2/5] Venv already exists at $VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# ---- 3. Torch 2.4.1 +cpu (pinned by qai-hub-models[qwen3_4b]) -------------
#
# qai-hub-models[qwen3_4b] requires torch==2.4.1, torchvision==0.19.1.
# Default PyPI torch wheels pull in ~2.5 GB of CUDA libs even on a CPU-only
# host; the +cpu wheels from PyTorch's CPU index are ~200 MB total.

if ! python -c "import torch; assert torch.__version__.startswith('2.4.1')" 2>/dev/null; then
    echo "[3/5] Installing torch 2.4.1+cpu (CPU-only, no CUDA bloat)"
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1+cpu \
        torchvision==0.19.1+cpu
else
    echo "[3/5] torch 2.4.1+cpu already installed"
fi

# ---- 4. aimet-onnx + qai-hub-models[qwen3_4b] -----------------------------
#
# aimet-onnx 2.26.0+cpu lives only on quic/aimet GitHub releases. The
# `aimet-onnx` PyPI package is a yanked stub. The +cpu wheel is 1 MB
# (CPU stubs only); +cu121 is ~80 MB (with CUDA libs).

if ! python -c "import aimet_onnx" 2>/dev/null; then
    echo "[4/5] Installing aimet-onnx 2.26.0+cpu + aimet-torch + qai-hub-models[qwen3_4b]"
    pip install --no-cache-dir \
        https://github.com/quic/aimet/releases/download/2.26.0/aimet_onnx-2.26.0%2Bcpu-cp310-cp310-manylinux_2_34_x86_64.whl \
        https://github.com/quic/aimet/releases/download/2.26.0/aimet_torch-2.26.0%2Bcpu-py310-none-any.whl \
        onnxruntime==1.22 \
        'qai-hub-models[qwen3_4b]'
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
        echo "[5/5] No AI Hub config at $WIN_QAI_HUB"
        echo "      Run on Windows side first: qai-hub configure --api_token <YOUR_TOKEN>"
        exit 1
    fi
else
    echo "[5/5] AI Hub config already at $QAI_HUB_DIR/client.ini"
fi

# ---- Sanity check ---------------------------------------------------------

echo
echo "=== Sanity check ==="
python -c "
import sys
print('  python    :', sys.version.split()[0])
import torch
print('  torch     :', torch.__version__)
import onnxruntime
print('  onnxruntime:', onnxruntime.__version__)
import aimet_onnx
print('  aimet_onnx: OK')
import qai_hub as hub
print('  qai_hub   : OK')
import qai_hub_models.models.qwen3_4b as q
print('  qwen3_4b  :', q.__file__)
"

echo
echo "=== Setup done. Now run the export: ==="
echo "    source $VENV/bin/activate"
echo "    cd $(pwd)"
echo "    HF_HOME=\$HOME/.cache/huggingface python qwen3_4b_v69_export.py \\"
echo "        --device 'Samsung Galaxy S22 Ultra 5G' \\"
echo "        --target-runtime qnn_context_binary \\"
echo "        --context-length 2048 \\"
echo "        --skip-inferencing --skip-profiling"
echo
echo "    NOTE: do NOT pass --precision; qai-hub-models has a bug where"
echo "    --precision overrides the parsed Precision enum with a raw string,"
echo "    failing the supported_precision_runtimes validation. The default"
echo "    DEFAULT_PRECISION (w4a16 from qwen3_4b/model.py) is what we want."
echo
echo "Expected: 4 PP + 4 TG context binaries, w4a16, seq=2048, V69-targeted."
echo "First run downloads ~8 GB of Qwen3-4B PyTorch weights to HF_HOME."
echo "AI Hub compile jobs: 8+, each 10-30 min. Polling-based, can leave running."
