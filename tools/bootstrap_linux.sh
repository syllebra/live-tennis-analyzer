#!/usr/bin/env bash
# Portable setup for a full Linux machine (tested target: Omarchy / Arch Linux).
#
# It assumes the NVIDIA driver is already installed and `nvidia-smi` works.
# It does NOT install or modify system packages unless you explicitly pass
# --install-uv.  The Python environment lives in ./.venv and is disposable.
#
# Usage:
#   ./tools/bootstrap_linux.sh                 # create env + install Python deps
#   ./tools/bootstrap_linux.sh --models        # ... and download/convert models
#   ./tools/bootstrap_linux.sh --install-uv    # install uv to ~/.local/bin first
#
# After it finishes:
#   source .venv/bin/activate
#   python tools/doctor.py

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

INSTALL_MODELS=0
INSTALL_UV=0
for arg in "$@"; do
    case "$arg" in
        --models) INSTALL_MODELS=1 ;;
        --install-uv) INSTALL_UV=1 ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

echo "== Tennis analyzer Linux bootstrap =="
echo "project: $PROJECT_ROOT"
echo "system:  $(uname -srm)"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: this script is for Linux. On Windows use WSL2." >&2
    exit 1
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "WARNING: TensorRT pip wheels used here are published for x86_64." >&2
fi

if [[ -f /etc/arch-release ]]; then
    echo "detected: Arch-based Linux (Omarchy/Arch)"
else
    echo "NOTE: not an Arch-based system; package commands below are Arch-specific."
fi

# ---------------------------------------------------------------------------
# System prerequisites (report, do not auto-install)
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1 && [[ ! -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    cat >&2 <<'EOF'
ERROR: nvidia-smi was not found. Install/verify the NVIDIA driver first.

On Omarchy / Arch (choose one driver family, then reboot):
    sudo pacman -Syu
    sudo pacman -S --needed nvidia nvidia-utils linux-headers
    # or, for newer GPUs / open kernel modules:
    # sudo pacman -S --needed nvidia-open nvidia-utils linux-headers

Verify after reboot:
    nvidia-smi

The Python wheels bring their own CUDA runtime; you do not need the full
system CUDA toolkit for this project.
EOF
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is required (to clone the RT-DETR configs)." >&2
    echo "       sudo pacman -S --needed git" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    if [[ "$INSTALL_UV" == "1" ]]; then
        echo "Installing uv into ~/.local/bin ..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    else
        cat >&2 <<'EOF'
ERROR: uv is not installed.

On Omarchy / Arch:
    sudo pacman -S --needed uv

Or install it without root:
    curl -LsSf https://astral.sh/uv/install.sh | sh

Then re-run:
    ./tools/bootstrap_linux.sh --models
EOF
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
if [[ ! -x .venv/bin/python ]]; then
    echo "Creating .venv with Python 3.11 ..."
    uv python install 3.11
    uv venv --python 3.11 .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -V

echo "Installing PyTorch CUDA 12.8 build ..."
if ! python -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    uv pip install --reinstall \
        torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128
else
    echo "PyTorch CUDA already available."
fi

echo "Installing remaining requirements ..."
# Install everything except TensorRT first; its wheel-stub build is the most
# fragile part of a fresh environment, so handle it separately with a fallback.
grep -v -E '^[[:space:]]*tensorrt' requirements.txt > .requirements-no-trt.txt
uv pip install -r .requirements-no-trt.txt
rm -f .requirements-no-trt.txt

if ! python -c "import tensorrt" >/dev/null 2>&1; then
    echo "Installing TensorRT 10.8.0.43 (CUDA 12) ..."
    if ! uv pip install tensorrt==10.8.0.43; then
        echo "uv build failed; falling back to pip inside the venv ..."
        uv pip install pip
        python -m pip install --no-cache-dir tensorrt==10.8.0.43
    fi
fi

echo
echo "Python environment ready."

if [[ "$INSTALL_MODELS" == "1" ]]; then
    echo
    echo "Downloading/exporting ONNX + TensorRT models ..."
    python tools/setup_models.py --all
fi

echo
python tools/doctor.py || true

cat <<'EOF'

Next:
    source .venv/bin/activate
    python tools/doctor.py
    python main.py --stream_url videos/sample_tennis.mp4 --no-display --max-frames 60

If models are missing:
    python tools/setup_models.py --all
EOF
