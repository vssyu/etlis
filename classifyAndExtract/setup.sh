#!/usr/bin/env bash
# =============================================================================
# setup.sh — Install Python (if missing), create a venv, install dependencies,
#            and verify the GPU is usable. Run with sudo on a fresh server.
# Tested on Ubuntu 20.04 / 22.04
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_VERSION="3.11"

echo "============================================================"
echo " ClassfierAndExtractor — environment setup"
echo " Project dir : $PROJECT_DIR"
echo "============================================================"

command_exists() { command -v "$1" &>/dev/null; }

# ── 1. Install Python (if missing) ───────────────────────────────────────────
if command_exists python3; then
    echo "[python] Found: $(python3 --version)"
else
    echo "[python] Not found — installing Python ${PYTHON_VERSION}..."
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y \
        "python${PYTHON_VERSION}" \
        "python${PYTHON_VERSION}-venv" \
        "python${PYTHON_VERSION}-dev" \
        python3-pip
    # Make `python3` resolve to the freshly installed version
    update-alternatives --install /usr/bin/python3 python3 "/usr/bin/python${PYTHON_VERSION}" 1
    echo "[python] Installed: $(python3 --version)"
fi

# Ensure the venv module is available for whichever python3 we ended up with
PY_MINOR=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if ! python3 -c "import ensurepip" &>/dev/null; then
    echo "[python] Installing python${PY_MINOR}-venv..."
    apt-get install -y "python${PY_MINOR}-venv"
fi

# ── 2. Create virtual environment ────────────────────────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    echo "[venv]   Existing venv found — reusing it"
else
    echo "[venv]   Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "[pip]    Upgrading pip..."
pip install --upgrade pip --quiet

# ── 3. Install project dependencies ──────────────────────────────────────────
echo "[pip]    Installing requirements (this can take several minutes)..."
pip install -r "$PROJECT_DIR/requirements.txt"

# ── 4. Create runtime directories ────────────────────────────────────────────
echo "[dirs]   Creating data/ and output/ directories..."
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/output/qwen3b"
mkdir -p "$PROJECT_DIR/output/qwen7b"

# ── 5. Verify GPU / CUDA availability ─────────────────────────────────────────
echo ""
echo "[gpu]    Checking CUDA / GPU availability..."
python3 - <<'EOF'
import torch

cuda_ok = torch.cuda.is_available()
print(f"  CUDA available : {cuda_ok}")
if cuda_ok:
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram  = props.total_memory / 1024 ** 3
        print(f"  GPU {i}           : {props.name}  ({vram:.1f} GB VRAM)")
else:
    print("  WARNING: No GPU detected — training will fall back to CPU (very slow).")
    print("  Check that the NVIDIA driver is installed: run `nvidia-smi`.")
EOF

echo ""
echo "============================================================"
echo " Setup complete."
echo ""
echo " Activate the environment in future sessions with:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo " Next steps:"
echo "   1. Upload your Excel + annotated data to: $PROJECT_DIR/data/"
echo "   2. Train 3B:"
echo "        python train.py --model-name Qwen/Qwen2.5-3B-Instruct \\"
echo "          --label-excel data/clauses.xlsx \\"
echo "          --examples    data/annotated.xlsx \\"
echo "          --output-dir  output/qwen3b"
echo "   3. Evaluate:"
echo "        python evaluate.py --model-dir output/qwen3b \\"
echo "          --base-model-name Qwen/Qwen2.5-3B-Instruct \\"
echo "          --label-excel data/clauses.xlsx \\"
echo "          --examples    data/annotated.xlsx"
echo "   4. Repeat steps 2-3 with Qwen2.5-7B-Instruct -> output/qwen7b"
echo "============================================================"
