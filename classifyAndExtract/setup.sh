#!/usr/bin/env bash
# =============================================================================
# setup.sh  —  Install Docker + NVIDIA Container Toolkit, then build the image
# Tested on Ubuntu 20.04 / 22.04
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo " ClassfierAndExtractor — Docker environment setup"
echo " Project dir : $PROJECT_DIR"
echo "============================================================"

# ── Helper ────────────────────────────────────────────────────────────────────
command_exists() { command -v "$1" &>/dev/null; }

# ── 1. Install Docker (if missing) ───────────────────────────────────────────
if command_exists docker; then
    echo "[docker]  Already installed: $(docker --version)"
else
    echo "[docker]  Installing Docker Engine..."
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg lsb-release
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update -qq
    apt-get install -y docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin
    systemctl enable --now docker
    echo "[docker]  Installed: $(docker --version)"
fi

# ── 2. Install NVIDIA Container Toolkit (if missing) ─────────────────────────
if docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo "[nvidia]  NVIDIA Container Toolkit already configured"
else
    echo "[nvidia]  Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    echo "[nvidia]  NVIDIA Container Toolkit installed"
fi

# ── 3. Verify GPU is visible to Docker ───────────────────────────────────────
echo "[gpu]     Verifying GPU access inside Docker..."
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi \
    || { echo "ERROR: GPU not accessible inside Docker. Check driver installation."; exit 1; }

# ── 4. Create host-side data and output directories ───────────────────────────
echo "[dirs]    Creating data/ and output/ directories..."
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/output/qwen3b"
mkdir -p "$PROJECT_DIR/output/qwen7b"

# ── 5. Build Docker image ─────────────────────────────────────────────────────
echo ""
echo "[build]   Building Docker image (first build downloads PyTorch base ~5 GB)..."
cd "$PROJECT_DIR"
docker compose build

echo ""
echo "============================================================"
echo " Setup complete."
echo ""
echo " Next steps:"
echo ""
echo " 1. Upload your data files:"
echo "      scp clauses.xlsx annotated.xlsx  <user>@<host>:$PROJECT_DIR/data/"
echo ""
echo " 2. Open an interactive shell inside the container:"
echo "      cd $PROJECT_DIR && docker compose run --rm classifier bash"
echo ""
echo " 3. Train (inside the container):"
echo "      python train.py \\"
echo "        --model-name Qwen/Qwen2.5-3B-Instruct \\"
echo "        --label-excel data/clauses.xlsx \\"
echo "        --examples   data/annotated.xlsx \\"
echo "        --output-dir output/qwen3b"
echo ""
echo " 4. Evaluate:"
echo "      python evaluate.py \\"
echo "        --model-dir        output/qwen3b \\"
echo "        --base-model-name  Qwen/Qwen2.5-3B-Instruct \\"
echo "        --label-excel      data/clauses.xlsx \\"
echo "        --examples         data/annotated.xlsx"
echo "============================================================"
