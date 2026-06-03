# =============================================================================
# Base: official PyTorch image with CUDA 12.1 — compatible with Nvidia L20
# =============================================================================
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        unzip \
        git \
        && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# torch is already provided by the base image; we skip reinstalling it to
# avoid version conflicts, and install the remaining packages from requirements.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        $(grep -vE '^\s*(torch|#|$)' requirements.txt | tr '\n' ' ')

# ── Project source ────────────────────────────────────────────────────────────
COPY *.py ./

# ── Runtime directories ───────────────────────────────────────────────────────
# data/   — mount your Excel + annotated examples here
# output/ — model checkpoints and results land here
# Hugging Face model weights are cached in /root/.cache/huggingface (mount it
# as a named volume so weights survive container restarts)
RUN mkdir -p data output/qwen3b output/qwen7b

# ── Smoke-test: confirm segmentation logic loads and passes unit tests ─────────
RUN python -m pytest test_segmentation.py -v --tb=short

ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface

CMD ["bash"]
