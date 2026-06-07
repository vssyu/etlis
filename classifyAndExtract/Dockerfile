# =============================================================================
# Base: official PyTorch image with CUDA 12.1 — compatible with Nvidia L20
# =============================================================================
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        unzip \
        git \
        openssh-server \
        && rm -rf /var/lib/apt/lists/*

# ── SSH server configuration (for PyCharm remote debugging) ──────────────────
# Change SSH_PASSWORD at build time:
#   docker compose build --build-arg SSH_PASSWORD=yourpassword
ARG SSH_PASSWORD=classifier123
RUN echo "root:${SSH_PASSWORD}" | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config \
    && ssh-keygen -A

# ── Python dependencies ───────────────────────────────────────────────────────
# torch is already provided by the base image; we skip reinstalling it to
# avoid version conflicts, and install the remaining packages from requirements.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        $(grep -vE '^\s*(torch|#|$)' requirements.txt | tr '\n' ' ')

# ── Project source ────────────────────────────────────────────────────────────
COPY *.py ./
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

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

EXPOSE 22

ENTRYPOINT ["./entrypoint.sh"]
CMD ["bash"]
