# LTX-2 Video + Audio Generation Worker for RunPod Serverless
# Lightweight image - models download to network volume at runtime

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # HuggingFace cache to network volume
    HF_HOME=/runpod-volume/models/.cache/hf \
    HF_HUB_CACHE=/runpod-volume/models/.cache/hf \
    TRANSFORMERS_CACHE=/runpod-volume/models/.cache/hf \
    # Temp files to network volume
    TMPDIR=/runpod-volume/tmp \
    # Model paths
    MODEL_DIR=/runpod-volume/models \
    # ComfyUI path
    COMFY_HOME=/runpod-volume/ComfyUI \
    # Enable faster HF downloads
    HF_HUB_ENABLE_HF_TRANSFER=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy handler and workflows
COPY handler.py /app/handler.py
COPY workflows/ /app/workflows/

# Start handler
CMD ["python", "-u", "/app/handler.py"]
