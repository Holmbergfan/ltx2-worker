# LTX-2 Video + Audio Generation Worker for RunPod Serverless
# Extends the official RunPod ComfyUI worker (much smaller build)

FROM runpod/worker-comfyui:5.5.1-base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # HuggingFace cache to network volume
    HF_HOME=/runpod-volume/models/.cache/hf \
    HF_HUB_CACHE=/runpod-volume/models/.cache/hf \
    TRANSFORMERS_CACHE=/runpod-volume/models/.cache/hf \
    # Temp files to network volume
    TMPDIR=/runpod-volume/tmp \
    # Model paths
    MODEL_DIR=/runpod-volume/models \
    # Enable faster HF downloads
    HF_HUB_ENABLE_HF_TRANSFER=1

# Install LTX-Video custom nodes
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    cd ComfyUI-LTXVideo && \
    pip install --no-cache-dir -r requirements.txt || true

# Install VideoHelperSuite for video handling
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    pip install --no-cache-dir -r requirements.txt || true

# Install additional dependencies
RUN pip install --no-cache-dir \
    boto3>=1.34 \
    huggingface_hub[cli]>=0.24 \
    hf_transfer>=0.1.8 \
    requests>=2.32 \
    decord>=0.6.0 \
    librosa>=0.10.0 \
    soundfile>=0.12.0 \
    accelerate>=0.30.0

# Create model directories (will be symlinked to network volume)
RUN mkdir -p /runpod-volume/models/diffusion_models \
    /runpod-volume/models/text_encoders \
    /runpod-volume/models/vae \
    /runpod-volume/models/upscale_models \
    /runpod-volume/models/.cache/hf \
    /runpod-volume/tmp

# Copy custom handler
COPY handler.py /handler.py
COPY workflows/ /workflows/

# Override the entry point to use our handler
CMD ["python", "-u", "/handler.py"]
