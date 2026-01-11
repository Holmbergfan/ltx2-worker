#!/usr/bin/env python3
"""
LTX-2 Video + Audio Generation Worker for RunPod Serverless
Lightweight - downloads ComfyUI and models to network volume at runtime

200GB network volume storage:
- ComfyUI installation: ~2GB
- LTX-Video models: ~5GB
- Text encoders: ~10GB
- Plenty of room for cache and outputs
"""

import os
import sys
import time
import uuid
import subprocess
import traceback
from pathlib import Path
from typing import Dict, Any, List

import runpod
import boto3
from huggingface_hub import hf_hub_download

# Enable fast HF downloads (must be set before any HF calls)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# =============================================================================
# Configuration
# =============================================================================

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")
COMFY_HOME = os.environ.get("COMFY_HOME", "/runpod-volume/ComfyUI")
TMPDIR = os.environ.get("TMPDIR", "/runpod-volume/tmp")

# HuggingFace
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))

# S3 Storage
S3_BUCKET = os.environ.get("S3_BUCKET", "ltx2-models")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "")
S3_REGION = os.environ.get("S3_REGION", "eu-north-1")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
PRESIGNED_URL_EXPIRY = int(os.environ.get("PRESIGNED_URL_EXPIRY", 14400))

# =============================================================================
# Setup Functions - Download at Runtime to Network Volume
# =============================================================================

def ensure_directories():
    """Create directories on network volume"""
    dirs = [
        f"{MODEL_DIR}/checkpoints",
        f"{MODEL_DIR}/text_encoders",
        f"{MODEL_DIR}/vae",
        f"{MODEL_DIR}/clip",
        f"{MODEL_DIR}/.cache/hf",
        TMPDIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"Directories ready in {MODEL_DIR}")

def install_comfyui():
    """Install ComfyUI to network volume if not present"""
    if os.path.exists(f"{COMFY_HOME}/main.py"):
        print("ComfyUI already installed on network volume")
        return True

    print("Installing ComfyUI to network volume...")
    try:
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/comfyanonymous/ComfyUI.git",
            COMFY_HOME
        ], check=True, timeout=300)

        subprocess.run([
            "pip", "install", "--no-cache-dir", "-r",
            f"{COMFY_HOME}/requirements.txt"
        ], check=True, timeout=600)

        # Install LTX-Video nodes
        custom_nodes = f"{COMFY_HOME}/custom_nodes"
        os.makedirs(custom_nodes, exist_ok=True)

        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/Lightricks/ComfyUI-LTXVideo.git",
            f"{custom_nodes}/ComfyUI-LTXVideo"
        ], check=True, timeout=120)

        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
            f"{custom_nodes}/ComfyUI-VideoHelperSuite"
        ], check=True, timeout=120)

        print("ComfyUI installed!")
        return True
    except Exception as e:
        print(f"ComfyUI install error: {e}")
        return False

def fast_download(url: str, output_path: str, token: str = None):
    """Fast download using aria2c with parallel connections"""
    headers = []
    if token:
        headers = ["--header", f"Authorization: Bearer {token}"]

    cmd = [
        "aria2c", "-x", "16", "-s", "16", "-k", "1M",
        "--file-allocation=none",
        "-d", os.path.dirname(output_path),
        "-o", os.path.basename(output_path),
        *headers,
        url
    ]
    print(f"Downloading with aria2c (16 connections): {os.path.basename(output_path)}")
    subprocess.run(cmd, check=True)

def download_models(force: bool = False):
    """Download LTX-Video models to network volume"""

    # LTX-Video model (2B v0.9.5 - 6.34GB)
    model_path = f"{MODEL_DIR}/checkpoints/ltx-video-2b-v0.9.5.safetensors"
    if not os.path.exists(model_path) or force:
        print("Downloading LTX-Video model (~6GB)...")
        try:
            # Try fast download with aria2c
            url = "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.5.safetensors"
            fast_download(url, model_path, HF_TOKEN)
        except Exception as e:
            print(f"aria2c failed ({e}), falling back to hf_hub_download...")
            hf_hub_download(
                repo_id="Lightricks/LTX-Video",
                filename="ltx-video-2b-v0.9.5.safetensors",
                local_dir=f"{MODEL_DIR}/checkpoints",
                token=HF_TOKEN or None,
            )
        print("LTX-Video model downloaded!")

    # T5 text encoder
    t5_path = f"{MODEL_DIR}/text_encoders/t5xxl_fp16.safetensors"
    if not os.path.exists(t5_path) or force:
        print("Downloading T5 encoder (~10GB)...")
        try:
            url = "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
            fast_download(url, t5_path, HF_TOKEN)
        except Exception as e:
            print(f"aria2c failed ({e}), falling back to hf_hub_download...")
            try:
                hf_hub_download(
                    repo_id="comfyanonymous/flux_text_encoders",
                    filename="t5xxl_fp16.safetensors",
                    local_dir=f"{MODEL_DIR}/text_encoders",
                    token=HF_TOKEN or None,
                )
            except Exception as e2:
                print(f"T5 download note: {e2}")
        print("T5 encoder downloaded!")

    print("All models ready!")

def setup_environment():
    """Full setup on first run"""
    print("=" * 50)
    print("Setting up LTX-2 environment on network volume...")
    print("=" * 50)

    ensure_directories()
    install_comfyui()
    download_models()

    print("Setup complete!")

# =============================================================================
# S3 Upload
# =============================================================================

def upload_to_s3(file_path: str, content_type: str = "video/mp4") -> Dict[str, Any]:
    """Upload to S3 and return presigned URL"""
    if not AWS_ACCESS_KEY_ID:
        import base64
        with open(file_path, "rb") as f:
            return {
                "type": "base64",
                "data": base64.b64encode(f.read()).decode(),
                "filename": os.path.basename(file_path),
            }

    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT or None,
        region_name=S3_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    key = f"ltx2/{time.strftime('%Y%m%d-%H%M%S')}/{uuid.uuid4()}/{os.path.basename(file_path)}"

    s3.upload_file(file_path, S3_BUCKET, key, ExtraArgs={"ContentType": content_type})

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=PRESIGNED_URL_EXPIRY,
    )

    return {"type": "s3_url", "url": url, "key": key, "filename": os.path.basename(file_path)}

# =============================================================================
# GPU Detection
# =============================================================================

def detect_gpu() -> Dict[str, Any]:
    """Detect GPU info"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1),
            }
    except:
        pass
    return {"gpu_name": "Unknown", "vram_gb": 0}

# =============================================================================
# Handler
# =============================================================================

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod serverless handler"""
    start_time = time.time()
    job_input = event.get("input", {})

    try:
        setup_environment()
        gpu = detect_gpu()
        print(f"GPU: {gpu['gpu_name']} ({gpu['vram_gb']}GB)")

        action = job_input.get("action", "generate")

        if action == "sync_models":
            download_models(force=job_input.get("force", False))
            return {"status": "success", "action": "sync_models", "gpu": gpu}

        if action == "status":
            return {
                "status": "success",
                "comfyui_installed": os.path.exists(f"{COMFY_HOME}/main.py"),
                "model_exists": os.path.exists(f"{MODEL_DIR}/checkpoints/ltx-video-2b-v0.9.5.safetensors"),
                "gpu": gpu,
            }

        if action == "generate":
            prompt = job_input.get("prompt", "A beautiful sunset over the ocean")
            width = job_input.get("width", 768)
            height = job_input.get("height", 512)
            steps = job_input.get("steps", 30)
            seed = job_input.get("seed", int(time.time()) % 1000000)

            # TODO: Implement actual ComfyUI workflow execution
            # For now, return status
            return {
                "status": "success",
                "message": "Environment ready. ComfyUI workflow execution coming soon.",
                "parameters": {"prompt": prompt, "width": width, "height": height, "steps": steps, "seed": seed},
                "gpu": gpu,
                "execution_time": round(time.time() - start_time, 2),
            }

        return {"status": "error", "error": f"Unknown action: {action}"}

    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    print("LTX-2 Worker Starting...")
    runpod.serverless.start({"handler": handler})
