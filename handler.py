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
import random
import subprocess
import traceback
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import runpod
import boto3
import requests
from huggingface_hub import hf_hub_download, snapshot_download

# Enable fast HF downloads (must be set before any HF calls)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# =============================================================================
# Configuration
# =============================================================================

def resolve_default_root() -> str:
    # Prioritize /runpod-volume (network volume) over /workspace (ephemeral container storage)
    if os.path.ismount("/runpod-volume"):
        return "/runpod-volume"
    if os.path.ismount("/workspace"):
        return "/workspace"
    if os.path.exists("/runpod-volume"):
        return "/runpod-volume"
    if os.path.exists("/workspace"):
        return "/workspace"
    return "/runpod-volume"


DEFAULT_ROOT = resolve_default_root()
MODEL_DIR = os.environ.get("MODEL_DIR", f"{DEFAULT_ROOT}/models")
COMFY_HOME = os.environ.get("COMFY_HOME", f"{DEFAULT_ROOT}/ComfyUI")
TMPDIR = os.environ.get("TMPDIR", f"{DEFAULT_ROOT}/tmp")
AUTO_SEED = os.getenv("AUTO_SEED", "true").lower() == "true"

# Only fall back to /workspace if /runpod-volume is NOT available at all
if not os.path.ismount("/runpod-volume") and not os.path.exists("/runpod-volume"):
    if os.path.ismount("/workspace") or os.path.exists("/workspace"):
        print("Warning: /runpod-volume not available, falling back to /workspace.")
        MODEL_DIR = "/workspace/models"
        COMFY_HOME = "/workspace/ComfyUI"
        TMPDIR = "/workspace/tmp"

# Create directories on the network volume if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(COMFY_HOME, exist_ok=True)
os.makedirs(TMPDIR, exist_ok=True)

def log_disk_usage(path: str):
    try:
        total, used, free = shutil.disk_usage(path)
        print(
            "Disk usage for %s: total=%.1fGB used=%.1fGB free=%.1fGB"
            % (path, total / (1024**3), used / (1024**3), free / (1024**3))
        )
    except FileNotFoundError:
        print(f"Disk usage for {path}: path not found")


def _sanitize_log_value(value, max_len: int = 800):
    if isinstance(value, str):
        if value.startswith("data:image"):
            return f"[data:image;base64 length={len(value)}]"
        if value.startswith("data:video"):
            return f"[data:video;base64 length={len(value)}]"
        if len(value) > max_len:
            return value[:max_len] + f"... ({len(value)} chars)"
        return value
    if isinstance(value, list):
        return [_sanitize_log_value(item, max_len=max_len) for item in value]
    if isinstance(value, dict):
        return {k: _sanitize_log_value(v, max_len=max_len) for k, v in value.items()}
    return value


def _log_ltx2_event(label: str, payload: Any) -> None:
    try:
        sanitized = _sanitize_log_value(payload)
        print(f"[LTX2] {label}: {json.dumps(sanitized, ensure_ascii=True)}")
    except Exception as exc:
        print(f"[LTX2] {label}: <log failed: {exc}>")


def _summarize_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"node_count": len(workflow)}
    clip_texts: List[str] = []
    ltxv_prompts: List[str] = []
    ltxv_negative: List[str] = []
    input_images: List[str] = []
    sizes: List[Dict[str, Any]] = []
    lengths: List[int] = []
    fps_values: List[float] = []
    noise_seeds: List[int] = []
    sampler_params: List[Dict[str, Any]] = []
    cfg_values: List[float] = []
    image_strengths: List[float] = []

    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        inputs = node.get("inputs", {})

        if class_type == "CLIPTextEncode":
            text = inputs.get("text")
            if text:
                clip_texts.append(text)

        if class_type == "LTXVTextEncode":
            prompt = inputs.get("prompt")
            if prompt:
                ltxv_prompts.append(prompt)
            negative_prompt = inputs.get("negative_prompt")
            if negative_prompt:
                ltxv_negative.append(negative_prompt)

        if class_type == "LoadImage":
            image = inputs.get("image")
            if image:
                input_images.append(image)

        if class_type == "EmptyImage":
            width = inputs.get("width")
            height = inputs.get("height")
            if width is not None or height is not None:
                sizes.append({"width": width, "height": height})

        if class_type == "EmptyLTXVLatentVideo":
            length = inputs.get("length")
            if length is not None:
                lengths.append(int(length))

        if class_type in {"LTXVConditioning", "CreateVideo"}:
            value = inputs.get("frame_rate") if class_type == "LTXVConditioning" else inputs.get("fps")
            if value is not None:
                fps_values.append(float(value))

        if class_type == "RandomNoise":
            seed = inputs.get("noise_seed")
            if seed is not None:
                noise_seeds.append(int(seed))

        if class_type == "LTXVSampler":
            sampler_entry = {}
            if inputs.get("seed") is not None:
                sampler_entry["seed"] = int(inputs.get("seed"))
            if inputs.get("steps") is not None:
                sampler_entry["steps"] = int(inputs.get("steps"))
            if inputs.get("cfg") is not None:
                sampler_entry["cfg"] = float(inputs.get("cfg"))
            if sampler_entry:
                sampler_params.append(sampler_entry)

        if class_type == "CFGGuider":
            cfg = inputs.get("cfg")
            if cfg is not None:
                cfg_values.append(float(cfg))

        if class_type == "LTXVImgToVideoInplace":
            strength = inputs.get("strength")
            if strength is not None:
                image_strengths.append(float(strength))

    if clip_texts:
        summary["clip_texts"] = clip_texts
    if ltxv_prompts:
        summary["ltxv_prompts"] = ltxv_prompts
    if ltxv_negative:
        summary["ltxv_negative_prompts"] = ltxv_negative
    if input_images:
        summary["input_images"] = input_images
    if sizes:
        summary["sizes"] = sizes
    if lengths:
        summary["lengths"] = lengths
    if fps_values:
        summary["fps"] = fps_values
    if noise_seeds:
        summary["noise_seeds"] = noise_seeds
    if sampler_params:
        summary["sampler"] = sampler_params
    if cfg_values:
        summary["cfg"] = cfg_values
    if image_strengths:
        summary["image_strengths"] = image_strengths

    return summary
print(f"Using storage root: {DEFAULT_ROOT}")
print(f"MODEL_DIR={MODEL_DIR}")
print(f"COMFY_HOME={COMFY_HOME}")
print(f"TMPDIR={TMPDIR}")
print(f"AUTO_SEED={AUTO_SEED}")
log_disk_usage("/workspace")
log_disk_usage("/runpod-volume")
COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"

# HuggingFace
HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACE_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    or ""
)

# LTX-2 (19B) Model - The newest and best model
# Options: ltx-2-19b-distilled-fp8.safetensors (smaller), ltx-2-19b-dev-fp8.safetensors (larger)
LTX2_MODEL_REPO = os.environ.get("LTX2_MODEL_REPO", "Lightricks/LTX-2")
LTX2_MODEL_FILENAME = os.environ.get("LTX2_MODEL_FILENAME", "ltx-2-19b-distilled-fp8.safetensors")
LTX2_FULL_MODEL_FILENAME = os.environ.get("LTX2_FULL_MODEL_FILENAME", "ltx-2-19b-dev-fp8.safetensors")
GEMMA_REPO = os.environ.get("GEMMA_REPO", "google/gemma-3-12b-it-qat-q4_0-unquantized")
GEMMA_DIRNAME = os.environ.get("GEMMA_DIRNAME", "gemma-3-12b-it-qat-q4_0-unquantized")
PRELOAD_MODELS = os.environ.get("PRELOAD_MODELS", "").lower() in ("1", "true", "yes")

# S3 Storage
S3_BUCKET = os.environ.get("S3_BUCKET", "ltx2-models")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "")
S3_REGION = os.environ.get("S3_REGION", "eu-north-1")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
PRESIGNED_URL_EXPIRY = int(os.environ.get("PRESIGNED_URL_EXPIRY", 14400))

COMFY_PROCESS = None
COMFY_RESTART_REQUIRED = False

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
        f"{MODEL_DIR}/loras",
        f"{MODEL_DIR}/latent_upscale_models",
        f"{MODEL_DIR}/.cache/hf",
        TMPDIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"Directories ready in {MODEL_DIR}")

def ensure_comfy_model_links():
    """Expose MODEL_DIR to ComfyUI models directory"""
    global COMFY_RESTART_REQUIRED
    comfy_models = f"{COMFY_HOME}/models"
    if os.path.islink(comfy_models):
        target = os.readlink(comfy_models)
        if os.path.abspath(target) == os.path.abspath(MODEL_DIR):
            return
        try:
            os.unlink(comfy_models)
            COMFY_RESTART_REQUIRED = True
        except OSError as e:
            print(f"Failed to remove models symlink ({e}); leaving as-is.")
            return

    if not os.path.exists(comfy_models):
        os.makedirs(os.path.dirname(comfy_models), exist_ok=True)
        try:
            os.symlink(MODEL_DIR, comfy_models)
            print(f"Linked ComfyUI models -> {MODEL_DIR}")
            return
        except OSError as e:
            print(f"Symlink to models failed ({e}), falling back to subdir links...")

    # If models dir exists, link subfolders to keep ComfyUI happy
    os.makedirs(comfy_models, exist_ok=True)
    for subdir in ["checkpoints", "text_encoders", "vae", "clip", "loras", "latent_upscale_models"]:
        src = os.path.join(MODEL_DIR, subdir)
        dst = os.path.join(comfy_models, subdir)
        if os.path.exists(dst) or os.path.islink(dst):
            continue
        try:
            os.symlink(src, dst)
        except OSError as e:
            print(f"Symlink for {subdir} failed: {e}")

    # Ensure ComfyUI picks up MODEL_DIR even if models/ exists without symlinks.
    extra_paths_path = Path(COMFY_HOME) / "extra_model_paths.yaml"
    extra_block = f"""ltx2:
  base_path: {MODEL_DIR}
  checkpoints: checkpoints
  text_encoders: |
    text_encoders
    clip
  loras: loras
  latent_upscale_models: latent_upscale_models
  vae: vae
"""
    if extra_paths_path.exists():
        existing = extra_paths_path.read_text(encoding="utf-8")
        if MODEL_DIR not in existing:
            extra_paths_path.write_text(existing.rstrip() + "\n\n" + extra_block, encoding="utf-8")
            COMFY_RESTART_REQUIRED = True
    else:
        extra_paths_path.write_text(extra_block, encoding="utf-8")
        COMFY_RESTART_REQUIRED = True

def install_comfyui_deps():
    """Ensure ComfyUI dependencies are installed"""
    req_file = f"{COMFY_HOME}/requirements.txt"
    if os.path.exists(req_file):
        print("Installing/updating ComfyUI dependencies...")
        subprocess.run([
            "pip", "install", "--no-cache-dir", "-q", "-r", req_file
        ], check=False, timeout=300)

def ensure_custom_nodes():
    """Ensure LTX-Video custom nodes are installed"""
    custom_nodes = f"{COMFY_HOME}/custom_nodes"
    os.makedirs(custom_nodes, exist_ok=True)

    ltxv_path = f"{custom_nodes}/ComfyUI-LTXVideo"
    vhs_path = f"{custom_nodes}/ComfyUI-VideoHelperSuite"

    # Install LTX-Video nodes if missing
    if not os.path.exists(f"{ltxv_path}/nodes.py"):
        print("Installing ComfyUI-LTXVideo custom nodes...")
        import shutil
        shutil.rmtree(ltxv_path, ignore_errors=True)
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/Lightricks/ComfyUI-LTXVideo.git",
            ltxv_path
        ], check=True, timeout=120)
        req = f"{ltxv_path}/requirements.txt"
        if os.path.exists(req):
            subprocess.run(["pip", "install", "--no-cache-dir", "-q", "-r", req], check=False, timeout=300)
        print("ComfyUI-LTXVideo installed!")

    # Install VideoHelperSuite if missing
    if not os.path.exists(f"{vhs_path}/__init__.py"):
        print("Installing ComfyUI-VideoHelperSuite...")
        import shutil
        shutil.rmtree(vhs_path, ignore_errors=True)
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
            vhs_path
        ], check=True, timeout=120)
        req = f"{vhs_path}/requirements.txt"
        if os.path.exists(req):
            subprocess.run(["pip", "install", "--no-cache-dir", "-q", "-r", req], check=False, timeout=300)
        print("ComfyUI-VideoHelperSuite installed!")

def install_comfyui(force_reinstall: bool = False):
    """Install ComfyUI to network volume if not present"""
    if force_reinstall and os.path.exists(COMFY_HOME):
        print("Force reinstalling ComfyUI...")
        import shutil
        shutil.rmtree(COMFY_HOME, ignore_errors=True)

    if os.path.exists(f"{COMFY_HOME}/main.py"):
        print("ComfyUI already installed on network volume")
        # Always ensure deps are up to date
        install_comfyui_deps()
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

        for req in [
            f"{custom_nodes}/ComfyUI-LTXVideo/requirements.txt",
            f"{custom_nodes}/ComfyUI-VideoHelperSuite/requirements.txt",
        ]:
            if os.path.exists(req):
                subprocess.run(
                    ["pip", "install", "--no-cache-dir", "-r", req],
                    check=True,
                    timeout=600,
                )

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

def download_hf_file(repo_id: str, filename: str, dest_dir: str, token: str = None):
    """Download a single file from Hugging Face into dest_dir."""
    dest_path = Path(dest_dir) / filename
    if dest_path.exists():
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if "/" in filename:
            raise ValueError("nested path")
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        fast_download(url, str(dest_path), token)
    except Exception as e:
        print(f"aria2c failed ({e}), falling back to hf_hub_download...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dest_dir,
            token=token or None,
        )
    return True

def download_gemma_text_encoder(token: str = None):
    """Download Gemma text encoder repo if missing."""
    gemma_root = Path(MODEL_DIR) / "text_encoders" / GEMMA_DIRNAME
    required_file = gemma_root / "model-00001-of-00005.safetensors"
    if required_file.exists():
        return False

    print(f"Downloading Gemma text encoder ({GEMMA_REPO})...")
    gemma_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=GEMMA_REPO,
        local_dir=str(gemma_root),
        token=token or None,
        local_dir_use_symlinks=False,
    )
    print("Gemma text encoder downloaded!")
    return True

def ensure_required_models(workflow: Dict[str, Any]):
    """Ensure required models are present based on the workflow."""
    required = {
        "checkpoints": set(),
        "loras": set(),
        "latent_upscale_models": set(),
        "text_encoders": set(),
    }
    downloaded_any = False

    for node in workflow.values():
        class_type = node.get("class_type")
        inputs = node.get("inputs", {})

        if class_type in {"CheckpointLoaderSimple", "LowVRAMCheckpointLoader"}:
            ckpt_name = inputs.get("ckpt_name")
            if ckpt_name:
                required["checkpoints"].add(ckpt_name)

        if class_type == "LTXVAudioVAELoader":
            ckpt_name = inputs.get("ckpt_name")
            if ckpt_name:
                required["checkpoints"].add(ckpt_name)

        if class_type == "LTXVGemmaCLIPModelLoader":
            gemma_path = inputs.get("gemma_path")
            ltxv_path = inputs.get("ltxv_path")
            if gemma_path:
                required["text_encoders"].add(gemma_path)
            if ltxv_path:
                required["checkpoints"].add(ltxv_path)

        if class_type in {"LoraLoaderModelOnly", "LTXVQ8LoraModelLoader"}:
            lora_name = inputs.get("lora_name")
            if lora_name:
                required["loras"].add(lora_name)

        if class_type in {"LatentUpscaleModelLoader", "LowVRAMLatentUpscaleModelLoader"}:
            model_name = inputs.get("model_name")
            if model_name:
                required["latent_upscale_models"].add(model_name)

    for ckpt_name in sorted(required["checkpoints"]):
        if ckpt_name.startswith("ltx-2-"):
            if download_hf_file(LTX2_MODEL_REPO, ckpt_name, f"{MODEL_DIR}/checkpoints", HF_TOKEN):
                downloaded_any = True
        else:
            print(f"Checkpoint not found locally and repo unknown: {ckpt_name}")

    for lora_name in sorted(required["loras"]):
        if lora_name.startswith("ltx-2-"):
            if download_hf_file(LTX2_MODEL_REPO, lora_name, f"{MODEL_DIR}/loras", HF_TOKEN):
                downloaded_any = True
        else:
            print(f"LoRA not found locally and repo unknown: {lora_name}")

    for model_name in sorted(required["latent_upscale_models"]):
        if model_name.startswith("ltx-2-"):
            if download_hf_file(LTX2_MODEL_REPO, model_name, f"{MODEL_DIR}/latent_upscale_models", HF_TOKEN):
                downloaded_any = True
        else:
            print(f"Upscaler not found locally and repo unknown: {model_name}")

    for encoder_path in sorted(required["text_encoders"]):
        if encoder_path.startswith(f"{GEMMA_DIRNAME}/") or encoder_path == GEMMA_DIRNAME:
            if download_gemma_text_encoder(HF_TOKEN):
                downloaded_any = True
        else:
            print(f"Text encoder not found locally and repo unknown: {encoder_path}")

    if downloaded_any:
        global COMFY_RESTART_REQUIRED
        COMFY_RESTART_REQUIRED = True

def download_models(force: bool = False):
    """Download LTX-2 models to network volume"""

    # LTX-2 (19B) main model - fp8 quantized (~20GB)
    model_path = f"{MODEL_DIR}/checkpoints/{LTX2_MODEL_FILENAME}"
    if not os.path.exists(model_path) or force:
        print(f"Downloading LTX-2 model ({LTX2_MODEL_FILENAME})...")
        try:
            url = f"https://huggingface.co/{LTX2_MODEL_REPO}/resolve/main/{LTX2_MODEL_FILENAME}"
            fast_download(url, model_path, HF_TOKEN)
        except Exception as e:
            print(f"aria2c failed ({e}), falling back to hf_hub_download...")
            hf_hub_download(
                repo_id=LTX2_MODEL_REPO,
                filename=LTX2_MODEL_FILENAME,
                local_dir=f"{MODEL_DIR}/checkpoints",
                token=HF_TOKEN or None,
            )
        print("LTX-2 model downloaded!")

    # Spatial Upscaler (required for high-res output)
    spatial_path = f"{MODEL_DIR}/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors"
    if not os.path.exists(spatial_path) or force:
        print("Downloading spatial upscaler...")
        try:
            url = f"https://huggingface.co/{LTX2_MODEL_REPO}/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors"
            fast_download(url, spatial_path, HF_TOKEN)
        except Exception as e:
            print(f"aria2c failed ({e}), falling back to hf_hub_download...")
            try:
                hf_hub_download(
                    repo_id=LTX2_MODEL_REPO,
                    filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
                    local_dir=f"{MODEL_DIR}/latent_upscale_models",
                    token=HF_TOKEN or None,
                )
            except Exception as e2:
                print(f"Spatial upscaler download note: {e2}")

    # Temporal Upscaler (for more frames)
    temporal_path = f"{MODEL_DIR}/latent_upscale_models/ltx-2-temporal-upscaler-x2-1.0.safetensors"
    if not os.path.exists(temporal_path) or force:
        print("Downloading temporal upscaler...")
        try:
            url = f"https://huggingface.co/{LTX2_MODEL_REPO}/resolve/main/ltx-2-temporal-upscaler-x2-1.0.safetensors"
            fast_download(url, temporal_path, HF_TOKEN)
        except Exception as e:
            print(f"aria2c failed ({e}), falling back to hf_hub_download...")
            try:
                hf_hub_download(
                    repo_id=LTX2_MODEL_REPO,
                    filename="ltx-2-temporal-upscaler-x2-1.0.safetensors",
                    local_dir=f"{MODEL_DIR}/latent_upscale_models",
                    token=HF_TOKEN or None,
                )
            except Exception as e2:
                print(f"Temporal upscaler download note: {e2}")

    # Distilled LoRA (for two-stage pipeline)
    lora_path = f"{MODEL_DIR}/loras/ltx-2-19b-distilled-lora-384.safetensors"
    if not os.path.exists(lora_path) or force:
        print("Downloading distilled LoRA...")
        try:
            url = f"https://huggingface.co/{LTX2_MODEL_REPO}/resolve/main/ltx-2-19b-distilled-lora-384.safetensors"
            fast_download(url, lora_path, HF_TOKEN)
        except Exception as e:
            print(f"aria2c failed ({e}), falling back to hf_hub_download...")
            try:
                hf_hub_download(
                    repo_id=LTX2_MODEL_REPO,
                    filename="ltx-2-19b-distilled-lora-384.safetensors",
                    local_dir=f"{MODEL_DIR}/loras",
                    token=HF_TOKEN or None,
                )
            except Exception as e2:
                print(f"Distilled LoRA download note: {e2}")

    print("All LTX-2 models ready!")

def setup_environment(preload_models: bool = False):
    """Full setup on first run"""
    print("=" * 50)
    print("Setting up LTX-2 environment on network volume...")
    print("=" * 50)

    ensure_directories()
    install_comfyui()
    ensure_custom_nodes()  # Ensure LTX-Video nodes are installed
    ensure_comfy_model_links()
    if preload_models:
        download_models()

    print("Setup complete!")

# =============================================================================
# ComfyUI Helpers
# =============================================================================

def ensure_comfyui_running():
    """Start ComfyUI once per worker and wait for the API to respond."""
    global COMFY_PROCESS, COMFY_RESTART_REQUIRED

    if COMFY_PROCESS and COMFY_PROCESS.poll() is None:
        if COMFY_RESTART_REQUIRED:
            print("Restarting ComfyUI to apply updated model paths...")
            COMFY_PROCESS.terminate()
            try:
                COMFY_PROCESS.wait(timeout=10)
            except Exception:
                COMFY_PROCESS.kill()
            COMFY_PROCESS = None
            COMFY_RESTART_REQUIRED = False
        else:
            return

    # Log file for ComfyUI output
    log_path = Path(TMPDIR) / "comfyui.log"
    log_file = open(log_path, "w")

    cmd = [
        sys.executable,
        f"{COMFY_HOME}/main.py",
        "--listen",
        COMFY_HOST,
        "--port",
        str(COMFY_PORT),
        "--disable-auto-launch",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print(f"Starting ComfyUI: {' '.join(cmd)}")
    COMFY_PROCESS = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=COMFY_HOME
    )

    for i in range(120):  # Wait up to 2 minutes
        # Check if process died
        if COMFY_PROCESS.poll() is not None:
            log_file.close()
            log_content = log_path.read_text()[-5000:]  # Last 5000 chars
            raise RuntimeError(f"ComfyUI exited with code {COMFY_PROCESS.returncode}. Log:\n{log_content}")

        try:
            resp = requests.get(f"{COMFY_URL}/history", timeout=2)
            if resp.status_code == 200:
                print(f"ComfyUI started successfully after {i} seconds")
                return
        except Exception:
            pass
        time.sleep(1)

    log_file.close()
    log_content = log_path.read_text()[-5000:]
    raise RuntimeError(f"ComfyUI failed to respond after 120s. Log:\n{log_content}")

def load_workflow_template(name: str) -> Dict[str, Any]:
    """Load a prompt-format workflow from /app/workflows."""
    path = Path("/app/workflows") / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Workflow not found: {path}")
    with path.open() as f:
        data = json.load(f)
    return data.get("workflow", data)

def download_input_image(value: str) -> str:
    """Download or decode an input image into COMFY_HOME/input and return the filename."""
    input_dir = Path(COMFY_HOME) / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    filename = f"input_{uuid.uuid4().hex}.png"
    dest = input_dir / filename

    if value.startswith("http://") or value.startswith("https://"):
        resp = requests.get(value, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return filename

    if value.startswith("data:image"):
        header, b64data = value.split(",", 1)
        import base64
        dest.write_bytes(base64.b64decode(b64data))
        return filename

    # Assume it's already a filename in input dir
    return value

def update_workflow_inputs(workflow: Dict[str, Any], job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Apply prompt parameters to the prompt-format workflow."""
    prompt = job_input.get("prompt")
    negative_prompt = job_input.get("negative_prompt")
    width = job_input.get("width")
    height = job_input.get("height")
    fps = job_input.get("fps")
    duration_seconds = job_input.get("duration_seconds")
    num_frames = job_input.get("num_frames")
    steps = job_input.get("steps")
    cfg = job_input.get("cfg_scale")
    seed = job_input.get("seed")
    image_strength = job_input.get("image_strength")
    model_choice = job_input.get("model")
    input_image = (
        job_input.get("input_image")
        or job_input.get("input_image_url")
        or job_input.get("image")
        or job_input.get("image_url")
    )
    has_load_image = False
    clip_nodes: Dict[str, Dict[str, Any]] = {}
    conditioning_nodes: List[Dict[str, Any]] = []
    noise_offset = 0

    if isinstance(prompt, str):
        prompt = prompt.strip()
    if not prompt:
        prompt = "" if input_image else None
    if isinstance(negative_prompt, str) and not negative_prompt.strip():
        negative_prompt = None
    if isinstance(seed, str) and not seed.strip():
        seed = None
    if seed is None and AUTO_SEED:
        seed = random.randint(0, 2**31 - 1)

    if model_choice:
        normalized = str(model_choice).strip().lower()
        use_full = normalized in ("full", "dev", "19b", "full-fp8", "ltx-2-19b-dev")
        selected_ckpt = LTX2_FULL_MODEL_FILENAME if use_full else LTX2_MODEL_FILENAME
        lora_strength = 0.0 if use_full else 1.0
        lora_name = "ltx-2-19b-distilled-lora-384.safetensors"
        gemma_default = f"{GEMMA_DIRNAME}/model-00001-of-00005.safetensors"

    apply_fps = fps is not None or duration_seconds is not None
    if fps is None and duration_seconds is not None:
        fps = 24
    if num_frames is None and duration_seconds is not None:
        num_frames = int(float(duration_seconds) * float(fps)) + 1

    for node_id, node in list(workflow.items()):
        class_type = node.get("class_type")
        inputs = node.get("inputs", {})

        if class_type == "CLIPTextEncode":
            clip_nodes[str(node_id)] = node

        if class_type == "LTXVConditioning":
            conditioning_nodes.append(node)

        if class_type == "CheckpointLoaderSimple" and model_choice:
            inputs["ckpt_name"] = selected_ckpt

        if class_type == "LTXVAudioVAELoader" and model_choice:
            inputs["ckpt_name"] = selected_ckpt

        if class_type == "LTXVGemmaCLIPModelLoader" and model_choice:
            inputs["ltxv_path"] = selected_ckpt
            inputs.setdefault("gemma_path", gemma_default)

        if class_type == "LoraLoaderModelOnly" and model_choice:
            inputs["lora_name"] = lora_name
            inputs["strength_model"] = float(lora_strength)

        if class_type == "CLIPTextEncode" and prompt is not None:
            inputs["text"] = prompt

        if class_type == "LTXVTextEncode":
            if prompt is not None:
                inputs["prompt"] = prompt
            if negative_prompt is not None:
                inputs["negative_prompt"] = negative_prompt

        if class_type == "LTXVConditioning" and apply_fps:
            inputs["frame_rate"] = float(fps)

        if class_type == "EmptyImage":
            if width is not None:
                inputs["width"] = int(width)
            if height is not None:
                inputs["height"] = int(height)

        if class_type == "EmptyLTXVLatentVideo" and num_frames is not None:
            inputs["length"] = int(num_frames)

        if class_type == "LTXVEmptyLatentAudio":
            if num_frames is not None:
                inputs["frames_number"] = int(num_frames)
            if apply_fps:
                inputs["frame_rate"] = int(fps)

        if class_type == "RandomNoise" and seed is not None:
            inputs["noise_seed"] = int(seed) + noise_offset
            noise_offset += 1

        if class_type == "CFGGuider" and cfg is not None:
            inputs["cfg"] = float(cfg)

        if class_type == "CreateVideo" and apply_fps:
            inputs["fps"] = float(fps)

        if class_type == "LoadImage":
            has_load_image = True
            if input_image:
                inputs["image"] = download_input_image(input_image)

        if class_type == "LTXVImgToVideoInplace" and image_strength is not None:
            inputs["strength"] = float(image_strength)

        if class_type == "LTXVSampler":
            if steps is not None:
                inputs["steps"] = int(steps)
            if cfg is not None:
                inputs["cfg"] = float(cfg)
            if seed is not None:
                inputs["seed"] = int(seed)

        if class_type == "LTXVAudioGenerate" and prompt is not None:
            inputs["prompt"] = prompt

    if negative_prompt is not None and clip_nodes and conditioning_nodes:
        existing_ids = [int(key) for key in workflow.keys() if str(key).isdigit()]
        next_id = max(existing_ids or [0]) + 1
        negative_clip_id = None
        for conditioning in conditioning_nodes:
            inputs = conditioning.get("inputs", {})
            negative_ref = inputs.get("negative")
            positive_ref = inputs.get("positive")
            neg_id = None
            pos_id = None
            if isinstance(negative_ref, list) and negative_ref:
                neg_id = str(negative_ref[0])
            if isinstance(positive_ref, list) and positive_ref:
                pos_id = str(positive_ref[0])

            if neg_id and neg_id in clip_nodes and neg_id != pos_id:
                clip_nodes[neg_id]["inputs"]["text"] = negative_prompt
                continue

            if pos_id and pos_id in clip_nodes:
                if negative_clip_id is None:
                    base_inputs = clip_nodes[pos_id].get("inputs", {})
                    workflow[str(next_id)] = {
                        "class_type": "CLIPTextEncode",
                        "inputs": {
                            "clip": base_inputs.get("clip"),
                            "text": negative_prompt,
                        },
                    }
                    negative_clip_id = str(next_id)
                    next_id += 1
                inputs["negative"] = [negative_clip_id, 0]

    if has_load_image and not input_image:
        raise ValueError("input_image is required for image-to-video workflows")

    return workflow

def queue_prompt(prompt: Dict[str, Any]) -> str:
    payload = {"prompt": prompt, "client_id": "runpod-ltx2"}
    resp = requests.post(f"{COMFY_URL}/prompt", json=payload, timeout=10)
    if resp.status_code != 200:
        error_detail = resp.text[:2000]
        raise RuntimeError(f"ComfyUI prompt rejected ({resp.status_code}): {error_detail}")
    data = resp.json()
    prompt_id = data["prompt_id"]
    print(f"ComfyUI prompt queued: {prompt_id}")
    return prompt_id

def get_queue_snapshot() -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(f"{COMFY_URL}/queue", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None

def wait_for_prompt(prompt_id: str, timeout_seconds: int = 1800) -> Dict[str, Any]:
    start = time.time()
    last_log = 0.0
    while time.time() - start < timeout_seconds:
        resp = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if prompt_id in data:
                entry = data[prompt_id]
                outputs = entry.get("outputs")
                if outputs:
                    return entry
        now = time.time()
        if now - last_log >= 30:
            elapsed = int(now - start)
            print(f"Waiting for prompt {prompt_id}... {elapsed}s elapsed")
            queue = get_queue_snapshot()
            if queue:
                running = len(queue.get("queue_running", []))
                pending = len(queue.get("queue_pending", []))
                print(f"ComfyUI queue: running={running} pending={pending}")
            last_log = now
        time.sleep(2)
    raise TimeoutError("ComfyUI prompt timed out")

def extract_output_files(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    outputs = entry.get("outputs", {})
    found = []
    for out in outputs.values():
        if isinstance(out, dict):
            for key in ("images", "gifs", "videos", "audio", "files"):
                for item in out.get(key, []) or []:
                    if isinstance(item, dict) and "filename" in item:
                        found.append(item)
    return found

def resolve_output_path(item: Dict[str, Any]) -> Path:
    filename = item["filename"]
    subfolder = item.get("subfolder", "")
    out_type = item.get("type", "output")
    base = Path(COMFY_HOME)
    if out_type == "input":
        base = base / "input"
    elif out_type == "temp":
        base = base / "temp"
    else:
        base = base / "output"
    return base / subfolder / filename

def run_comfyui_workflow(workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
    ensure_comfyui_running()
    prompt_id = queue_prompt(workflow)
    entry = wait_for_prompt(prompt_id)
    files = extract_output_files(entry)
    if not files:
        raise RuntimeError("No output files produced by ComfyUI")
    print(f"ComfyUI produced {len(files)} file(s)")
    for item in files:
        filename = item.get("filename", "unknown")
        out_type = item.get("type", "output")
        subfolder = item.get("subfolder", "")
        print(f"ComfyUI output: {filename} (type={out_type}, subfolder={subfolder})")
    return files

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
    _log_ltx2_event("job_input", job_input)

    try:
        action = job_input.get("action", "generate")
        _log_ltx2_event("action", {"action": action})

        setup_environment(preload_models=PRELOAD_MODELS and action != "sync_models")
        gpu = detect_gpu()
        print(f"GPU: {gpu['gpu_name']} ({gpu['vram_gb']}GB)")

        if action == "sync_models":
            download_models(force=job_input.get("force", False))
            return {"status": "success", "action": "sync_models", "gpu": gpu}

        if action == "reinstall_comfyui":
            install_comfyui(force_reinstall=True)
            return {"status": "success", "action": "reinstall_comfyui", "gpu": gpu}

        if action == "status":
            return {
                "status": "success",
                "comfyui_installed": os.path.exists(f"{COMFY_HOME}/main.py"),
                "model_exists": os.path.exists(f"{MODEL_DIR}/checkpoints/{LTX2_MODEL_FILENAME}"),
                "gpu": gpu,
            }

        if action == "list_nodes":
            # Start ComfyUI and get available nodes
            ensure_comfyui_running()
            try:
                resp = requests.get(f"{COMFY_URL}/object_info", timeout=30)
                if resp.status_code == 200:
                    nodes = list(resp.json().keys())
                    ltx_nodes = [n for n in nodes if "ltx" in n.lower() or "LTX" in n]
                    return {
                        "status": "success",
                        "total_nodes": len(nodes),
                        "ltx_nodes": ltx_nodes,
                        "gpu": gpu,
                    }
            except Exception as e:
                return {"status": "error", "error": f"Failed to get nodes: {e}"}

        if action == "list_models":
            checkpoints = sorted(Path(f"{MODEL_DIR}/checkpoints").glob("*.safetensors"))
            loras = sorted(Path(f"{MODEL_DIR}/loras").glob("*.safetensors"))
            upscalers = sorted(Path(f"{MODEL_DIR}/latent_upscale_models").glob("*.safetensors"))
            return {
                "status": "success",
                "checkpoints": [p.name for p in checkpoints],
                "loras": [p.name for p in loras],
                "latent_upscalers": [p.name for p in upscalers],
                "gpu": gpu,
            }

        if action == "generate":
            workflow_input = job_input.get("workflow")
            if not workflow_input:
                return {"status": "error", "error": "workflow is required - pass a ComfyUI API workflow JSON"}

            if isinstance(workflow_input, dict):
                workflow = workflow_input
            else:
                # Try loading from template file
                workflow = load_workflow_template(str(workflow_input))

            prompt_value = job_input.get("prompt")
            prompt_text = prompt_value.strip() if isinstance(prompt_value, str) else ""
            input_image = (
                job_input.get("input_image")
                or job_input.get("input_image_url")
                or job_input.get("image")
                or job_input.get("image_url")
            )
            requires_prompt = isinstance(workflow_input, str) and "image" not in str(workflow_input).lower()
            if requires_prompt and not prompt_text and not input_image:
                error_msg = "prompt is required for text-to-video workflows"
                _log_ltx2_event("validation_error", {"error": error_msg, "workflow": workflow_input})
                return {"status": "error", "error": error_msg}

            # Optionally update workflow inputs if provided
            workflow_label = "<inline>" if isinstance(workflow_input, dict) else str(workflow_input)
            _log_ltx2_event("workflow_selected", {"workflow": workflow_label})
            updated = False
            if any(k in job_input for k in ["prompt", "negative_prompt", "width", "height", "num_frames", "fps", "steps", "cfg_scale", "seed", "duration_seconds", "input_image", "input_image_url", "image", "image_url", "image_strength", "model"]):
                workflow = update_workflow_inputs(workflow, job_input)
                updated = True
            summary = _summarize_workflow(workflow)
            summary["updated"] = updated
            seed_value = job_input.get("seed")
            seed_missing = seed_value is None or (isinstance(seed_value, str) and not seed_value.strip())
            if seed_missing and AUTO_SEED:
                summary["seed_source"] = "auto"
            elif seed_missing:
                summary["seed_source"] = "default"
            else:
                summary["seed_source"] = "client"
            _log_ltx2_event("workflow_summary", summary)

            # Ensure required models exist before sending to ComfyUI
            ensure_required_models(workflow)

            outputs = run_comfyui_workflow(workflow)

            uploaded = []
            for item in outputs:
                path = resolve_output_path(item)
                if path.exists():
                    content_type = "video/mp4"
                    if path.suffix.lower() == ".gif":
                        content_type = "image/gif"
                    uploaded.append(upload_to_s3(str(path), content_type=content_type))

            _log_ltx2_event("generation_outputs", uploaded)

            return {
                "status": "success",
                "outputs": uploaded,
                "gpu": gpu,
                "execution_time": round(time.time() - start_time, 2),
            }

        error_result = {"status": "error", "error": f"Unknown action: {action}"}
        _log_ltx2_event("handler_error", error_result)
        return error_result

    except Exception as e:
        error_result = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
        _log_ltx2_event("handler_exception", {"error": str(e)})
        return error_result

if __name__ == "__main__":
    print("LTX-2 Worker Starting...")
    runpod.serverless.start({"handler": handler})
