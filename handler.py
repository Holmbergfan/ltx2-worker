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
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files

# Enable fast HF downloads (must be set before any HF calls)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Logging configuration
LOG_LEVEL = os.environ.get("LTX2_LOG_LEVEL", "INFO").upper()
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARN": 30,
    "WARNING": 30,
    "ERROR": 40,
}
LOG_LEVEL_NUM = LOG_LEVELS.get(LOG_LEVEL, 20)
LOG_VERBOSE = LOG_LEVEL_NUM <= LOG_LEVELS["DEBUG"]
STATUS_LOG_INTERVAL = int(os.environ.get("LTX2_STATUS_LOG_INTERVAL", "60"))

if not LOG_VERBOSE:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def log(message: str, level: str = "INFO") -> None:
    level_upper = level.upper()
    if LOG_LEVELS.get(level_upper, 20) >= LOG_LEVEL_NUM:
        print(f"[LTX2] {level_upper}: {message}")

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
        log("Warning: /runpod-volume not available, falling back to /workspace.", level="WARN")
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
        log(
            "Disk usage for %s: total=%.1fGB used=%.1fGB free=%.1fGB"
            % (path, total / (1024**3), used / (1024**3), free / (1024**3)),
            level="DEBUG"
        )
    except FileNotFoundError:
        log(f"Disk usage for {path}: path not found", level="DEBUG")


def run_cmd(cmd: List[str], desc: str, timeout: Optional[int] = None, check: bool = True, quiet: Optional[bool] = None, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    if quiet is None:
        quiet = not LOG_VERBOSE
    if quiet:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
    else:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
    if result.returncode != 0:
        message = f"{desc} failed (exit {result.returncode})"
        if quiet and result.stdout:
            tail = result.stdout.strip().splitlines()[-6:]
            if tail:
                message = f"{message}: {' | '.join(tail)}"
        log(message, level="ERROR" if check else "WARN")
        if check:
            raise RuntimeError(message)
    return result


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


def _log_ltx2_event(label: str, payload: Any, level: str = "DEBUG") -> None:
    if LOG_LEVELS.get(level.upper(), 20) < LOG_LEVEL_NUM:
        return
    try:
        sanitized = _sanitize_log_value(payload)
        log(f"{label}: {json.dumps(sanitized, ensure_ascii=True)}", level=level)
    except Exception as exc:
        log(f"{label}: <log failed: {exc}>", level="WARN")


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


def _range_or_value(values: List[Any]) -> Optional[str]:
    if not values:
        return None
    try:
        unique = sorted(set(values))
        if len(unique) == 1:
            return str(unique[0])
        return f"{unique[0]}-{unique[-1]}"
    except Exception:
        return str(values[0])


def _count_and_max_len(values: List[str]) -> Optional[str]:
    if not values:
        return None
    lengths = [len(v) for v in values if isinstance(v, str)]
    if not lengths:
        return str(len(values))
    return f"{len(lengths)} (max {max(lengths)} chars)"


def summarize_job_input(job_input: Dict[str, Any]) -> Dict[str, Any]:
    prompt = job_input.get("prompt") or ""
    negative_prompt = job_input.get("negative_prompt") or ""
    input_image = (
        job_input.get("input_image")
        or job_input.get("input_image_url")
        or job_input.get("image")
        or job_input.get("image_url")
    )
    input_images = _coerce_image_list(job_input.get("input_images"))
    input_keyframes = _coerce_keyframes(
        job_input.get("input_keyframes")
        or job_input.get("input_keyframe_indices")
    )
    input_label = None
    if isinstance(input_image, str) and input_image:
        input_label = "inline" if input_image.startswith("data:") else "url"
    if input_images:
        input_label = f"{len(input_images)} refs"

    width = job_input.get("width")
    height = job_input.get("height")
    size = f"{width}x{height}" if width and height else None
    workflow_value = job_input.get("workflow")
    workflow_label = None
    if isinstance(workflow_value, dict):
        workflow_label = "<inline>"
    elif workflow_value:
        workflow_label = str(workflow_value)

    return {
        "action": job_input.get("action", "generate"),
        "workflow": workflow_label,
        "model": job_input.get("model"),
        "size": size,
        "frames": job_input.get("num_frames"),
        "fps": job_input.get("fps"),
        "steps": job_input.get("steps"),
        "cfg": job_input.get("cfg_scale"),
        "seed": job_input.get("seed"),
        "image_strength": job_input.get("image_strength"),
        "prompt_len": len(prompt) if prompt else 0,
        "negative_len": len(negative_prompt) if negative_prompt else 0,
        "input_image": input_label,
        "keyframes": len(input_keyframes) if input_keyframes else None,
    }


def log_job_summary(job_input: Dict[str, Any]) -> None:
    summary = summarize_job_input(job_input)
    parts = []
    for key in ("action", "workflow", "model"):
        if summary.get(key):
            parts.append(f"{key}={summary[key]}")
    if summary.get("size"):
        parts.append(f"size={summary['size']}")
    if summary.get("frames") is not None:
        parts.append(f"frames={summary['frames']}")
    if summary.get("fps") is not None:
        parts.append(f"fps={summary['fps']}")
    if summary.get("steps") is not None:
        parts.append(f"steps={summary['steps']}")
    if summary.get("cfg") is not None:
        parts.append(f"cfg={summary['cfg']}")
    if summary.get("image_strength") is not None:
        parts.append(f"image_strength={summary['image_strength']}")
    if summary.get("seed") is not None:
        parts.append(f"seed={summary['seed']}")
    if summary.get("input_image"):
        parts.append(f"input_image={summary['input_image']}")
    if summary.get("keyframes") is not None:
        parts.append(f"keyframes={summary['keyframes']}")
    if summary.get("prompt_len"):
        parts.append(f"prompt_len={summary['prompt_len']}")
    if summary.get("negative_len"):
        parts.append(f"negative_len={summary['negative_len']}")

    if parts:
        log("Job " + " | ".join(parts), level="INFO")


def log_workflow_summary(summary: Dict[str, Any]) -> None:
    parts = [f"nodes={summary.get('node_count', 0)}"]
    sizes = summary.get("sizes") or []
    if sizes and isinstance(sizes[0], dict):
        width = sizes[0].get("width")
        height = sizes[0].get("height")
        if width and height:
            size_label = f"{width}x{height}"
            if len(sizes) > 1:
                size_label = f"{size_label} (+{len(sizes) - 1})"
            parts.append(f"size={size_label}")
    frames = _range_or_value(summary.get("lengths", []))
    if frames:
        parts.append(f"frames={frames}")
    fps = _range_or_value(summary.get("fps", []))
    if fps:
        parts.append(f"fps={fps}")
    cfg = _range_or_value(summary.get("cfg", []))
    if cfg:
        parts.append(f"cfg={cfg}")
    strengths = _range_or_value(summary.get("image_strengths", []))
    if strengths:
        parts.append(f"image_strength={strengths}")
    prompts = _count_and_max_len(summary.get("clip_texts", []) + summary.get("ltxv_prompts", []))
    if prompts:
        parts.append(f"prompts={prompts}")
    negatives = _count_and_max_len(summary.get("ltxv_negative_prompts", []))
    if negatives:
        parts.append(f"negatives={negatives}")
    input_images = summary.get("input_images") or []
    if input_images:
        parts.append(f"input_images={len(input_images)}")
    if summary.get("seed_source"):
        parts.append(f"seed_source={summary['seed_source']}")
    if summary.get("updated"):
        parts.append("updated_inputs=yes")

    log("Workflow " + " | ".join(parts), level="INFO")
log(
    f"Storage root={DEFAULT_ROOT} | MODEL_DIR={MODEL_DIR} | COMFY_HOME={COMFY_HOME} | TMPDIR={TMPDIR} | AUTO_SEED={AUTO_SEED}",
    level="INFO"
)
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
PRELOAD_MODELS = os.environ.get("PRELOAD_MODELS", "true").lower() in ("1", "true", "yes")
LTX2_DOWNLOAD_ALL = os.environ.get("LTX2_DOWNLOAD_ALL", "true").lower() in ("1", "true", "yes")

# Actual files that exist in Lightricks/LTX-2 repo (verified 2026-01-18)
LTX2_README_FILES = [
    "ltx-2-19b-distilled-fp8.safetensors",      # Main distilled model (fp8) - ~27GB
    "ltx-2-19b-dev-fp8.safetensors",            # Main full model (fp8) - ~27GB
    "ltx-2-19b-distilled.safetensors",          # Distilled full precision - ~43GB
    "ltx-2-19b-dev.safetensors",                # Full model full precision - ~43GB
    "ltx-2-19b-dev-fp4.safetensors",            # Full model fp4 quantized - ~20GB
    "ltx-2-19b-distilled-lora-384.safetensors", # Distilled LoRA - ~8GB
    "ltx-2-spatial-upscaler-x2-1.0.safetensors", # Spatial upscaler - ~1GB
    "ltx-2-temporal-upscaler-x2-1.0.safetensors", # Temporal upscaler - ~0.3GB
]

# IC-LoRA and Camera Control LoRAs are in separate repos (each ~624MB)
# Format: (repo_id, filename, dest_subdir)
LTX2_LORA_REPOS = [
    ("Lightricks/LTX-2-19b-IC-LoRA-Canny-Control", "ltx-2-19b-ic-lora-canny-control.safetensors", "loras"),
    ("Lightricks/LTX-2-19b-IC-LoRA-Depth-Control", "ltx-2-19b-ic-lora-depth-control.safetensors", "loras"),
    ("Lightricks/LTX-2-19b-IC-LoRA-Pose-Control", "ltx-2-19b-ic-lora-pose-control.safetensors", "loras"),
    ("Lightricks/LTX-2-19b-IC-LoRA-Detailer", "ltx-2-19b-ic-lora-detailer.safetensors", "loras"),
    ("Lightricks/LTX-2-19b-LoRA-Camera-Control-Static", "ltx-2-19b-lora-camera-control-static.safetensors", "loras"),
    ("Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up", "ltx-2-19b-lora-camera-control-jib-up.safetensors", "loras"),
    ("Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down", "ltx-2-19b-lora-camera-control-jib-down.safetensors", "loras"),
    ("Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In", "ltx-2-19b-lora-camera-control-dolly-in.safetensors", "loras"),
    ("Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out", "ltx-2-19b-lora-camera-control-dolly-out.safetensors", "loras"),
    ("Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left", "ltx-2-19b-lora-camera-control-dolly-left.safetensors", "loras"),
    ("Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right", "ltx-2-19b-lora-camera-control-dolly-right.safetensors", "loras"),
]
LTX2_DOWNLOAD_LORAS = os.environ.get("LTX2_DOWNLOAD_LORAS", "false").lower() in ("1", "true", "yes")

# S3 Storage
S3_BUCKET = os.environ.get("S3_BUCKET", "ltx2-models")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "")
S3_REGION = os.environ.get("S3_REGION", "eu-north-1")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
PRESIGNED_URL_EXPIRY = int(os.environ.get("PRESIGNED_URL_EXPIRY", 14400))

COMFY_PROCESS = None
COMFY_RESTART_REQUIRED = False
LTX2_ALL_DOWNLOAD_ATTEMPTED = False

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
    log(f"Directories ready in {MODEL_DIR}", level="DEBUG")

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
            log(f"Failed to remove models symlink ({e}); leaving as-is.", level="WARN")
            return

    if not os.path.exists(comfy_models):
        os.makedirs(os.path.dirname(comfy_models), exist_ok=True)
        try:
            os.symlink(MODEL_DIR, comfy_models)
            log(f"Linked ComfyUI models -> {MODEL_DIR}", level="DEBUG")
            return
        except OSError as e:
            log(f"Symlink to models failed ({e}), falling back to subdir links...", level="WARN")

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
            log(f"Symlink for {subdir} failed: {e}", level="WARN")

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
        log("Installing/updating ComfyUI dependencies...", level="INFO")
        run_cmd(
            ["pip", "install", "--no-cache-dir", "-q", "--root-user-action=ignore", "--disable-pip-version-check", "-r", req_file],
            desc="ComfyUI deps install",
            timeout=300,
            check=False,
        )

def ensure_custom_nodes():
    """Ensure LTX-Video custom nodes are installed"""
    custom_nodes = f"{COMFY_HOME}/custom_nodes"
    os.makedirs(custom_nodes, exist_ok=True)

    ltxv_path = f"{custom_nodes}/ComfyUI-LTXVideo"
    vhs_path = f"{custom_nodes}/ComfyUI-VideoHelperSuite"

    # Install LTX-Video nodes if missing
    if not os.path.exists(f"{ltxv_path}/nodes.py"):
        log("Installing ComfyUI-LTXVideo custom nodes...", level="INFO")
        import shutil
        shutil.rmtree(ltxv_path, ignore_errors=True)
        clone_cmd = [
            "git", "clone", "--depth", "1",
            "https://github.com/Lightricks/ComfyUI-LTXVideo.git",
            ltxv_path
        ]
        if not LOG_VERBOSE:
            clone_cmd.insert(2, "--quiet")
        run_cmd(clone_cmd, desc="Clone ComfyUI-LTXVideo", timeout=120, check=True)
        req = f"{ltxv_path}/requirements.txt"
        if os.path.exists(req):
            run_cmd(
                ["pip", "install", "--no-cache-dir", "-q", "--root-user-action=ignore", "--disable-pip-version-check", "-r", req],
                desc="ComfyUI-LTXVideo deps",
                timeout=300,
                check=False,
            )
        log("ComfyUI-LTXVideo installed!", level="INFO")

    # Install VideoHelperSuite if missing
    if not os.path.exists(f"{vhs_path}/__init__.py"):
        log("Installing ComfyUI-VideoHelperSuite...", level="INFO")
        import shutil
        shutil.rmtree(vhs_path, ignore_errors=True)
        clone_cmd = [
            "git", "clone", "--depth", "1",
            "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
            vhs_path
        ]
        if not LOG_VERBOSE:
            clone_cmd.insert(2, "--quiet")
        run_cmd(clone_cmd, desc="Clone ComfyUI-VideoHelperSuite", timeout=120, check=True)
        req = f"{vhs_path}/requirements.txt"
        if os.path.exists(req):
            run_cmd(
                ["pip", "install", "--no-cache-dir", "-q", "--root-user-action=ignore", "--disable-pip-version-check", "-r", req],
                desc="ComfyUI-VideoHelperSuite deps",
                timeout=300,
                check=False,
            )
        log("ComfyUI-VideoHelperSuite installed!", level="INFO")

def install_comfyui(force_reinstall: bool = False):
    """Install ComfyUI to network volume if not present"""
    if force_reinstall and os.path.exists(COMFY_HOME):
        log("Force reinstalling ComfyUI...", level="WARN")
        import shutil
        shutil.rmtree(COMFY_HOME, ignore_errors=True)

    if os.path.exists(f"{COMFY_HOME}/main.py"):
        log("ComfyUI already installed on network volume", level="DEBUG")
        # Always ensure deps are up to date
        install_comfyui_deps()
        return True

    log("Installing ComfyUI to network volume...", level="INFO")
    try:
        clone_cmd = [
            "git", "clone", "--depth", "1",
            "https://github.com/comfyanonymous/ComfyUI.git",
            COMFY_HOME
        ]
        if not LOG_VERBOSE:
            clone_cmd.insert(2, "--quiet")
        run_cmd(clone_cmd, desc="Clone ComfyUI", timeout=300, check=True)

        run_cmd(
            ["pip", "install", "--no-cache-dir", "--root-user-action=ignore", "--disable-pip-version-check", "-r", f"{COMFY_HOME}/requirements.txt"],
            desc="ComfyUI requirements install",
            timeout=600,
            check=True,
        )

        # Install LTX-Video nodes
        custom_nodes = f"{COMFY_HOME}/custom_nodes"
        os.makedirs(custom_nodes, exist_ok=True)

        ltxv_path = f"{custom_nodes}/ComfyUI-LTXVideo"
        if not os.path.exists(f"{ltxv_path}/nodes.py"):
            import shutil
            shutil.rmtree(ltxv_path, ignore_errors=True)
            clone_cmd = [
                "git", "clone", "--depth", "1",
                "https://github.com/Lightricks/ComfyUI-LTXVideo.git",
                ltxv_path
            ]
            if not LOG_VERBOSE:
                clone_cmd.insert(2, "--quiet")
            run_cmd(clone_cmd, desc="Clone ComfyUI-LTXVideo", timeout=120, check=True)
        else:
            log("ComfyUI-LTXVideo already installed, skipping clone", level="INFO")

        vhs_path = f"{custom_nodes}/ComfyUI-VideoHelperSuite"
        if not os.path.exists(f"{vhs_path}/videohelpersuite"):
            import shutil
            shutil.rmtree(vhs_path, ignore_errors=True)
            clone_cmd = [
                "git", "clone", "--depth", "1",
                "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
                vhs_path
            ]
            if not LOG_VERBOSE:
                clone_cmd.insert(2, "--quiet")
            run_cmd(clone_cmd, desc="Clone ComfyUI-VideoHelperSuite", timeout=120, check=True)
        else:
            log("ComfyUI-VideoHelperSuite already installed, skipping clone", level="INFO")

        for req in [
            f"{custom_nodes}/ComfyUI-LTXVideo/requirements.txt",
            f"{custom_nodes}/ComfyUI-VideoHelperSuite/requirements.txt",
        ]:
            if os.path.exists(req):
                run_cmd(
                    ["pip", "install", "--no-cache-dir", "--root-user-action=ignore", "--disable-pip-version-check", "-r", req],
                    desc=f"Install {Path(req).parent.name} requirements",
                    timeout=600,
                    check=True,
                )

        log("ComfyUI installed!", level="INFO")
        return True
    except Exception as e:
        log(f"ComfyUI install error: {e}", level="ERROR")
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
    log(f"Downloading with aria2c: {os.path.basename(output_path)}", level="INFO")
    run_cmd(cmd, desc="aria2c download", check=True)

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
        log(f"aria2c failed ({e}), falling back to hf_hub_download...", level="WARN")
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

    log(f"Downloading Gemma text encoder ({GEMMA_REPO})...", level="INFO")
    gemma_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=GEMMA_REPO,
        local_dir=str(gemma_root),
        token=token or None,
        local_dir_use_symlinks=False,
    )
    log("Gemma text encoder downloaded!", level="INFO")
    return True

def _classify_ltx2_asset(filename: str) -> str:
    lower = filename.lower()
    if "upscaler" in lower:
        return f"{MODEL_DIR}/latent_upscale_models"
    if "lora" in lower:
        return f"{MODEL_DIR}/loras"
    return f"{MODEL_DIR}/checkpoints"

# Minimum file sizes (in bytes) for validation - files smaller than this are likely corrupted
MIN_FILE_SIZES = {
    "ltx-2-19b": 1_000_000_000,  # Main models should be >1GB
    "upscaler": 100_000_000,     # Upscalers should be >100MB
    "lora": 100_000_000,         # LoRAs should be >100MB
}

def _get_min_size(filename: str) -> int:
    """Get minimum expected file size for validation."""
    lower = filename.lower()
    if "upscaler" in lower:
        return MIN_FILE_SIZES["upscaler"]
    if "lora" in lower:
        return MIN_FILE_SIZES["lora"]
    if "ltx-2-19b" in lower:
        return MIN_FILE_SIZES["ltx-2-19b"]
    return 0

def _download_repo_asset(repo_id: str, repo_path: str, dest_dir: str, token: str = None, force: bool = False) -> Optional[bool]:
    """Download asset from HuggingFace repo.

    Returns:
        True: Downloaded successfully
        False: Download failed
        None: File already exists and is valid (no download needed)
    """
    dest_path = Path(dest_dir) / os.path.basename(repo_path)
    min_size = _get_min_size(repo_path)

    # Check if file exists and is valid (not empty/corrupted)
    if dest_path.exists() and not force:
        file_size = dest_path.stat().st_size
        if file_size >= min_size:
            log(f"Skipping {repo_path} (already exists, {file_size / 1e9:.2f}GB)", level="DEBUG")
            return None  # File exists and is valid
        else:
            log(f"Removing corrupted/empty file: {dest_path} ({file_size} bytes)", level="WARN")
            dest_path.unlink()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading: {repo_path}", level="INFO")
    try:
        url = f"https://huggingface.co/{repo_id}/resolve/main/{repo_path}"
        fast_download(url, str(dest_path), token)
    except Exception as e:
        log(f"aria2c failed ({e}), falling back to hf_hub_download...", level="WARN")
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=repo_path,
                token=token or None,
            )
            shutil.copyfile(local_path, dest_path)
        except Exception as e2:
            log(f"hf_hub_download failed for {repo_path}: {e2}", level="WARN")
            return False
    return True

def download_all_ltx2_assets(force: bool = False) -> bool:
    """Download every LTX-2 safetensors asset from the repo + README list."""
    global LTX2_ALL_DOWNLOAD_ATTEMPTED
    if LTX2_ALL_DOWNLOAD_ATTEMPTED and not force:
        return False

    repo_files: List[str] = []
    try:
        repo_files = list_repo_files(LTX2_MODEL_REPO, token=HF_TOKEN or None)
    except Exception as e:
        log(f"Failed to list LTX-2 repo files ({e}); falling back to README list.", level="WARN")

    paths_by_base: Dict[str, List[str]] = {}
    paths_by_base_lower: Dict[str, List[str]] = {}
    for path in repo_files:
        base = os.path.basename(path)
        paths_by_base.setdefault(base, []).append(path)
        paths_by_base_lower.setdefault(base.lower(), []).append(path)

    candidates = set(LTX2_README_FILES)
    for path in repo_files:
        base = os.path.basename(path)
        lower = base.lower()
        if lower.endswith(".safetensors") and lower.startswith("ltx-2-"):
            candidates.add(base)

    downloaded_any = False
    failed = 0
    skipped = 0
    total = 0

    for base in sorted(candidates):
        if not base.lower().endswith(".safetensors"):
            continue
        total += 1
        dest_dir = _classify_ltx2_asset(base)
        possible_paths = paths_by_base.get(base) or paths_by_base_lower.get(base.lower()) or [base]
        result = False
        for repo_path in possible_paths:
            result = _download_repo_asset(LTX2_MODEL_REPO, repo_path, dest_dir, HF_TOKEN, force=force)
            if result is True:
                downloaded_any = True
                break
            elif result is None:
                skipped += 1  # File already exists and is valid
                break
        if result is False:
            failed += 1
            log(f"Failed to download {base}", level="WARN")

    ready = total - failed
    if failed:
        log(f"LTX-2 assets: {ready}/{total} ready ({skipped} cached, {ready - skipped} downloaded, {failed} failed)", level="WARN")
    else:
        log(f"LTX-2 assets: {total} ready ({skipped} cached, {total - skipped} downloaded)", level="INFO")

    LTX2_ALL_DOWNLOAD_ATTEMPTED = True
    return downloaded_any

def download_lora_repos(force: bool = False) -> bool:
    """Download IC-LoRA and Camera Control LoRAs from their separate repos."""
    if not LTX2_DOWNLOAD_LORAS:
        return False

    downloaded_any = False
    for repo_id, filename, dest_subdir in LTX2_LORA_REPOS:
        dest_dir = f"{MODEL_DIR}/{dest_subdir}"
        dest_path = Path(dest_dir) / filename
        if dest_path.exists() and not force:
            continue

        log(f"Downloading {filename} from {repo_id}...", level="INFO")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Try aria2c first
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            fast_download(url, str(dest_path), HF_TOKEN)
            downloaded_any = True
        except Exception as e:
            log(f"aria2c failed ({e}), falling back to hf_hub_download...", level="WARN")
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=HF_TOKEN or None,
                )
                shutil.copyfile(local_path, dest_path)
                downloaded_any = True
            except Exception as e2:
                log(f"Failed to download {filename}: {e2}", level="WARN")

    if downloaded_any:
        log("Additional LoRA downloads complete!", level="INFO")
    return downloaded_any

def ensure_required_models(workflow: Dict[str, Any]):
    """Ensure required models are present based on the workflow."""
    if LTX2_DOWNLOAD_ALL:
        download_all_ltx2_assets()
        download_gemma_text_encoder(HF_TOKEN)

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
            log(f"Checkpoint not found locally and repo unknown: {ckpt_name}", level="WARN")

    # Build lookup for LoRAs in separate repos
    lora_repo_map = {filename: (repo_id, filename) for repo_id, filename, _ in LTX2_LORA_REPOS}

    for lora_name in sorted(required["loras"]):
        lora_path = Path(f"{MODEL_DIR}/loras") / lora_name
        if lora_path.exists():
            continue

        # Check if it's in a separate repo
        lora_lower = lora_name.lower()
        found_in_repo = None
        for repo_filename, (repo_id, _) in lora_repo_map.items():
            if lora_lower == repo_filename.lower():
                found_in_repo = (repo_id, repo_filename)
                break

        if found_in_repo:
            repo_id, filename = found_in_repo
            log(f"Downloading {filename} from {repo_id}...", level="INFO")
            try:
                url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
                fast_download(url, str(lora_path), HF_TOKEN)
                downloaded_any = True
            except Exception as e:
                log(f"aria2c failed ({e}), falling back to hf_hub_download...", level="WARN")
                try:
                    local_path = hf_hub_download(repo_id=repo_id, filename=filename, token=HF_TOKEN or None)
                    shutil.copyfile(local_path, lora_path)
                    downloaded_any = True
                except Exception as e2:
                    log(f"Failed to download {lora_name}: {e2}", level="WARN")
        elif lora_name.startswith("ltx-2-"):
            # Try main LTX-2 repo
            if download_hf_file(LTX2_MODEL_REPO, lora_name, f"{MODEL_DIR}/loras", HF_TOKEN):
                downloaded_any = True
        else:
            log(f"LoRA not found locally and repo unknown: {lora_name}", level="WARN")

    for model_name in sorted(required["latent_upscale_models"]):
        if model_name.startswith("ltx-2-"):
            if download_hf_file(LTX2_MODEL_REPO, model_name, f"{MODEL_DIR}/latent_upscale_models", HF_TOKEN):
                downloaded_any = True
        else:
            log(f"Upscaler not found locally and repo unknown: {model_name}", level="WARN")

    for encoder_path in sorted(required["text_encoders"]):
        if encoder_path.startswith(f"{GEMMA_DIRNAME}/") or encoder_path == GEMMA_DIRNAME:
            if download_gemma_text_encoder(HF_TOKEN):
                downloaded_any = True
        else:
            log(f"Text encoder not found locally and repo unknown: {encoder_path}", level="WARN")

    if downloaded_any:
        global COMFY_RESTART_REQUIRED
        COMFY_RESTART_REQUIRED = True

def download_models(force: bool = False):
    """Download LTX-2 models to network volume"""
    if LTX2_DOWNLOAD_ALL:
        log("Downloading full LTX-2 asset set...", level="INFO")
        download_all_ltx2_assets(force=force)
    else:
        # LTX-2 (19B) main model - fp8 quantized (~20GB)
        model_path = f"{MODEL_DIR}/checkpoints/{LTX2_MODEL_FILENAME}"
        if not os.path.exists(model_path) or force:
            log(f"Downloading LTX-2 model ({LTX2_MODEL_FILENAME})...", level="INFO")
            try:
                url = f"https://huggingface.co/{LTX2_MODEL_REPO}/resolve/main/{LTX2_MODEL_FILENAME}"
                fast_download(url, model_path, HF_TOKEN)
            except Exception as e:
                log(f"aria2c failed ({e}), falling back to hf_hub_download...", level="WARN")
                hf_hub_download(
                    repo_id=LTX2_MODEL_REPO,
                    filename=LTX2_MODEL_FILENAME,
                    local_dir=f"{MODEL_DIR}/checkpoints",
                    token=HF_TOKEN or None,
                )
            log("LTX-2 model downloaded!", level="INFO")

        # Spatial Upscaler (required for high-res output)
        spatial_path = f"{MODEL_DIR}/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors"
        if not os.path.exists(spatial_path) or force:
            log("Downloading spatial upscaler...", level="INFO")
            try:
                url = f"https://huggingface.co/{LTX2_MODEL_REPO}/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors"
                fast_download(url, spatial_path, HF_TOKEN)
            except Exception as e:
                log(f"aria2c failed ({e}), falling back to hf_hub_download...", level="WARN")
                try:
                    hf_hub_download(
                        repo_id=LTX2_MODEL_REPO,
                        filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
                        local_dir=f"{MODEL_DIR}/latent_upscale_models",
                        token=HF_TOKEN or None,
                    )
                except Exception as e2:
                    log(f"Spatial upscaler download note: {e2}", level="WARN")

        # Temporal Upscaler (for more frames)
        temporal_path = f"{MODEL_DIR}/latent_upscale_models/ltx-2-temporal-upscaler-x2-1.0.safetensors"
        if not os.path.exists(temporal_path) or force:
            log("Downloading temporal upscaler...", level="INFO")
            try:
                url = f"https://huggingface.co/{LTX2_MODEL_REPO}/resolve/main/ltx-2-temporal-upscaler-x2-1.0.safetensors"
                fast_download(url, temporal_path, HF_TOKEN)
            except Exception as e:
                log(f"aria2c failed ({e}), falling back to hf_hub_download...", level="WARN")
                try:
                    hf_hub_download(
                        repo_id=LTX2_MODEL_REPO,
                        filename="ltx-2-temporal-upscaler-x2-1.0.safetensors",
                        local_dir=f"{MODEL_DIR}/latent_upscale_models",
                        token=HF_TOKEN or None,
                    )
                except Exception as e2:
                    log(f"Temporal upscaler download note: {e2}", level="WARN")

        # Distilled LoRA (for two-stage pipeline)
        lora_path = f"{MODEL_DIR}/loras/ltx-2-19b-distilled-lora-384.safetensors"
        if not os.path.exists(lora_path) or force:
            log("Downloading distilled LoRA...", level="INFO")
            try:
                url = f"https://huggingface.co/{LTX2_MODEL_REPO}/resolve/main/ltx-2-19b-distilled-lora-384.safetensors"
                fast_download(url, lora_path, HF_TOKEN)
            except Exception as e:
                log(f"aria2c failed ({e}), falling back to hf_hub_download...", level="WARN")
                try:
                    hf_hub_download(
                        repo_id=LTX2_MODEL_REPO,
                        filename="ltx-2-19b-distilled-lora-384.safetensors",
                        local_dir=f"{MODEL_DIR}/loras",
                        token=HF_TOKEN or None,
                    )
                except Exception as e2:
                    log(f"Distilled LoRA download note: {e2}", level="WARN")

    download_gemma_text_encoder(HF_TOKEN)

    # Download additional LoRAs from separate repos if enabled
    download_lora_repos(force=force)

    log("All LTX-2 models ready!", level="INFO")

def setup_environment(preload_models: bool = False):
    """Full setup on first run"""
    log("Setting up LTX-2 environment on network volume...", level="INFO")

    ensure_directories()
    install_comfyui()
    ensure_custom_nodes()  # Ensure LTX-Video nodes are installed
    ensure_comfy_model_links()
    if preload_models:
        download_models()

    log("Setup complete!", level="INFO")

# =============================================================================
# ComfyUI Helpers
# =============================================================================

def ensure_comfyui_running():
    """Start ComfyUI once per worker and wait for the API to respond."""
    global COMFY_PROCESS, COMFY_RESTART_REQUIRED

    if COMFY_PROCESS and COMFY_PROCESS.poll() is None:
        if COMFY_RESTART_REQUIRED:
            log("Restarting ComfyUI to apply updated model paths...", level="INFO")
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

    log(f"Starting ComfyUI: {' '.join(cmd)}", level="INFO")
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
                log(f"ComfyUI started successfully after {i} seconds", level="INFO")
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

def _coerce_image_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []

def _coerce_keyframes(value: Any) -> List[int]:
    if value is None:
        return []
    frames: List[int] = []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        for part in parts:
            try:
                frames.append(int(float(part)))
            except ValueError:
                continue
        return frames
    if isinstance(value, list):
        for item in value:
            try:
                frames.append(int(float(item)))
            except (TypeError, ValueError):
                continue
    return frames

def _normalize_keyframes(count: int, keyframes: List[int], num_frames: Optional[int]) -> List[int]:
    if count <= 0:
        return []
    if keyframes and len(keyframes) == count:
        if num_frames:
            min_idx = -(num_frames - 1)
            max_idx = num_frames - 1
            return [max(min(idx, max_idx), min_idx) for idx in keyframes]
        return keyframes

    if not num_frames:
        return list(range(count))

    if count == 1:
        return [0]

    max_idx = num_frames - 1
    return [
        min(max(round(i * max_idx / (count - 1)), 0), max_idx)
        for i in range(count)
    ]

def _next_node_id(workflow: Dict[str, Any], start: int = 0) -> int:
    ids = [int(key) for key in workflow.keys() if str(key).isdigit()]
    return max(ids or [start]) + 1

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
    input_images = _coerce_image_list(job_input.get("input_images"))
    input_keyframes = _coerce_keyframes(
        job_input.get("input_keyframes")
        or job_input.get("input_keyframe_indices")
    )
    if not input_images and input_image:
        input_images = [input_image]
    if input_images:
        input_image = input_images[0]
    has_load_image = False
    clip_nodes: Dict[str, Dict[str, Any]] = {}
    conditioning_nodes: List[Dict[str, Any]] = []
    conditioning_node_ids: List[str] = []
    load_image_nodes: List[str] = []
    img_to_video_nodes: List[tuple] = []
    concat_nodes: List[tuple] = []
    cfg_guiders: List[str] = []
    empty_latent_node_id: Optional[str] = None
    upscaled_latent_node_id: Optional[str] = None
    vae_ref = None
    noise_offset = 0
    use_keyframes = bool(input_keyframes) or len(input_images) > 1

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
            conditioning_node_ids.append(str(node_id))

        if class_type == "LoadImage":
            load_image_nodes.append(str(node_id))

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
            if input_image and not use_keyframes:
                inputs["image"] = download_input_image(input_image)

        if class_type == "LTXVImgToVideoInplace":
            if image_strength is not None:
                inputs["strength"] = float(image_strength)
            img_to_video_nodes.append((str(node_id), inputs))
            if vae_ref is None and inputs.get("vae"):
                vae_ref = inputs.get("vae")

        if class_type == "LTXVSampler":
            if steps is not None:
                inputs["steps"] = int(steps)
            if cfg is not None:
                inputs["cfg"] = float(cfg)
            if seed is not None:
                inputs["seed"] = int(seed)

        if class_type == "LTXVAudioGenerate" and prompt is not None:
            inputs["prompt"] = prompt

        if class_type == "LTXVConcatAVLatent":
            concat_nodes.append((str(node_id), inputs))

        if class_type == "CFGGuider":
            cfg_guiders.append(str(node_id))

        if class_type == "EmptyLTXVLatentVideo":
            empty_latent_node_id = str(node_id)

        if class_type == "LTXVLatentUpsampler":
            upscaled_latent_node_id = str(node_id)

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

    if use_keyframes and input_images:
        if not load_image_nodes:
            raise ValueError("Workflow is missing a LoadImage node required for keyframes.")
        if not conditioning_node_ids:
            raise ValueError("Workflow is missing LTXVConditioning required for keyframes.")
        if not empty_latent_node_id:
            raise ValueError("Workflow is missing EmptyLTXVLatentVideo required for keyframes.")
        if not vae_ref:
            raise ValueError("Workflow is missing VAE reference required for keyframes.")

        image_files = [download_input_image(value) for value in input_images]
        workflow[load_image_nodes[0]]["inputs"]["image"] = image_files[0]

        image_node_ids = [load_image_nodes[0]]
        next_id = _next_node_id(workflow)
        for image_file in image_files[1:]:
            node_id = str(next_id)
            next_id += 1
            workflow[node_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": image_file},
            }
            image_node_ids.append(node_id)

        effective_frames = num_frames
        if effective_frames is None and empty_latent_node_id:
            try:
                effective_frames = int(workflow[empty_latent_node_id]["inputs"].get("length"))
            except Exception:
                effective_frames = None

        keyframes = _normalize_keyframes(len(image_node_ids), input_keyframes, effective_frames)
        strength_value = float(image_strength) if image_strength is not None else 0.6

        def add_keyframe_chain(base_latent_ref):
            nonlocal next_id
            pos_ref = [conditioning_node_ids[0], 0]
            neg_ref = [conditioning_node_ids[0], 1]
            latent_ref = base_latent_ref
            for image_node_id, frame_idx in zip(image_node_ids, keyframes):
                node_id = str(next_id)
                next_id += 1
                workflow[node_id] = {
                    "class_type": "LTXVAddGuideAdvanced",
                    "inputs": {
                        "positive": pos_ref,
                        "negative": neg_ref,
                        "vae": vae_ref,
                        "latent": latent_ref,
                        "image": [image_node_id, 0],
                        "frame_idx": int(frame_idx),
                        "strength": strength_value,
                        "crf": 29,
                        "blur_radius": 0,
                        "interpolation": "lanczos",
                        "crop": "disabled",
                    },
                }
                pos_ref = [node_id, 0]
                neg_ref = [node_id, 1]
                latent_ref = [node_id, 2]
            return pos_ref, neg_ref, latent_ref

        stage1_pos = stage1_neg = stage1_latent = None
        stage2_pos = stage2_neg = stage2_latent = None

        stage1_pos, stage1_neg, stage1_latent = add_keyframe_chain([empty_latent_node_id, 0])
        if upscaled_latent_node_id:
            stage2_pos, stage2_neg, stage2_latent = add_keyframe_chain([upscaled_latent_node_id, 0])
        else:
            stage2_pos, stage2_neg, stage2_latent = stage1_pos, stage1_neg, stage1_latent

        cfg_sorted = sorted(cfg_guiders, key=lambda key: int(key) if str(key).isdigit() else key)
        if cfg_sorted and stage1_pos:
            workflow[cfg_sorted[0]]["inputs"]["positive"] = stage1_pos
            workflow[cfg_sorted[0]]["inputs"]["negative"] = stage1_neg
        if len(cfg_sorted) > 1 and stage2_pos:
            workflow[cfg_sorted[1]]["inputs"]["positive"] = stage2_pos
            workflow[cfg_sorted[1]]["inputs"]["negative"] = stage2_neg

        stage1_img_node = None
        stage2_img_node = None
        for node_id, inputs in img_to_video_nodes:
            latent_ref = inputs.get("latent")
            if isinstance(latent_ref, list) and empty_latent_node_id and latent_ref[0] == empty_latent_node_id:
                stage1_img_node = node_id
            elif isinstance(latent_ref, list) and upscaled_latent_node_id and latent_ref[0] == upscaled_latent_node_id:
                stage2_img_node = node_id

        updated = False
        for node_id, inputs in concat_nodes:
            video_ref = inputs.get("video_latent")
            if stage1_latent and stage1_img_node and isinstance(video_ref, list) and video_ref[0] == stage1_img_node:
                inputs["video_latent"] = stage1_latent
                updated = True
            elif stage2_latent and stage2_img_node and isinstance(video_ref, list) and video_ref[0] == stage2_img_node:
                inputs["video_latent"] = stage2_latent
                updated = True

        if not updated and concat_nodes and stage1_latent:
            concat_nodes_sorted = sorted(concat_nodes, key=lambda entry: int(entry[0]) if str(entry[0]).isdigit() else entry[0])
            concat_nodes_sorted[0][1]["video_latent"] = stage1_latent
            if len(concat_nodes_sorted) > 1 and stage2_latent:
                concat_nodes_sorted[1][1]["video_latent"] = stage2_latent

    if has_load_image and not input_images:
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
    log(f"ComfyUI prompt queued: {prompt_id}", level="INFO")
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
    last_queue = None
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
        if now - last_log >= STATUS_LOG_INTERVAL:
            elapsed = int(now - start)
            log(f"ComfyUI running... prompt={prompt_id} elapsed={elapsed}s", level="INFO")
            queue = get_queue_snapshot()
            if queue:
                running = len(queue.get("queue_running", []))
                pending = len(queue.get("queue_pending", []))
                snapshot = (running, pending)
                if snapshot != last_queue:
                    log(f"ComfyUI queue: running={running} pending={pending}", level="DEBUG")
                    last_queue = snapshot
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
    log(f"ComfyUI produced {len(files)} file(s)", level="INFO")
    for item in files:
        filename = item.get("filename", "unknown")
        out_type = item.get("type", "output")
        subfolder = item.get("subfolder", "")
        log(f"ComfyUI output: {filename} (type={out_type}, subfolder={subfolder})", level="DEBUG")
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
    log_job_summary(job_input)
    _log_ltx2_event("job_input", job_input, level="DEBUG")

    try:
        action = job_input.get("action", "generate")
        _log_ltx2_event("action", {"action": action}, level="DEBUG")

        setup_environment(preload_models=PRELOAD_MODELS and action != "sync_models")
        gpu = detect_gpu()
        log(f"GPU: {gpu['gpu_name']} ({gpu['vram_gb']}GB)", level="INFO")

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
            input_images = _coerce_image_list(job_input.get("input_images"))
            if input_images:
                input_image = input_images[0]
            requires_prompt = isinstance(workflow_input, str) and "image" not in str(workflow_input).lower()
            if requires_prompt and not prompt_text and not input_image:
                error_msg = "prompt is required for text-to-video workflows"
                _log_ltx2_event("validation_error", {"error": error_msg, "workflow": workflow_input}, level="WARN")
                return {"status": "error", "error": error_msg}

            # Optionally update workflow inputs if provided
            workflow_label = "<inline>" if isinstance(workflow_input, dict) else str(workflow_input)
            log(f"Workflow selected: {workflow_label}", level="INFO")
            _log_ltx2_event("workflow_selected", {"workflow": workflow_label}, level="DEBUG")
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
            log_workflow_summary(summary)
            _log_ltx2_event("workflow_summary", summary, level="DEBUG")

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

            _log_ltx2_event("generation_outputs", uploaded, level="DEBUG")

            return {
                "status": "success",
                "outputs": uploaded,
                "gpu": gpu,
                "execution_time": round(time.time() - start_time, 2),
            }

        error_result = {"status": "error", "error": f"Unknown action: {action}"}
        _log_ltx2_event("handler_error", error_result, level="WARN")
        return error_result

    except Exception as e:
        error_result = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
        _log_ltx2_event("handler_exception", {"error": str(e)}, level="ERROR")
        return error_result

if __name__ == "__main__":
    log("LTX-2 Worker Starting...", level="INFO")
    runpod.serverless.start({"handler": handler})
