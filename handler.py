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
import json
from pathlib import Path
from typing import Dict, Any, List

import runpod
import boto3
import requests
from huggingface_hub import hf_hub_download

# Enable fast HF downloads (must be set before any HF calls)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# =============================================================================
# Configuration
# =============================================================================

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")
COMFY_HOME = os.environ.get("COMFY_HOME", "/runpod-volume/ComfyUI")
TMPDIR = os.environ.get("TMPDIR", "/runpod-volume/tmp")
COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"

# HuggingFace
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))

# LTX-2 (19B) Model - The newest and best model
# Options: ltx-2-19b-distilled-fp8.safetensors (smaller), ltx-2-19b-dev-fp8.safetensors (larger)
LTX2_MODEL_REPO = os.environ.get("LTX2_MODEL_REPO", "Lightricks/LTX-Video-0.9.7")
LTX2_MODEL_FILENAME = os.environ.get("LTX2_MODEL_FILENAME", "ltx-2-19b-distilled-fp8.safetensors")

# S3 Storage
S3_BUCKET = os.environ.get("S3_BUCKET", "ltx2-models")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "")
S3_REGION = os.environ.get("S3_REGION", "eu-north-1")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
PRESIGNED_URL_EXPIRY = int(os.environ.get("PRESIGNED_URL_EXPIRY", 14400))

COMFY_PROCESS = None

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
    comfy_models = f"{COMFY_HOME}/models"
    if os.path.islink(comfy_models):
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
    for subdir in ["checkpoints", "text_encoders", "vae", "clip"]:
        src = os.path.join(MODEL_DIR, subdir)
        dst = os.path.join(comfy_models, subdir)
        if os.path.exists(dst) or os.path.islink(dst):
            continue
        try:
            os.symlink(src, dst)
        except OSError as e:
            print(f"Symlink for {subdir} failed: {e}")

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
            print(f"Spatial upscaler download note: {e}")

    # Temporal Upscaler (for more frames)
    temporal_path = f"{MODEL_DIR}/latent_upscale_models/ltx-2-temporal-upscaler-x2-1.0.safetensors"
    if not os.path.exists(temporal_path) or force:
        print("Downloading temporal upscaler...")
        try:
            url = f"https://huggingface.co/{LTX2_MODEL_REPO}/resolve/main/ltx-2-temporal-upscaler-x2-1.0.safetensors"
            fast_download(url, temporal_path, HF_TOKEN)
        except Exception as e:
            print(f"Temporal upscaler download note: {e}")

    # Distilled LoRA (for two-stage pipeline)
    lora_path = f"{MODEL_DIR}/loras/ltx-2-19b-distilled-lora-384.safetensors"
    if not os.path.exists(lora_path) or force:
        print("Downloading distilled LoRA...")
        try:
            url = f"https://huggingface.co/{LTX2_MODEL_REPO}/resolve/main/ltx-2-19b-distilled-lora-384.safetensors"
            fast_download(url, lora_path, HF_TOKEN)
        except Exception as e:
            print(f"Distilled LoRA download note: {e}")

    print("All LTX-2 models ready!")

def setup_environment():
    """Full setup on first run"""
    print("=" * 50)
    print("Setting up LTX-2 environment on network volume...")
    print("=" * 50)

    ensure_directories()
    install_comfyui()
    ensure_custom_nodes()  # Ensure LTX-Video nodes are installed
    ensure_comfy_model_links()
    download_models()

    print("Setup complete!")

# =============================================================================
# ComfyUI Helpers
# =============================================================================

def ensure_comfyui_running():
    """Start ComfyUI once per worker and wait for the API to respond."""
    global COMFY_PROCESS

    if COMFY_PROCESS and COMFY_PROCESS.poll() is None:
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

    if fps is None:
        fps = 24
    if num_frames is None and duration_seconds:
        num_frames = int(float(duration_seconds) * float(fps)) + 1

    for node in workflow.values():
        class_type = node.get("class_type")
        inputs = node.get("inputs", {})

        if class_type == "LTXVTextEncode":
            if prompt is not None:
                inputs["prompt"] = prompt
            if negative_prompt is not None:
                inputs["negative_prompt"] = negative_prompt

        if class_type == "LTXVAudioGenerate" and prompt is not None:
            inputs["prompt"] = prompt

        if class_type == "EmptyLTXVLatentVideo":
            if width is not None:
                inputs["width"] = int(width)
            if height is not None:
                inputs["height"] = int(height)
            if num_frames is not None:
                inputs["length"] = int(num_frames)

        if class_type == "LTXVSampler":
            if steps is not None:
                inputs["steps"] = int(steps)
            if cfg is not None:
                inputs["cfg"] = float(cfg)
            if seed is not None:
                inputs["seed"] = int(seed)

        if class_type == "VHS_VideoCombine":
            if fps is not None:
                inputs["fps"] = int(fps)

        if class_type == "LoadImage":
            input_image = job_input.get("input_image")
            if input_image:
                inputs["image"] = download_input_image(input_image)

    return workflow

def queue_prompt(prompt: Dict[str, Any]) -> str:
    payload = {"prompt": prompt, "client_id": "runpod-ltx2"}
    resp = requests.post(f"{COMFY_URL}/prompt", json=payload, timeout=10)
    if resp.status_code != 200:
        error_detail = resp.text[:2000]
        raise RuntimeError(f"ComfyUI prompt rejected ({resp.status_code}): {error_detail}")
    data = resp.json()
    return data["prompt_id"]

def wait_for_prompt(prompt_id: str, timeout_seconds: int = 1800) -> Dict[str, Any]:
    start = time.time()
    while time.time() - start < timeout_seconds:
        resp = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if prompt_id in data:
                entry = data[prompt_id]
                outputs = entry.get("outputs")
                if outputs:
                    return entry
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

    try:
        setup_environment()
        gpu = detect_gpu()
        print(f"GPU: {gpu['gpu_name']} ({gpu['vram_gb']}GB)")

        action = job_input.get("action", "generate")

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

            # Optionally update workflow inputs if provided
            if any(k in job_input for k in ["prompt", "width", "height", "num_frames", "fps", "steps", "cfg_scale", "seed"]):
                workflow = update_workflow_inputs(workflow, job_input)

            outputs = run_comfyui_workflow(workflow)

            uploaded = []
            for item in outputs:
                path = resolve_output_path(item)
                if path.exists():
                    content_type = "video/mp4"
                    if path.suffix.lower() == ".gif":
                        content_type = "image/gif"
                    uploaded.append(upload_to_s3(str(path), content_type=content_type))

            return {
                "status": "success",
                "outputs": uploaded,
                "gpu": gpu,
                "execution_time": round(time.time() - start_time, 2),
            }

        return {"status": "error", "error": f"Unknown action: {action}"}

    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    print("LTX-2 Worker Starting...")
    runpod.serverless.start({"handler": handler})
