#!/usr/bin/env python3
"""
LTX-2 Video + Audio Generation Worker for RunPod Serverless
Generates videos with synchronized audio using the LTX-2 model via ComfyUI

Architecture follows the patterns from vace-web and aura workers:
- Models downloaded to network volume on first run
- Outputs uploaded to S3 with presigned URLs
- GPU-adaptive configuration
"""

import os
import sys
import json
import time
import uuid
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

import runpod
import boto3
import requests
from botocore.config import Config as BotoConfig
from huggingface_hub import hf_hub_download, snapshot_download

# =============================================================================
# Configuration
# =============================================================================

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")
COMFY_HOME = os.environ.get("COMFY_HOME", "/opt/ComfyUI")
WORKFLOWS_DIR = os.environ.get("WORKFLOWS_DIR", "/opt/workflows")
TMPDIR = os.environ.get("TMPDIR", "/runpod-volume/tmp")

# HuggingFace
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))

# S3 Storage (RunPod S3)
S3_BUCKET = os.environ.get("S3_BUCKET", "07mhk8ul6o")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "https://s3api-eu-ro-1.runpod.io")
S3_REGION = os.environ.get("S3_REGION", "eu-ro-1")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

# Presigned URL expiry (seconds)
PRESIGNED_URL_EXPIRY = int(os.environ.get("PRESIGNED_URL_EXPIRY", 3600))  # 1 hour

# =============================================================================
# LTX-2 Model Definitions
# =============================================================================

LTX2_MODELS = {
    "distilled-fp8": {
        "name": "LTX-2 19B Distilled FP8",
        "repo": "Lightricks/LTX-Video-2",
        "file": "ltx-2-19b-distilled-fp8.safetensors",
        "subdir": "diffusion_models",
        "min_vram": 24,
        "default_steps": 8,
        "description": "Fastest - 8 steps, good quality"
    },
    "dev-fp8": {
        "name": "LTX-2 19B Dev FP8",
        "repo": "Lightricks/LTX-Video-2",
        "file": "ltx-2-19b-dev-fp8.safetensors",
        "subdir": "diffusion_models",
        "min_vram": 32,
        "default_steps": 25,
        "description": "Standard - 25 steps, higher quality"
    },
    "distilled": {
        "name": "LTX-2 19B Distilled",
        "repo": "Lightricks/LTX-Video-2",
        "file": "ltx-2-19b-distilled.safetensors",
        "subdir": "diffusion_models",
        "min_vram": 48,
        "default_steps": 8,
        "description": "Full precision distilled"
    },
    "dev": {
        "name": "LTX-2 19B Dev",
        "repo": "Lightricks/LTX-Video-2",
        "file": "ltx-2-19b-dev.safetensors",
        "subdir": "diffusion_models",
        "min_vram": 48,
        "default_steps": 25,
        "description": "Full precision dev - best quality"
    }
}

SUPPORTING_MODELS = {
    "vae": {
        "repo": "Lightricks/LTX-Video-2",
        "file": "ltx-video-2b-v0.9.5-vae.safetensors",
        "subdir": "vae",
    },
    "text_encoder": {
        "repo": "google/gemma-2-2b-it",
        "subdir": "text_encoders/gemma-2-2b-it",
        "is_full_repo": True,
    },
    "spatial_upscaler": {
        "repo": "Lightricks/LTX-Video-2",
        "file": "ltx2-spatial-upscaler-0.9.7.safetensors",
        "subdir": "upscale_models",
        "optional": True,
    },
    "temporal_upscaler": {
        "repo": "Lightricks/LTX-Video-2",
        "file": "ltx2-temporal-upscaler-0.9.7.safetensors",
        "subdir": "upscale_models",
        "optional": True,
    },
}

# =============================================================================
# GPU Detection & Configuration
# =============================================================================

def detect_gpu() -> Dict[str, Any]:
    """Detect GPU and return optimal configuration"""
    import torch

    if not torch.cuda.is_available():
        return {"error": "No CUDA GPU available"}

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # GPU-specific optimizations
    config = {
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 1),
        "dtype": "float16",
        "offload": False,
        "compile": False,
        "recommended_model": "distilled-fp8",
    }

    if vram_gb >= 48:
        config.update({
            "dtype": "bfloat16",
            "offload": False,
            "compile": True,
            "recommended_model": "distilled-fp8",  # Still use FP8 for speed
        })
    elif vram_gb >= 32:
        config.update({
            "dtype": "float16",
            "offload": False,
            "compile": True,
            "recommended_model": "distilled-fp8",
        })
    elif vram_gb >= 24:
        config.update({
            "dtype": "float16",
            "offload": True,
            "compile": False,
            "recommended_model": "distilled-fp8",
        })
    else:
        config.update({
            "dtype": "float16",
            "offload": True,
            "compile": False,
            "recommended_model": "distilled-fp8",
            "warning": "Low VRAM - may need to reduce resolution",
        })

    print(f"GPU Detected: {gpu_name} ({vram_gb:.1f}GB VRAM)")
    print(f"Configuration: dtype={config['dtype']}, offload={config['offload']}")

    return config

# =============================================================================
# Model Management
# =============================================================================

def ensure_directories():
    """Create necessary directories"""
    dirs = [
        f"{MODEL_DIR}/diffusion_models",
        f"{MODEL_DIR}/text_encoders",
        f"{MODEL_DIR}/vae",
        f"{MODEL_DIR}/upscale_models",
        f"{MODEL_DIR}/loras",
        f"{MODEL_DIR}/.cache/hf",
        TMPDIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def download_model(model_key: str, force: bool = False) -> str:
    """Download a specific LTX-2 model variant"""
    if model_key not in LTX2_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(LTX2_MODELS.keys())}")

    model_info = LTX2_MODELS[model_key]
    target_dir = f"{MODEL_DIR}/{model_info['subdir']}"
    target_path = f"{target_dir}/{model_info['file']}"

    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(target_path) and not force:
        print(f"Model already exists: {target_path}")
        return target_path

    print(f"Downloading {model_info['name']}...")
    start_time = time.time()

    hf_hub_download(
        repo_id=model_info["repo"],
        filename=model_info["file"],
        local_dir=target_dir,
        token=HF_TOKEN if HF_TOKEN else None,
        resume_download=True,
    )

    elapsed = time.time() - start_time
    print(f"Downloaded {model_info['file']} in {elapsed:.1f}s")

    return target_path

def download_supporting_models(force: bool = False) -> Dict[str, str]:
    """Download VAE, text encoder, and optional upscalers"""
    paths = {}

    for name, info in SUPPORTING_MODELS.items():
        target_dir = f"{MODEL_DIR}/{info['subdir']}"
        os.makedirs(target_dir, exist_ok=True)

        if info.get("is_full_repo"):
            # Download entire repository (for text encoders)
            if os.path.exists(target_dir) and os.listdir(target_dir) and not force:
                print(f"Text encoder already exists: {target_dir}")
                paths[name] = target_dir
                continue

            print(f"Downloading {name} from {info['repo']}...")
            snapshot_download(
                repo_id=info["repo"],
                local_dir=target_dir,
                token=HF_TOKEN if HF_TOKEN else None,
                resume_download=True,
            )
            paths[name] = target_dir
        else:
            # Download single file
            target_path = f"{target_dir}/{info['file']}"

            if os.path.exists(target_path) and not force:
                print(f"{name} already exists: {target_path}")
                paths[name] = target_path
                continue

            if info.get("optional"):
                try:
                    print(f"Downloading optional {name}...")
                    hf_hub_download(
                        repo_id=info["repo"],
                        filename=info["file"],
                        local_dir=target_dir,
                        token=HF_TOKEN if HF_TOKEN else None,
                    )
                    paths[name] = target_path
                except Exception as e:
                    print(f"Optional {name} download failed: {e}")
            else:
                print(f"Downloading {name}...")
                hf_hub_download(
                    repo_id=info["repo"],
                    filename=info["file"],
                    local_dir=target_dir,
                    token=HF_TOKEN if HF_TOKEN else None,
                )
                paths[name] = target_path

    return paths

def ensure_models(model_key: str = "distilled-fp8", force: bool = False) -> Dict[str, str]:
    """Ensure all required models are downloaded"""
    ensure_directories()

    paths = {}

    # Download main model
    paths["model"] = download_model(model_key, force)

    # Download supporting models
    supporting = download_supporting_models(force)
    paths.update(supporting)

    print("All models ready!")
    return paths

# =============================================================================
# S3 Upload
# =============================================================================

def get_s3_client():
    """Get boto3 S3 client configured for RunPod S3"""
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        region_name=S3_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path"}
        )
    )

def upload_to_s3(file_path: str, content_type: str = "video/mp4") -> Dict[str, str]:
    """Upload file to S3 and return presigned URL"""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        # Return base64 if no S3 configured
        import base64
        with open(file_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode()
        return {
            "type": "base64",
            "data": b64_data,
            "content_type": content_type,
            "filename": os.path.basename(file_path),
        }

    s3 = get_s3_client()

    # Generate unique key
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = os.path.basename(file_path)
    key = f"ltx2/{timestamp}/{unique_id}/{filename}"

    # Upload
    print(f"Uploading to S3: {key}")
    s3.upload_file(
        file_path,
        S3_BUCKET,
        key,
        ExtraArgs={"ContentType": content_type}
    )

    # Generate presigned URL
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=PRESIGNED_URL_EXPIRY,
    )

    return {
        "type": "s3_url",
        "url": url,
        "key": key,
        "bucket": S3_BUCKET,
        "filename": filename,
        "expires_in": PRESIGNED_URL_EXPIRY,
    }

# =============================================================================
# Image/Video Download Helpers
# =============================================================================

def download_input_file(url_or_data: str, suffix: str = ".png") -> str:
    """Download input file from URL or decode from base64"""
    import base64

    temp_path = f"{TMPDIR}/{uuid.uuid4()}{suffix}"

    if url_or_data.startswith("data:"):
        # Base64 data URL
        header, data = url_or_data.split(",", 1)
        with open(temp_path, "wb") as f:
            f.write(base64.b64decode(data))
    elif url_or_data.startswith(("http://", "https://")):
        # HTTP URL
        response = requests.get(url_or_data, timeout=60)
        response.raise_for_status()
        with open(temp_path, "wb") as f:
            f.write(response.content)
    else:
        # Assume it's a file path
        if os.path.exists(url_or_data):
            shutil.copy(url_or_data, temp_path)
        else:
            raise ValueError(f"Invalid input: {url_or_data[:100]}...")

    return temp_path

# =============================================================================
# ComfyUI Workflow Execution
# =============================================================================

def build_ltx2_workflow(
    prompt: str,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 512,
    num_frames: int = 97,
    fps: int = 24,
    steps: int = 8,
    cfg_scale: float = 3.0,
    seed: int = -1,
    model_file: str = "ltx-2-19b-distilled-fp8.safetensors",
    with_audio: bool = True,
    input_image: Optional[str] = None,
) -> Dict[str, Any]:
    """Build LTX-2 ComfyUI workflow JSON"""

    if seed == -1:
        seed = int(time.time() * 1000) % (2**32)

    workflow = {
        # Load LTX-2 model
        "1": {
            "class_type": "LTXVLoader",
            "inputs": {
                "ckpt_name": model_file,
            }
        },
        # Text encoding
        "2": {
            "class_type": "LTXVTextEncode",
            "inputs": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "ltxv_model": ["1", 0],
            }
        },
        # Empty latent or from image
        "3": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
            }
        },
        # Sampler
        "4": {
            "class_type": "LTXVSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "ltxv_model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["2", 1],
                "latent": ["3", 0],
            }
        },
        # Decode
        "5": {
            "class_type": "LTXVDecode",
            "inputs": {
                "samples": ["4", 0],
                "ltxv_model": ["1", 0],
            }
        },
    }

    # Add image-to-video support if input image provided
    if input_image:
        workflow["10"] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": input_image,
            }
        }
        workflow["3"]["class_type"] = "LTXVImageToLatent"
        workflow["3"]["inputs"] = {
            "image": ["10", 0],
            "length": num_frames,
            "ltxv_model": ["1", 0],
        }

    # Add audio generation if requested
    if with_audio:
        workflow["6"] = {
            "class_type": "LTXVAudioGenerate",
            "inputs": {
                "prompt": prompt,
                "video": ["5", 0],
                "ltxv_model": ["1", 0],
            }
        }
        workflow["7"] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["5", 0],
                "audio": ["6", 0],
                "fps": fps,
                "filename_prefix": "ltx2_output",
                "format": "video/h264-mp4",
            }
        }
    else:
        workflow["6"] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["5", 0],
                "fps": fps,
                "filename_prefix": "ltx2_output",
                "format": "video/h264-mp4",
            }
        }

    return workflow

def execute_comfyui_workflow(workflow: Dict[str, Any]) -> List[str]:
    """Execute ComfyUI workflow and return output file paths"""

    # Add ComfyUI to path
    sys.path.insert(0, COMFY_HOME)
    os.chdir(COMFY_HOME)

    # Set up output directory
    output_dir = f"{TMPDIR}/comfyui_output_{uuid.uuid4()}"
    os.makedirs(output_dir, exist_ok=True)

    # Configure ComfyUI output
    os.environ["COMFYUI_OUTPUT_DIR"] = output_dir

    try:
        # Import ComfyUI modules
        import folder_paths
        folder_paths.set_output_directory(output_dir)

        import execution
        import nodes

        # Validate workflow
        valid = execution.validate_prompt(workflow)
        if valid[0] is not True:
            raise ValueError(f"Invalid workflow: {valid[1]}")

        # Execute
        prompt_id = str(uuid.uuid4())
        executor = execution.PromptExecutor(None)
        executor.execute(workflow, prompt_id, {}, [])

        # Collect outputs
        output_files = []
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith((".mp4", ".webm", ".gif", ".png", ".jpg")):
                    output_files.append(os.path.join(root, f))

        return output_files

    except Exception as e:
        print(f"ComfyUI execution error: {e}")
        traceback.print_exc()
        raise

# =============================================================================
# RunPod Handler
# =============================================================================

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod serverless handler"""

    start_time = time.time()
    job_input = event.get("input", {})

    try:
        # Detect GPU and get config
        gpu_config = detect_gpu()
        if "error" in gpu_config:
            return {"status": "error", "error": gpu_config["error"]}

        # Get action type
        action = job_input.get("action", "generate")

        # =================================================================
        # Action: sync_models - Force download/update models
        # =================================================================
        if action == "sync_models":
            model_key = job_input.get("model", "distilled-fp8")
            force = job_input.get("force", False)
            paths = ensure_models(model_key, force)
            return {
                "status": "success",
                "action": "sync_models",
                "model_paths": paths,
                "gpu_config": gpu_config,
            }

        # =================================================================
        # Action: list_models - List available models
        # =================================================================
        if action == "list_models":
            return {
                "status": "success",
                "action": "list_models",
                "available_models": {
                    k: {
                        "name": v["name"],
                        "description": v["description"],
                        "min_vram": v["min_vram"],
                        "default_steps": v["default_steps"],
                    }
                    for k, v in LTX2_MODELS.items()
                },
                "gpu_config": gpu_config,
            }

        # =================================================================
        # Action: generate - Generate video
        # =================================================================
        if action == "generate":
            # Parse parameters
            prompt = job_input.get("prompt", "A cat walking in a garden, cinematic")
            negative_prompt = job_input.get("negative_prompt", "blurry, low quality, distorted")
            width = job_input.get("width", 768)
            height = job_input.get("height", 512)

            # Duration in seconds -> frames (24 fps)
            duration_seconds = job_input.get("duration_seconds", 4)
            fps = job_input.get("fps", 24)
            num_frames = int(duration_seconds * fps) + 1  # +1 for inclusive
            num_frames = min(num_frames, 257)  # Max ~10 seconds

            # Generation settings
            model_key = job_input.get("model", gpu_config["recommended_model"])
            steps = job_input.get("steps", LTX2_MODELS.get(model_key, {}).get("default_steps", 8))
            cfg_scale = job_input.get("cfg_scale", 3.0)
            seed = job_input.get("seed", -1)
            with_audio = job_input.get("with_audio", True)

            # Optional input image for image-to-video
            input_image_url = job_input.get("input_image")
            input_image_path = None
            if input_image_url:
                input_image_path = download_input_file(input_image_url, ".png")

            # Ensure models are downloaded
            model_paths = ensure_models(model_key)
            model_file = LTX2_MODELS[model_key]["file"]

            print(f"Generating video: {width}x{height}, {num_frames} frames, {steps} steps")
            print(f"Model: {model_file}")
            print(f"Prompt: {prompt[:100]}...")

            # Build and execute workflow
            if "workflow" in job_input:
                # Use custom workflow if provided
                workflow = job_input["workflow"]
            else:
                workflow = build_ltx2_workflow(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=fps,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    seed=seed,
                    model_file=model_file,
                    with_audio=with_audio,
                    input_image=input_image_path,
                )

            # Execute
            output_files = execute_comfyui_workflow(workflow)

            if not output_files:
                return {
                    "status": "error",
                    "error": "No output files generated",
                }

            # Upload outputs
            outputs = []
            for file_path in output_files:
                ext = os.path.splitext(file_path)[1].lower()
                content_type = {
                    ".mp4": "video/mp4",
                    ".webm": "video/webm",
                    ".gif": "image/gif",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                }.get(ext, "application/octet-stream")

                result = upload_to_s3(file_path, content_type)
                outputs.append(result)

            # Clean up temp files
            if input_image_path and os.path.exists(input_image_path):
                os.remove(input_image_path)

            execution_time = time.time() - start_time

            return {
                "status": "success",
                "outputs": outputs,
                "parameters": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_frames": num_frames,
                    "duration_seconds": duration_seconds,
                    "fps": fps,
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "seed": seed,
                    "model": model_key,
                    "with_audio": with_audio,
                },
                "execution_time_seconds": round(execution_time, 2),
                "gpu_config": gpu_config,
            }

        # Unknown action
        return {
            "status": "error",
            "error": f"Unknown action: {action}",
            "available_actions": ["generate", "sync_models", "list_models"],
        }

    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "execution_time_seconds": round(execution_time, 2),
        }

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("Starting LTX-2 Video Worker...")
    print(f"Model directory: {MODEL_DIR}")
    print(f"ComfyUI home: {COMFY_HOME}")
    print(f"S3 bucket: {S3_BUCKET}")

    runpod.serverless.start({"handler": handler})
