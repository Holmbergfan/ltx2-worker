#!/usr/bin/env python3
"""
LTX-Video Simple Handler for RunPod Serverless
Uses diffusers pipeline directly - no ComfyUI dependency

This is a simpler alternative to the full ComfyUI-based handler.
"""

import os
import sys
import time
import uuid
import traceback
from pathlib import Path
from typing import Dict, Any

import runpod
import boto3
import torch

# Configuration
MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")
TMPDIR = os.environ.get("TMPDIR", "/runpod-volume/tmp")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/runpod-volume/outputs")

# S3 Storage
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "")
S3_REGION = os.environ.get("S3_REGION", "eu-north-1")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
PRESIGNED_URL_EXPIRY = int(os.environ.get("PRESIGNED_URL_EXPIRY", 14400))

# HuggingFace
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Global pipeline instance
PIPELINE = None


def ensure_directories():
    """Create required directories"""
    for d in [MODEL_DIR, TMPDIR, OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)


def load_pipeline():
    """Load LTX-Video pipeline"""
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    print("Loading LTX-Video pipeline...")
    from diffusers import LTXPipeline

    PIPELINE = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch.bfloat16,
        cache_dir=f"{MODEL_DIR}/.cache",
        token=HF_TOKEN or None,
    )
    PIPELINE.to("cuda")

    # Enable memory optimizations
    PIPELINE.enable_model_cpu_offload()

    print("Pipeline loaded!")
    return PIPELINE


def generate_video(
    prompt: str,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 512,
    num_frames: int = 49,
    num_inference_steps: int = 25,
    guidance_scale: float = 3.0,
    seed: int = None,
) -> str:
    """Generate video from text prompt"""
    pipe = load_pipeline()

    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)

    print(f"Generating video: {prompt[:50]}...")
    print(f"  Resolution: {width}x{height}, Frames: {num_frames}, Steps: {num_inference_steps}")

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    # Save video
    video = output.frames[0]
    filename = f"ltxv_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)

    from diffusers.utils import export_to_video
    export_to_video(video, output_path, fps=24)

    print(f"Video saved: {output_path}")
    return output_path


def upload_to_s3(file_path: str, content_type: str = "video/mp4") -> Dict[str, Any]:
    """Upload to S3 and return presigned URL"""
    if not AWS_ACCESS_KEY_ID or not S3_BUCKET:
        # Return base64 if no S3 configured
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

    key = f"ltxv/{time.strftime('%Y%m%d-%H%M%S')}/{os.path.basename(file_path)}"
    s3.upload_file(file_path, S3_BUCKET, key, ExtraArgs={"ContentType": content_type})

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=PRESIGNED_URL_EXPIRY,
    )

    return {"type": "s3_url", "url": url, "key": key, "filename": os.path.basename(file_path)}


def detect_gpu() -> Dict[str, Any]:
    """Detect GPU info"""
    try:
        if torch.cuda.is_available():
            return {
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1),
            }
    except:
        pass
    return {"gpu_name": "Unknown", "vram_gb": 0}


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod serverless handler"""
    start_time = time.time()
    job_input = event.get("input", {})

    try:
        ensure_directories()
        gpu = detect_gpu()
        print(f"GPU: {gpu['gpu_name']} ({gpu['vram_gb']}GB)")

        action = job_input.get("action", "generate")

        if action == "status":
            return {
                "status": "success",
                "gpu": gpu,
                "pipeline_loaded": PIPELINE is not None,
            }

        if action == "generate":
            prompt = job_input.get("prompt")
            if not prompt:
                return {"status": "error", "error": "prompt is required"}

            video_path = generate_video(
                prompt=prompt,
                negative_prompt=job_input.get("negative_prompt", ""),
                width=job_input.get("width", 768),
                height=job_input.get("height", 512),
                num_frames=job_input.get("num_frames", 49),
                num_inference_steps=job_input.get("steps", 25),
                guidance_scale=job_input.get("cfg_scale", 3.0),
                seed=job_input.get("seed"),
            )

            result = upload_to_s3(video_path)

            return {
                "status": "success",
                "output": result,
                "gpu": gpu,
                "execution_time": round(time.time() - start_time, 2),
            }

        return {"status": "error", "error": f"Unknown action: {action}"}

    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    print("LTX-Video Simple Worker Starting...")
    runpod.serverless.start({"handler": handler})
