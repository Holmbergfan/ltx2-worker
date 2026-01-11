# LTX-2 Video + Audio Generation Worker

RunPod serverless worker for LTX-2 video generation with synchronized audio.

## Features

- **Text-to-Video**: Generate videos from text prompts
- **Image-to-Video**: Animate static images
- **Audio Generation**: Synchronized audio (dialogue, music, ambient sounds)
- **Multiple Models**: Support for distilled (fast) and full precision variants
- **GPU Adaptive**: Automatically configures based on available VRAM

## Model Variants

| Model | VRAM | Steps | Quality | Speed |
|-------|------|-------|---------|-------|
| `distilled-fp8` | 24GB+ | 8 | Good | Fastest |
| `dev-fp8` | 32GB+ | 25 | Better | Standard |
| `distilled` | 48GB+ | 8 | Good | Fast |
| `dev` | 48GB+ | 25 | Best | Slower |

## Deployment Steps

### 1. Get RunPod S3 Access Keys

1. Go to [RunPod Console](https://runpod.io/console/storage)
2. Click on your bucket `07mhk8ul6o`
3. Go to **Access Keys** tab
4. Create new access key or copy existing
5. Update `.env` with `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

### 2. Build Docker Image

```bash
cd /root/comfyui/runpod-ltx2-worker

# Build locally
docker build -t ltx2-worker:latest .

# Or build and push to Docker Hub
docker build -t yourusername/ltx2-worker:latest .
docker push yourusername/ltx2-worker:latest
```

### 3. Create RunPod Network Volume

1. Go to [RunPod Console](https://runpod.io/console/storage)
2. Click **+ Network Volume**
3. Settings:
   - **Name**: `ltx2-models`
   - **Region**: EU-RO-1 (same as S3)
   - **Size**: 100GB
4. Note the volume ID

### 4. Create Serverless Endpoint

1. Go to [RunPod Serverless](https://runpod.io/console/serverless)
2. Click **+ New Endpoint**
3. Settings:
   - **Name**: `LTX-2 Video Generator`
   - **Docker Image**: `yourusername/ltx2-worker:latest`
   - **GPU**: 48GB Pro (A6000) recommended
   - **Idle Timeout**: 5 seconds
   - **Max Workers**: 1-2
   - **Network Volume**: Select `ltx2-models`, mount at `/runpod-volume`
4. Add environment variables from `.env`
5. Deploy!

### 5. First Run (Model Download)

First request will download models (~30GB). Use the sync action:

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "sync_models",
      "model": "distilled-fp8"
    }
  }'
```

## API Usage

### Generate Video (Text-to-Video)

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "generate",
      "workflow": "text-to-video-audio",
      "prompt": "A golden retriever running through a field of sunflowers, slow motion, cinematic",
      "negative_prompt": "blurry, low quality",
      "width": 768,
      "height": 512,
      "duration_seconds": 4,
      "fps": 24,
      "with_audio": true,
      "model": "distilled-fp8"
    }
  }'
```

### Generate Video (Image-to-Video)

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "action": "generate",
      "workflow": "image-to-video",
      "prompt": "The subject comes to life with gentle movement",
      "input_image": "https://example.com/image.png",
      "duration_seconds": 4,
      "with_audio": false
    }
  }'
```

### Check Job Status

```bash
curl "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### List Available Models

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "list_models"}}'
```

## Response Format

```json
{
  "status": "success",
  "outputs": [
    {
      "type": "s3_url",
      "url": "https://s3api-eu-ro-1.runpod.io/...",
      "filename": "ltx2_output_00001.mp4",
      "expires_in": 3600
    }
  ],
  "parameters": {
    "prompt": "...",
    "width": 768,
    "height": 512,
    "num_frames": 97,
    "duration_seconds": 4,
    "seed": 12345
  },
  "execution_time_seconds": 45.2,
  "gpu_config": {
    "gpu_name": "NVIDIA A6000",
    "vram_gb": 48.0
  }
}
```

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action` | string | `generate` | Action type |
| `prompt` | string | required | Text prompt |
| `negative_prompt` | string | `""` | Negative prompt |
| `width` | int | 768 | Video width |
| `height` | int | 512 | Video height |
| `duration_seconds` | int | 4 | Video duration (1-10) |
| `fps` | int | 24 | Frames per second |
| `steps` | int | 8 | Sampling steps |
| `cfg_scale` | float | 3.0 | Classifier-free guidance |
| `seed` | int | -1 | Random seed (-1 = random) |
| `model` | string | `distilled-fp8` | Model variant |
| `with_audio` | bool | true | Generate audio |
| `input_image` | string | null | Image URL for I2V |
| `workflow` | string/object | `text-to-video-audio` | Built-in workflow name or full ComfyUI API prompt |

## Example Prompts

**Cinematic:**
```
A golden retriever running through a field of sunflowers, slow motion, cinematic lighting, 4K quality
```

**Sci-Fi:**
```
A futuristic city at night with flying cars and neon lights, rain falling, cyberpunk atmosphere, electronic music
```

**Nature:**
```
Ocean waves crashing on a rocky beach at sunset, drone shot pulling back, nature sounds, peaceful
```

**Food:**
```
A chef preparing sushi in a traditional Japanese kitchen, close-up shots, ambient kitchen sounds, ASMR style
```

**Space:**
```
An astronaut floating in space with Earth in the background, stars twinkling, peaceful ambient music
```

## Costs

| GPU | Cost/sec | 4-sec video (720p) |
|-----|----------|-------------------|
| 48GB Pro (A6000) | $0.00076 | ~$0.15 |
| 48GB (A40) | $0.00069 | ~$0.12 |
| 24GB Pro (4090) | $0.00044 | ~$0.20 (slower) |

## Troubleshooting

**Model download stuck:**
```bash
# Force re-download
{"input": {"action": "sync_models", "force": true}}
```

**Out of VRAM:**
- Use `distilled-fp8` model
- Reduce resolution to 640x384
- Reduce `duration_seconds` to 2-3

**No audio generated:**
- Ensure `with_audio: true`
- Audio adds ~2GB VRAM usage
- May need 32GB+ GPU for audio

## Files

```
runpod-ltx2-worker/
├── Dockerfile           # Container build
├── handler.py           # Main worker logic
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
├── .env.example         # Template
├── README.md            # This file
└── workflows/
    ├── text-to-video-audio.json
    └── image-to-video.json
```
