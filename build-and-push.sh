#!/bin/bash
set -e

# LTX-2 Worker Build & Push Script
# Usage: ./build-and-push.sh [dockerhub_username]

DOCKERHUB_USER="${1:-holmbergpro}"
IMAGE_NAME="ltx2-worker"
VERSION="1.0.0"

echo "=========================================="
echo "Building LTX-2 Worker Docker Image"
echo "=========================================="

cd "$(dirname "$0")"

# Build the image
echo "Building image..."
docker build -t ${IMAGE_NAME}:${VERSION} -t ${IMAGE_NAME}:latest .

# Tag for Docker Hub
echo "Tagging for Docker Hub..."
docker tag ${IMAGE_NAME}:${VERSION} ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}
docker tag ${IMAGE_NAME}:latest ${DOCKERHUB_USER}/${IMAGE_NAME}:latest

# Push to Docker Hub
echo "Pushing to Docker Hub..."
docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}
docker push ${DOCKERHUB_USER}/${IMAGE_NAME}:latest

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "Docker Image: ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}"
echo ""
echo "Next steps:"
echo "1. Create network volume in RunPod (100GB, EU-RO-1)"
echo "2. Create serverless endpoint with:"
echo "   - Image: ${DOCKERHUB_USER}/${IMAGE_NAME}:${VERSION}"
echo "   - GPU: 48GB Pro (A6000) recommended"
echo "   - Volume mount: /runpod-volume"
echo "3. Add environment variables from .env"
echo "4. First run: sync_models action to download models"
echo ""
