#!/bin/bash
# Build and run Docker container for training

# Build Docker image
echo "Building Docker image..."
docker build -t question-decomposition-llm:latest .

# Run Docker container with GPU support
echo "Running Docker container..."
docker run --gpus all \
  --shm-size=8g \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -it question-decomposition-llm:latest \
  /bin/bash
