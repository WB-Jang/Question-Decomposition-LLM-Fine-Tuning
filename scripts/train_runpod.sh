#!/bin/bash
# RunPod training script
# This script is designed to run on RunPod GPU instances

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export HF_HOME=/workspace/.cache/huggingface

# Install dependencies if not already installed
if [ ! -d "/workspace/.venv" ]; then
    echo "Installing dependencies..."
    poetry install --no-interaction --no-root
fi

# Activate virtual environment (if using standard venv)
# source /workspace/.venv/bin/activate

# Set Hugging Face token if available (uncomment and set your token)
# export HF_TOKEN="your_huggingface_token"

# Run training
echo "Starting training on RunPod..."
python /workspace/train.py \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --output_dir "/workspace/outputs" \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --use_4bit \
  --max_seq_length 512 \
  --seed 42

echo "Training completed!"
echo "Model saved to /workspace/outputs/final_model"
