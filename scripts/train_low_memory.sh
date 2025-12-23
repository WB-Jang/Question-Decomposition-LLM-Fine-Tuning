#!/bin/bash
# Training script optimized for RTX 4060 8GB VRAM

# Set environment variables for memory optimization
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# Run training with low memory configuration
python train.py \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --output_dir "./outputs" \
  --num_epochs 3 \
  --batch_size 1 \
  --learning_rate 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --use_4bit \
  --max_seq_length 256 \
  --seed 42
