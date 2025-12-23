#!/bin/bash
# Training script for local execution

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run training
python train.py \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --output_dir "./outputs" \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --use_4bit \
  --max_seq_length 512 \
  --seed 42
