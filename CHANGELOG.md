# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2024-12-23

### Added
- Initial project setup with LoRA fine-tuning support
- Docker configuration with CUDA support
- Poetry dependency management
- VSCode Devcontainer configuration
- Main training script (`train.py`)
- Inference script (`inference.py`)
- Data loading and preprocessing modules
- Model loading utilities with quantization support
- Training configuration files (default and low-memory)
- Shell scripts for different training scenarios:
  - Local training
  - Low-memory training (RTX 4060 8GB)
  - RunPod cloud training
  - Docker execution
- Comprehensive documentation:
  - Main README with full setup guide
  - Quick Start guide
  - RunPod deployment guide
  - Data format documentation
- Sample training data in JSON and JSONL formats
- Environment variable template
- MIT License
- `.gitignore` for Python/ML projects

### Features
- LoRA (Low-Rank Adaptation) fine-tuning
- 4-bit and 8-bit quantization support
- Optimized for RTX 4060 8GB VRAM
- RunPod cloud GPU compatibility
- Automatic dataset splitting
- Weights & Biases integration
- Korean language support
- Sample data for quick testing

### Technical Details
- Python 3.10+ support
- PyTorch 2.1.0+
- Transformers 4.36.0+
- PEFT 0.7.0+
- bitsandbytes for quantization
- Accelerate for distributed training
