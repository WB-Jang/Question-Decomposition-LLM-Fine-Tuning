# Quick Start Guide

빠르게 시작하기 위한 가이드입니다.

## 5분 안에 시작하기

### 1. 프로젝트 클론
```bash
git clone https://github.com/WB-Jang/Question-Decomposition-LLM-Fine-Tuning.git
cd Question-Decomposition-LLM-Fine-Tuning
```

### 2. 환경 설정 (택 1)

#### 옵션 A: Poetry (권장)
```bash
pip install poetry
poetry install
poetry shell
```

#### 옵션 B: pip
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 옵션 C: Docker
```bash
docker-compose up -d
docker-compose exec training bash
```

### 3. 샘플 데이터로 테스트 실행

```bash
python train.py \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --output_dir "./outputs" \
  --num_epochs 1 \
  --batch_size 1
```

이 명령은 내장된 샘플 데이터로 빠른 테스트를 실행합니다.

## RTX 4060 8GB 환경 (저메모리)

```bash
./scripts/train_low_memory.sh
```

또는:

```bash
python train.py \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --output_dir "./outputs" \
  --num_epochs 3 \
  --batch_size 1 \
  --lora_r 8 \
  --lora_alpha 16 \
  --use_4bit \
  --max_seq_length 256
```

## 자신의 데이터로 학습

### 1. 데이터 준비

`data/train.json` 파일 생성:
```json
[
  {
    "question": "복잡한 질문",
    "sub_questions": [
      "하위 질문 1",
      "하위 질문 2",
      "하위 질문 3"
    ]
  }
]
```

### 2. 학습 실행

```bash
python train.py \
  --train_data "data/train.json" \
  --output_dir "./outputs" \
  --num_epochs 3
```

## RunPod에서 실행

### 1. RunPod GPU Pod 생성
- GPU: RTX 4090 또는 A100
- Template: PyTorch

### 2. Pod에서 실행
```bash
cd /workspace
git clone https://github.com/WB-Jang/Question-Decomposition-LLM-Fine-Tuning.git
cd Question-Decomposition-LLM-Fine-Tuning
pip install poetry
poetry install
./scripts/train_runpod.sh
```

자세한 내용은 [RUNPOD_GUIDE.md](RUNPOD_GUIDE.md)를 참조하세요.

## 학습된 모델 사용

```bash
python inference.py \
  --base_model "meta-llama/Llama-2-7b-hf" \
  --adapter_path "./outputs/final_model" \
  --question "기계 학습의 주요 알고리즘과 각각의 장단점은?"
```

또는 대화형 모드:
```bash
python inference.py \
  --base_model "meta-llama/Llama-2-7b-hf" \
  --adapter_path "./outputs/final_model" \
  --interactive
```

## 일반적인 문제 해결

### CUDA Out of Memory
```bash
# 배치 크기 줄이기
python train.py --batch_size 1 ...
```

### 느린 모델 다운로드
```bash
# Hugging Face 토큰 설정
export HF_TOKEN="your_token"
```

### Docker GPU 인식 안 됨
```bash
# NVIDIA Docker 설치 확인
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## 다음 단계

- [전체 README](README.md) - 상세한 문서
- [RunPod 가이드](RUNPOD_GUIDE.md) - 클라우드 GPU 사용
- [데이터 형식](data/DATA_FORMAT.md) - 데이터 준비 가이드

## 도움이 필요하신가요?

GitHub Issues에서 질문하거나 문제를 보고해주세요:
https://github.com/WB-Jang/Question-Decomposition-LLM-Fine-Tuning/issues
