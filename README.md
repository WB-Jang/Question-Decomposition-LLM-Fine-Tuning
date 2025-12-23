# Question-Decomposition-LLM-Fine-Tuning

복잡한 질문을 단순한 여러 개의 하위 질문으로 분해하는 언어 모델 파인튜닝 프로젝트

이 프로젝트는 LoRA(Low-Rank Adaptation) 방식을 사용하여 대형 언어 모델(LLM)을 효율적으로 파인튜닝합니다. Docker, Poetry, Devcontainer를 활용하며, RTX 4060 8GB VRAM 환경 및 RunPod에서 실행 가능하도록 최적화되어 있습니다.

## 주요 기능

- ✅ **LoRA 파인튜닝**: 메모리 효율적인 LoRA를 사용한 모델 학습
- ✅ **4-bit/8-bit 양자화**: 제한된 GPU 메모리에서도 대형 모델 학습 가능
- ✅ **Docker 지원**: 재현 가능한 개발 및 학습 환경
- ✅ **Devcontainer**: VSCode에서 바로 개발 가능
- ✅ **RunPod 호환**: 클라우드 GPU에서 즉시 실행 가능
- ✅ **Poetry 의존성 관리**: 깔끔한 패키지 관리
- ✅ **저메모리 설정**: RTX 4060 8GB VRAM에 최적화

## 프로젝트 구조

```
.
├── configs/                    # 설정 파일
│   ├── config.yaml            # 기본 학습 설정
│   └── config_low_memory.yaml # 저메모리 설정
├── data/                      # 데이터 디렉토리
│   └── examples/              # 샘플 데이터
├── scripts/                   # 실행 스크립트
│   ├── docker_run.sh         # Docker 실행
│   ├── train_local.sh        # 로컬 학습
│   ├── train_low_memory.sh   # 저메모리 학습
│   └── train_runpod.sh       # RunPod 학습
├── src/                       # 소스 코드
│   ├── data/                 # 데이터 로딩
│   ├── models/               # 모델 관련
│   ├── training/             # 학습 관련
│   └── utils/                # 유틸리티
├── .devcontainer/            # Devcontainer 설정
├── Dockerfile                # Docker 이미지 설정
├── docker-compose.yml        # Docker Compose 설정
├── pyproject.toml            # Poetry 의존성
└── train.py                  # 메인 학습 스크립트
```

## 시작하기

### 사전 요구사항

- Python 3.10+
- NVIDIA GPU (CUDA 지원)
- Docker (선택사항)
- Poetry (선택사항)

### 설치 방법

#### 방법 1: Poetry 사용 (권장)

```bash
# Poetry 설치
curl -sSL https://install.python-poetry.org | python3 -

# 의존성 설치
poetry install

# 가상환경 활성화
poetry shell
```

#### 방법 2: pip 사용

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install torch transformers peft datasets accelerate bitsandbytes wandb
```

#### 방법 3: Docker 사용

```bash
# Docker 이미지 빌드 및 실행
./scripts/docker_run.sh

# 또는 docker-compose 사용
docker-compose up -d
docker-compose exec training bash
```

#### 방법 4: VSCode Devcontainer

1. VSCode에서 프로젝트 열기
2. Command Palette (Ctrl+Shift+P) → "Reopen in Container"
3. 자동으로 개발 환경 구성

## 사용 방법

### 1. 데이터 준비

데이터는 JSON 또는 JSONL 형식으로 준비합니다:

**JSON 형식 (`data/train.json`):**
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

**JSONL 형식 (`data/train.jsonl`):**
```jsonl
{"question": "복잡한 질문", "sub_questions": ["하위 질문 1", "하위 질문 2"]}
```

샘플 데이터는 `data/examples/` 디렉토리에서 확인할 수 있습니다.

### 2. 학습 실행

#### 기본 학습

```bash
python train.py \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --train_data "data/train.json" \
  --output_dir "./outputs" \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4
```

#### 저메모리 설정 (RTX 4060 8GB)

```bash
# 스크립트 사용
./scripts/train_low_memory.sh

# 또는 직접 실행
python train.py \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --output_dir "./outputs" \
  --num_epochs 3 \
  --batch_size 1 \
  --learning_rate 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --use_4bit \
  --max_seq_length 256
```

#### 샘플 데이터로 테스트

```bash
python train.py \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --output_dir "./outputs" \
  --num_epochs 1 \
  --batch_size 1
```

### 3. RunPod에서 실행

RunPod에서 GPU 인스턴스를 생성한 후:

```bash
# 1. 프로젝트 클론
git clone https://github.com/WB-Jang/Question-Decomposition-LLM-Fine-Tuning.git
cd Question-Decomposition-LLM-Fine-Tuning

# 2. 의존성 설치
pip install poetry
poetry install

# 3. 학습 실행
./scripts/train_runpod.sh
```

**RunPod 설정 팁:**
- GPU: RTX 4090 또는 A100 권장
- Disk Space: 최소 50GB
- Docker Template: PyTorch 또는 CUDA 기반 이미지
- Volume Mount: `/workspace`에 프로젝트 마운트

## 설정 옵션

### 모델 설정

- `--model_name`: 사용할 Hugging Face 모델 (기본: `meta-llama/Llama-2-7b-hf`)
  - 한국어 모델 추천: `beomi/KoAlpaca-Polyglot-5.8B`
- `--use_4bit`: 4-bit 양자화 사용 (메모리 절약)

### LoRA 설정

- `--lora_r`: LoRA rank (기본: 16, 저메모리: 8)
- `--lora_alpha`: LoRA alpha (기본: 32, 저메모리: 16)

### 학습 설정

- `--num_epochs`: 학습 에폭 수 (기본: 3)
- `--batch_size`: 배치 크기 (기본: 4, 저메모리: 1)
- `--learning_rate`: 학습률 (기본: 2e-4)
- `--max_seq_length`: 최대 시퀀스 길이 (기본: 512, 저메모리: 256)

## 메모리 최적화 팁

### RTX 4060 8GB VRAM 환경

1. **4-bit 양자화 사용**: `--use_4bit` 플래그
2. **작은 배치 크기**: `--batch_size 1`
3. **Gradient Accumulation**: 효과적인 배치 크기 유지
4. **짧은 시퀀스**: `--max_seq_length 256`
5. **작은 LoRA rank**: `--lora_r 8`
6. **메모리 효율적 옵티마이저**: `paged_adamw_8bit`

### 환경 변수 설정

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TRANSFORMERS_CACHE=/path/to/cache
```

## 학습 모니터링

프로젝트는 Weights & Biases (wandb)를 사용하여 학습을 모니터링합니다:

```bash
# wandb 로그인
wandb login

# 학습 실행 (자동으로 wandb에 로그)
python train.py ...
```

## 모델 저장 및 사용

학습이 완료되면 모델은 `outputs/final_model/` 디렉토리에 저장됩니다:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 베이스 모델 로드
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(base_model, "outputs/final_model")

# 추론
question = "복잡한 질문을 입력하세요"
prompt = f"### Instruction:\n복잡한 질문을 단순한 여러 개의 하위 질문으로 분해하세요.\n\n### Question:\n{question}\n\n### Sub-questions:\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## 트러블슈팅

### CUDA Out of Memory

- 배치 크기 줄이기: `--batch_size 1`
- 시퀀스 길이 줄이기: `--max_seq_length 256`
- LoRA rank 줄이기: `--lora_r 8`
- Gradient checkpointing 활성화

### 모델 다운로드 느림

```bash
# Hugging Face 캐시 디렉토리 설정
export HF_HOME=/path/to/large/disk/.cache/huggingface
```

### Permission Denied (Docker)

```bash
# 스크립트 실행 권한 부여
chmod +x scripts/*.sh
```

## 참고 자료

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [RunPod Documentation](https://docs.runpod.io/)

## 라이선스

MIT License

## 기여

이슈와 풀 리퀘스트는 언제나 환영합니다!

## 문의

프로젝트 관련 문의사항이 있으시면 이슈를 생성해주세요.
