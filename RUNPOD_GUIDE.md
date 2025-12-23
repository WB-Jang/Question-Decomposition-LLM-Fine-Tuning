# RunPod Setup Guide

RunPod에서 Question Decomposition LLM Fine-Tuning 프로젝트를 실행하는 방법

## 1. RunPod 계정 설정

1. [RunPod](https://www.runpod.io/) 계정 생성
2. 크레딧 충전 (최소 $10 권장)

## 2. GPU Pod 생성

### 권장 사양
- **GPU**: RTX 4090 (24GB VRAM) 또는 A100 (40GB/80GB)
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **Storage**: 50GB+

### Pod 생성 단계

1. RunPod Dashboard → "Deploy" 클릭
2. GPU 선택 (RTX 4090 또는 A100 권장)
3. Template 선택:
   - **추천**: `RunPod PyTorch`
   - 또는 `RunPod Tensorflow` (PyTorch가 포함됨)
4. Volume 설정:
   - 새 Volume 생성 (50GB 이상)
   - 기존 Volume 사용 가능
5. "Deploy" 클릭

## 3. Pod 접속

### 방법 1: Web Terminal

1. Pod 카드에서 "Connect" → "Start Web Terminal"
2. 브라우저에서 터미널 접속

### 방법 2: SSH

```bash
# RunPod에서 제공하는 SSH 명령어 복사
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa
```

### 방법 3: Jupyter Lab

1. "Connect" → "Jupyter Lab" 클릭
2. 터미널 열기

## 4. 프로젝트 설정

```bash
# 1. 작업 디렉토리로 이동
cd /workspace

# 2. 프로젝트 클론
git clone https://github.com/WB-Jang/Question-Decomposition-LLM-Fine-Tuning.git
cd Question-Decomposition-LLM-Fine-Tuning

# 3. Poetry 설치 (없는 경우)
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"

# 4. 의존성 설치
poetry install

# 5. GPU 확인
nvidia-smi
```

## 5. 데이터 업로드

### 방법 1: Git으로 관리
```bash
# 데이터가 포함된 브랜치/저장소에서 pull
git pull origin main
```

### 방법 2: RunPod File Manager
1. Pod 카드에서 "Connect" → "File Manager"
2. 파일 업로드

### 방법 3: wget/curl
```bash
# 외부 URL에서 다운로드
wget https://your-data-url.com/train.json -O data/train.json
```

### 방법 4: rsync/scp
```bash
# 로컬에서 Pod로 전송
scp -P <port> -i ~/.ssh/id_rsa data/train.json root@<pod-ip>:/workspace/Question-Decomposition-LLM-Fine-Tuning/data/
```

## 6. 학습 실행

### 기본 학습
```bash
cd /workspace/Question-Decomposition-LLM-Fine-Tuning

# 스크립트 실행
./scripts/train_runpod.sh
```

### 커스텀 설정
```bash
python train.py \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --train_data "/workspace/data/train.json" \
  --output_dir "/workspace/outputs" \
  --num_epochs 3 \
  --batch_size 8 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --use_4bit
```

## 7. 학습 모니터링

### Weights & Biases
```bash
# wandb 로그인
wandb login <your-api-key>

# 학습 시작 (자동으로 wandb에 로그)
python train.py ...
```

### Tensorboard (대안)
```bash
# 설정에서 report_to를 "tensorboard"로 변경
tensorboard --logdir outputs/runs --port 6006
```

## 8. 학습 중 모니터링

### 터미널에서 로그 확인
```bash
# 실시간 로그 확인
tail -f outputs/training.log
```

### GPU 사용량 확인
```bash
# 실시간 GPU 모니터링
watch -n 1 nvidia-smi
```

### 시스템 리소스 확인
```bash
htop
```

## 9. 모델 저장 및 다운로드

### 모델 위치
학습 완료 후: `/workspace/Question-Decomposition-LLM-Fine-Tuning/outputs/final_model/`

### 다운로드 방법

#### 방법 1: File Manager
1. "Connect" → "File Manager"
2. outputs 폴더에서 모델 다운로드

#### 방법 2: SCP
```bash
# 로컬에서 실행
scp -P <port> -i ~/.ssh/id_rsa -r root@<pod-ip>:/workspace/Question-Decomposition-LLM-Fine-Tuning/outputs/final_model ./
```

#### 방법 3: Hugging Face Hub에 업로드
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="outputs/final_model",
    repo_id="your-username/your-model-name",
    token="your-hf-token"
)
```

## 10. 비용 최적화 팁

### 1. Spot Instances 사용
- 비용 50-80% 절감
- 단, 인스턴스가 중단될 수 있음
- 체크포인트를 자주 저장하여 대비

### 2. Auto-Stop 설정
```bash
# 학습 완료 후 자동 종료
python train.py ... && sudo shutdown -h now
```

### 3. Volume 활용
- Volume에 모델과 데이터 저장
- Pod 종료 후에도 데이터 유지
- 다음 실행 시 재사용

### 4. 효율적인 학습 설정
- Mixed precision training (bf16/fp16)
- Gradient accumulation
- 적절한 batch size 선택

## 11. 트러블슈팅

### 문제: CUDA Out of Memory
```bash
# 환경 변수 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 더 작은 배치 크기 사용
python train.py --batch_size 1 ...
```

### 문제: 모델 다운로드 느림
```bash
# Hugging Face 미러 사용 (중국의 경우)
export HF_ENDPOINT=https://hf-mirror.com
```

### 문제: 디스크 공간 부족
```bash
# 캐시 정리
rm -rf ~/.cache/huggingface/*

# Volume 확장 (RunPod Dashboard에서)
```

### 문제: SSH 연결 끊김
```bash
# tmux 또는 screen 사용
tmux new -s training
python train.py ...
# Ctrl+B, D로 분리
# 재접속 시: tmux attach -t training
```

## 12. 학습 완료 후

1. **모델 다운로드**: 위의 방법으로 모델 저장
2. **Pod 종료**: "Stop" 버튼 클릭 (비용 절감)
3. **Volume 유지**: 다음 학습에 재사용 가능
4. **Pod 삭제**: Volume만 유지하고 Pod 삭제 가능

## 13. 재시작 (Volume 사용)

```bash
# 1. 동일한 Volume으로 새 Pod 생성
# 2. 작업 디렉토리 확인
cd /workspace/Question-Decomposition-LLM-Fine-Tuning

# 3. 이전 체크포인트에서 재개
python train.py --resume_from_checkpoint outputs/checkpoint-XXX ...
```

## 참고 링크

- [RunPod 공식 문서](https://docs.runpod.io/)
- [RunPod Pricing](https://www.runpod.io/pricing)
- [RunPod Community Discord](https://discord.gg/runpod)
