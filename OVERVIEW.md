# Project Overview

## 프로젝트 설명

Question-Decomposition-LLM-Fine-Tuning은 복잡한 질문을 단순한 여러 개의 하위 질문으로 분해하는 언어 모델을 LoRA(Low-Rank Adaptation) 방식으로 파인튜닝하는 프로젝트입니다.

## 주요 특징

### 1. 효율적인 학습
- **LoRA 파인튜닝**: 전체 모델 파라미터의 0.1-1%만 학습
- **4-bit/8-bit 양자화**: QLoRA를 통한 메모리 효율적 학습
- **Gradient Accumulation**: 작은 배치 크기로도 효과적인 학습

### 2. 다양한 환경 지원
- **로컬 환경**: RTX 4060 8GB VRAM에 최적화
- **Docker**: 재현 가능한 환경
- **Devcontainer**: VSCode 통합 개발 환경
- **RunPod**: 클라우드 GPU 플랫폼

### 3. 개발자 친화적
- **Poetry**: 깔끔한 의존성 관리
- **샘플 데이터**: 즉시 테스트 가능
- **상세한 문서**: 단계별 가이드
- **유연한 설정**: YAML 기반 구성

## 프로젝트 구조

```
Question-Decomposition-LLM-Fine-Tuning/
│
├── configs/                  # 설정 파일
│   ├── config.yaml          # 기본 설정
│   └── config_low_memory.yaml  # 저메모리 설정
│
├── data/                    # 데이터 디렉토리
│   ├── examples/           # 샘플 데이터
│   │   ├── sample_train.json
│   │   └── sample_train.jsonl
│   └── DATA_FORMAT.md      # 데이터 형식 문서
│
├── scripts/                # 실행 스크립트
│   ├── docker_run.sh      # Docker 실행
│   ├── train_local.sh     # 로컬 학습
│   ├── train_low_memory.sh # 저메모리 학습
│   └── train_runpod.sh    # RunPod 학습
│
├── src/                    # 소스 코드
│   ├── data/              # 데이터 처리
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models/            # 모델 관련
│   │   ├── __init__.py
│   │   └── model_loader.py
│   ├── training/          # 학습 관련
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── utils/             # 유틸리티
│   │   ├── __init__.py
│   │   └── helpers.py
│   ├── __init__.py
│   └── config.py          # 설정 클래스
│
├── .devcontainer/         # VSCode Devcontainer
│   └── devcontainer.json
│
├── .env.example           # 환경 변수 템플릿
├── .gitignore            # Git 제외 파일
├── CHANGELOG.md          # 변경 이력
├── Dockerfile            # Docker 이미지
├── docker-compose.yml    # Docker Compose
├── inference.py          # 추론 스크립트
├── LICENSE               # MIT 라이선스
├── pyproject.toml        # Poetry 설정
├── QUICKSTART.md         # 빠른 시작 가이드
├── README.md             # 메인 문서
├── requirements.txt      # pip 의존성
├── RUNPOD_GUIDE.md       # RunPod 가이드
└── train.py              # 학습 메인 스크립트
```

## 핵심 컴포넌트

### 1. 데이터 처리 (`src/data/`)
- `dataset.py`: 질문 분해 데이터셋 로더
  - JSON/JSONL 형식 지원
  - 자동 데이터 분할
  - 프롬프트 생성
  - 샘플 데이터 제공

### 2. 모델 (`src/models/`)
- `model_loader.py`: 모델 로딩 및 LoRA 설정
  - Hugging Face 모델 로드
  - 4-bit/8-bit 양자화
  - LoRA 어댑터 추가
  - PEFT 모델 생성

### 3. 학습 (`src/training/`)
- `trainer.py`: 커스텀 트레이너
  - Hugging Face Trainer 래퍼
  - 학습 설정 관리
  - 체크포인트 저장
  - 평가 및 로깅

### 4. 유틸리티 (`src/utils/`)
- `helpers.py`: 헬퍼 함수
  - 시드 설정
  - GPU 정보 확인
  - 메모리 모니터링
  - 시간 포맷팅

### 5. 설정 (`src/config.py`)
- 데이터클래스 기반 설정
- 모델, LoRA, 학습, 데이터 설정
- YAML 파일 로드 지원

## 기술 스택

### 핵심 라이브러리
- **PyTorch** (2.1.0+): 딥러닝 프레임워크
- **Transformers** (4.36.0+): Hugging Face 모델
- **PEFT** (0.7.0+): LoRA 구현
- **bitsandbytes** (0.41.0+): 양자화
- **Accelerate** (0.25.0+): 분산 학습

### 개발 도구
- **Poetry**: 의존성 관리
- **Docker**: 컨테이너화
- **Weights & Biases**: 실험 추적

## 학습 워크플로우

1. **데이터 준비**
   - JSON/JSONL 형식으로 데이터 작성
   - `data/` 디렉토리에 배치

2. **모델 선택**
   - Hugging Face 모델 선택
   - 한국어: `beomi/KoAlpaca-Polyglot-5.8B`
   - 영어: `meta-llama/Llama-2-7b-hf`

3. **설정 조정**
   - `configs/config.yaml` 수정
   - 또는 CLI 인자 사용

4. **학습 실행**
   ```bash
   python train.py --train_data data/train.json
   ```

5. **모니터링**
   - Weights & Biases 대시보드
   - 로컬 로그 파일
   - GPU 사용률 확인

6. **평가**
   - 자동 평가 (eval 데이터셋)
   - 수동 추론 테스트

7. **배포**
   - LoRA 어댑터 저장
   - Hugging Face Hub 업로드 (선택)
   - API 서버 구축 (선택)

## 메모리 최적화

### RTX 4060 8GB VRAM 설정
```python
# config_low_memory.yaml 사용
- 4-bit 양자화: 모델 크기 75% 감소
- 배치 크기 1: 메모리 사용 최소화
- LoRA rank 8: 파라미터 수 감소
- 시퀀스 길이 256: 메모리 사용 절반
- 8-bit 옵티마이저: 옵티마이저 메모리 감소
```

### 추정 메모리 사용량
- **Llama-2-7B (4-bit)**
  - 모델: ~4GB
  - 옵티마이저: ~1GB
  - Gradient: ~1GB
  - 활성화: ~1GB
  - 여유: ~1GB
  - **총계: ~8GB**

## 성능 예상

### 학습 시간 (예상)
- **RTX 4060 (8GB)**
  - 1,000 샘플: ~2-3시간
  - 10,000 샘플: ~20-30시간

- **RTX 4090 (24GB)**
  - 1,000 샘플: ~30분
  - 10,000 샘플: ~5시간

- **A100 (40GB/80GB)**
  - 1,000 샘플: ~15분
  - 10,000 샘플: ~2.5시간

### 모델 크기
- **Base 모델**: 13GB (FP16)
- **4-bit 양자화**: 3.5GB
- **LoRA 어댑터**: 10-50MB

## 확장 가능성

### 다른 모델 사용
```python
# config.yaml에서 model_name 변경
model_name: "다른-모델-이름"
```

지원 모델:
- LLaMA, Llama-2
- Mistral
- GPT-NeoX
- Falcon
- 기타 Hugging Face 모델

### 다른 태스크 적용
현재 구조는 다음 태스크에도 적용 가능:
- 요약 (Summarization)
- 번역 (Translation)
- 감정 분석 (Sentiment Analysis)
- 명명된 개체 인식 (NER)

`src/data/dataset.py`의 프롬프트 템플릿만 수정하면 됩니다.

## 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

## 기여

- Issue 제보: 버그, 기능 요청
- Pull Request: 코드 기여 환영
- 문서 개선: 오타, 설명 추가

## 참고 자료

- [LoRA 논문](https://arxiv.org/abs/2106.09685)
- [QLoRA 논문](https://arxiv.org/abs/2305.14314)
- [PEFT 문서](https://huggingface.co/docs/peft)
- [Transformers 문서](https://huggingface.co/docs/transformers)

## 문의

GitHub Issues를 통해 질문하거나 문제를 보고해주세요.
