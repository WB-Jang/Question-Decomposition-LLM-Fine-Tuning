"""
모델을 미리 다운로드만 하는 스크립트
메모리에 로딩하지 않고 캐시에만 저장
"""
import os
from huggingface_hub import snapshot_download
from pathlib import Path

# 다운로드 설정
MODEL_NAME = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
CACHE_DIR = os.path.expanduser("~/. cache/huggingface")

print("=" * 80)
print("🚀 모델 다운로드 (로딩하지 않음)")
print("=" * 80)
print(f"모델:  {MODEL_NAME}")
print(f"저장 위치: {CACHE_DIR}")
print("\n다운로드 중단 시 재실행하면 이어받습니다.")
print("=" * 80)

try:
    # 다운로드만 수행 (메모리에 로딩 X)
    model_path = snapshot_download(
        repo_id=MODEL_NAME,
        cache_dir=CACHE_DIR,
        resume_download=True,  # 중단 시 이어받기
        local_files_only=False,
        max_workers=4,  # 동시 다운로드 수
        ignore_patterns=["*.msgpack", "*.h5"],  # 불필요한 파일 제외
    )
    
    print("\n" + "=" * 80)
    print("✅ 다운로드 완료!")
    print("=" * 80)
    print(f"\n📁 다운로드 경로:")
    print(f"   {model_path}")
    
    # 실제 파일들 확인
    print(f"\n📦 다운로드된 파일:")
    for file in sorted(Path(model_path).glob("*")):
        size = file.stat().st_size / (1024**3)  # GB
        print(f"   - {file.name:40s} ({size:.2f} GB)")
    
    print("\n" + "=" * 80)
    print("🎯 다음 단계:")
    print("=" * 80)
    print("train.py 실행 시 다음 경로 사용:\n")
    print(f'python train.py --model_name "{MODEL_NAME}"')
    print("\n또는 로컬 경로 직접 지정:\n")
    print(f'python train.py --model_name "{model_path}"')
    print("=" * 80)
    
except KeyboardInterrupt:
    print("\n\n⚠️ 다운로드 중단됨")
    print("재실행하면 이어받습니다:")
    print(f"python download_model.py")
    
except Exception as e:
    print(f"\n\n❌ 오류 발생: {e}")
    print("\n해결 방법:")
    print("1. 인터넷 연결 확인")
    print("2. Hugging Face 로그인:")
    print("   huggingface-cli login")
    print("3. 모델 접근 권한 확인:")
    print(f"   https://huggingface.co/{MODEL_NAME}")