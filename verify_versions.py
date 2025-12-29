"""
설치된 패키지 및 CUDA 호환성 검증
"""
import sys

print("=" * 80)
print("🔍 환경 검증")
print("=" * 80)

# 1. Python 버전
print(f"\n✅ Python: {sys.version. split()[0]}")

# 2. PyTorch 및 CUDA
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ GPU:  {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Compute Capability: {torch.cuda. get_device_capability(0)}")
except ImportError: 
    print("❌ PyTorch not installed")

# 3. Accelerate
try:
    import accelerate
    print(f"✅ Accelerate: {accelerate.__version__}")
    
    # 버전 체크
    from packaging import version
    if version.parse(accelerate.__version__) >= version.parse("0.26.0"):
        print("   ✅ Version OK for CUDA 12.1")
    else:
        print("   ⚠️ Consider upgrading to 0.26.0+")
except ImportError:
    print("❌ Accelerate not installed")

# 4. Bitsandbytes
try:
    import bitsandbytes as bnb
    print(f"✅ Bitsandbytes: {bnb.__version__}")
    
    # 4-bit 지원 확인
    if hasattr(bnb.nn, 'Linear4bit'):
        print("   ✅ 4-bit quantization supported")
    else:
        print("   ⚠️ 4-bit quantization NOT supported - upgrade to 0.43.0+")
    
    # CUDA 인식 확인
    if torch.cuda.is_available():
        try:
            # 간단한 4-bit 테스트
            from transformers import BitsAndBytesConfig
            config = BitsAndBytesConfig(load_in_4bit=True)
            print("   ✅ BitsAndBytesConfig works")
        except Exception as e:
            print(f"   ⚠️ BitsAndBytesConfig error: {e}")
            
    # 버전 권장
    from packaging import version
    if version. parse(bnb.__version__) >= version.parse("0.43.0"):
        print("   ✅ Version OK for RTX 4060 + CUDA 12.1")
    else:
        print("   ⚠️ Upgrade to 0.43.0+ recommended for RTX 4060")
        
except ImportError as e:
    print(f"❌ Bitsandbytes not installed: {e}")

# 5. 기타 패키지
try: 
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except ImportError:
    print("❌ Transformers not installed")

try:
    import peft
    print(f"✅ PEFT: {peft.__version__}")
except ImportError:
    print("❌ PEFT not installed")

print("\n" + "=" * 80)
print("✅ 검증 완료!")
print("=" * 80)
