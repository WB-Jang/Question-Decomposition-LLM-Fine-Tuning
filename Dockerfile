# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false\
    PATH="/opt/poetry/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    curl \
    wget \
    vim \
    build-essential \
    cmake \
    ninja-build \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python and pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip setuptools wheel

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry --version


# Set working directory
WORKDIR /workspace

# ✅ 최신 안정 버전 설치 (권장)
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# ✅ GPU 관련 패키지 - 최신 버전 사용
RUN pip install accelerate==0.27.2 bitsandbytes==0.43.0

# 또는 보수적으로 가려면:
# RUN pip install accelerate==0.26.0 bitsandbytes==0.43.0

# ✅ 최신 안정 버전 설치 (권장)
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# ✅ GPU 관련 패키지 - 최신 버전 사용
RUN pip install transformers==4.36.2 \
    peft==0.7.1 \
    accelerate==0.27.2 \
    bitsandbytes==0.43.0 \
    datasets==2.16.1
# Copy project files
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN poetry install --no-root --no-dev || \
    (echo "Poetry install failed, trying without lock file" && \
    rm -f poetry.lock && poetry install --no-root --no-dev)

# Copy the rest of the project
COPY . .

# Install project in editable mode
RUN poetry install --no-dev || \
    (echo "Poetry install failed, installing manually" && \
    pip install -e .)

# ✅ 설치 검증
RUN python -c "import torch; print(f'PyTorch:  {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" && \
    python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')" 
# python -c "import bitsandbytes as bnb; print(f'Bitsandbytes: {bnb.__version__}')" && \
# python -c "import bitsandbytes as bnb; print('✅ 4-bit quantization supported!' if hasattr(bnb.nn, 'Linear4bit') else '⚠️ Limited support')"

# Expose port for Jupyter or other services
EXPOSE 8888

# Default command
CMD ["/bin/bash"]
