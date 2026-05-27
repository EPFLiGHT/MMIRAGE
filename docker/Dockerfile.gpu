FROM docker.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    build-essential \
    libnuma1 \
  && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace/MMIRAGE

COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu129 \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1

COPY . .

RUN pip install --no-cache-dir ".[gpu]"

ENTRYPOINT ["/bin/bash"]