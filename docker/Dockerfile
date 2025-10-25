FROM ubuntu:22.04 AS base-ubuntu

ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash \
    PYTHONUNBUFFERED=1 \
    PIP_PREFER_BINARY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /workspace

RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
        git \
        wget \
        curl \
        unzip \
        ffmpeg \
        cmake \
        lsof \
        bash \
        libgl1 \
        software-properties-common \
        build-essential \
        openssh-server \
        rsync \
        supervisor \
        locales && \
    locale-gen en_US.UTF-8 && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Pixi (manages Python 3.13)
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

# Install Bun
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:${PATH}"

# Copy Pixi config and create env for linux-64
COPY pixi.toml pixi.lock ./
RUN pixi install

# Common runtime assets (paths are relative to build context root)
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/supervisord.conf /etc/supervisord.conf
RUN chmod +x /entrypoint.sh && mkdir -p /var/log/supervisor

ENV REPO_URL=https://github.com/cjkihl/comfy-serverless.git \
    CODE_DIR=/comfy \
    REQUIREMENTS_PATH=requirements.txt \
    COMFY_ARGS="--listen 0.0.0.0 --port 8188" \
    PROXY_PORT=8190 \
    PROXY_COMFY_URL=http://127.0.0.1:8188 \
    PROXY_JWT_ISSUERS=http://localhost:3002,https://colorbun.com \
    PROXY_JWT_AUDIENCES=http://localhost:3002,https://colorbun.com \
    PROXY_JWT_ALG_ALLOWLIST=EdDSA \
    PROXY_JWT_JWKS_URL=http://localhost:3002/api/auth/jwks

EXPOSE 8188 8190 9001


FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base-cuda

ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash \
    PYTHONUNBUFFERED=1 \
    PIP_PREFER_BINARY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /workspace

RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
        git \
        wget \
        curl \
        unzip \
        ffmpeg \
        cmake \
        lsof \
        bash \
        libgl1 \
        software-properties-common \
        build-essential \
        openssh-server \
        rsync \
        supervisor \
        locales && \
    locale-gen en_US.UTF-8 && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

# Install Bun
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:${PATH}"

COPY pixi.toml pixi.lock ./
RUN pixi install

COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/supervisord.conf /etc/supervisord.conf
RUN chmod +x /entrypoint.sh && mkdir -p /var/log/supervisor

ENV REPO_URL=https://github.com/cjkihl/comfy-serverless.git \
    CODE_DIR=/comfy \
    REQUIREMENTS_PATH=requirements.txt \
    COMFY_ARGS="--listen 0.0.0.0 --port 8188" \
    PROXY_PORT=8190 \
    PROXY_COMFY_URL=http://127.0.0.1:8188 \
    PROXY_JWT_ISSUERS=http://localhost:3002,https://colorbun.com \
    PROXY_JWT_AUDIENCES=http://localhost:3002,https://colorbun.com \
    PROXY_JWT_ALG_ALLOWLIST=EdDSA \
    PROXY_JWT_JWKS_URL=http://localhost:3002/api/auth/jwks

EXPOSE 8188 8190 9001


FROM base-ubuntu AS dev
RUN pixi run install-torch-cpu
CMD [ "/entrypoint.sh" ]


FROM base-cuda AS gpu
RUN pixi run install-torch-gpu && \
    pixi run python -m pip install --upgrade onnxruntime-gpu
CMD [ "/entrypoint.sh" ]


FROM gpu AS gpu-installed
# Copy entire codebase into the image
COPY . ${CODE_DIR}
WORKDIR ${CODE_DIR}

# Install all requirements and extensions during build
RUN pixi run bash -lc './scripts/install_requirements.sh'

# Use a lightweight entrypoint that skips git operations
COPY docker/entrypoint-installed.sh /entrypoint-installed.sh
RUN chmod +x /entrypoint-installed.sh
CMD [ "/entrypoint-installed.sh" ]
