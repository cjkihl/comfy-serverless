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
COPY docker/env_vars.ts /docker/env_vars.ts
RUN chmod +x /entrypoint.sh && mkdir -p /var/log/supervisor


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
RUN pixi install && \
    rm -rf /tmp/* /var/tmp/* /root/.cache/pip && \
    find /workspace/.pixi -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /workspace/.pixi -type f -name "*.pyc" -delete 2>/dev/null || true

COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/supervisord.conf /etc/supervisord.conf
COPY docker/env_vars.ts /docker/env_vars.ts
RUN chmod +x /entrypoint.sh && mkdir -p /var/log/supervisor


EXPOSE 8188 8190 9001


FROM base-ubuntu AS dev
RUN pixi run install-torch-cpu
CMD [ "/entrypoint.sh" ]


FROM base-cuda AS gpu
# Clean up before installing large packages to maximize available space
RUN rm -rf /tmp/* /var/tmp/* /root/.cache/pip && \
    find /workspace/.pixi -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /workspace/.pixi -type f -name "*.pyc" -delete 2>/dev/null || true
RUN pixi run install-torch-gpu && \
    rm -rf /tmp/* /var/tmp/* /root/.cache/pip
RUN pixi run python -m pip install --upgrade onnxruntime-gpu && \
    rm -rf /tmp/* /var/tmp/* /root/.cache/pip && \
    find /workspace/.pixi -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /workspace/.pixi -type f -name "*.pyc" -delete 2>/dev/null || true
CMD [ "/entrypoint.sh" ]


FROM gpu AS gpu-installed
# Copy entire codebase into the image
COPY . /comfy
WORKDIR /comfy

# Install all requirements and extensions during build
RUN pixi run bash -lc './scripts/install_requirements.sh' && \
    rm -rf /tmp/* /var/tmp/* /root/.cache/pip

# Install proxy dependencies (Bun is already installed in base image)
RUN cd /comfy/cj/proxy && bun install && \
    rm -rf /tmp/* /var/tmp/*

# Use a lightweight entrypoint that skips git operations
COPY docker/entrypoint-installed.sh /entrypoint-installed.sh
RUN chmod +x /entrypoint-installed.sh
CMD [ "/entrypoint-installed.sh" ]
