#!/bin/bash
# Container boot script for pre-installed image
#
# This boot script is used for the gpu-installed target where all code and extensions
# are already installed in the image. It skips all installation steps and only handles
# models persistence and supervisor startup.
#
# Responsibilities:
# - Ensure models persistence: seed configs and link /comfy/models -> /models
# - Start Supervisor which launches ComfyUI

set -e

export PYTHONUNBUFFERED=1

# Defaults (can be overridden via env)
CODE_DIR=${CODE_DIR:-/comfy}

# Load .env file if it exists in CODE_DIR
if [[ -f "$CODE_DIR/.env" ]]; then
    echo "[container_boot-installed] Loading environment variables from .env"
    set -a  # automatically export all variables
    source "$CODE_DIR/.env"
    set +a
fi

echo "[container_boot-installed] Running in $CODE_DIR (pre-installed)"
echo "[container_boot-installed] Skipping all installation steps"

# Ensure models mount and symlink
mkdir -p /models
if [[ -d "$CODE_DIR/models/config" && ! -d "/models/config" ]]; then
    echo "[container_boot-installed] Seeding /models/config from repo ..."
    cp -a "$CODE_DIR/models/config" /models/
fi
if [[ -d "$CODE_DIR/models/configs" && ! -d "/models/configs" ]]; then
    echo "[container_boot-installed] Seeding /models/configs from repo ..."
    cp -a "$CODE_DIR/models/configs" /models/
fi
# Use atomic symlink creation to avoid race conditions
if [[ -e "$CODE_DIR/models" && ! -L "$CODE_DIR/models" ]]; then
    rm -rf "$CODE_DIR/models"
fi
ln -sfn /models "$CODE_DIR/models.tmp"
mv -Tf "$CODE_DIR/models.tmp" "$CODE_DIR/models"

echo "[container_boot-installed] Boot complete; starting Supervisor"
exec supervisord -c /etc/supervisord.conf

