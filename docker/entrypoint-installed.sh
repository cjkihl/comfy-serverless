#!/bin/bash
# Comfy Serverless container entrypoint for pre-installed image
#
# This entrypoint is used for the gpu-installed target where all code and extensions
# are already baked into the image at build time. This skips all git operations and
# dependency installations for faster startup.
#
# Responsibilities:
# - Export selected environment variables for interactive shells
# - Ensure models persistence by linking /comfy/models -> /models
# - Start Supervisor which launches ComfyUI
#
set -e # Exit the script if any statement returns a non-true return value

export PYTHONUNBUFFERED=1 # Flush Python output directly to logs

# Load .env file if it exists
if [[ -f "/workspace/.env" ]]; then
    echo "Loading environment variables from .env"
    set -a  # automatically export all variables
    source /workspace/.env
    set +a
fi

# Persist a curated subset of environment variables for interactive shells
export_env_vars() {
    echo "Exporting environment variables..."
    printenv | grep -E '^RUNPOD_|^PATH=|^_=' | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /etc/rp_environment
    echo 'source /etc/rp_environment' >> ~/.bashrc
}

echo "Pod Started (pre-installed image)"
export_env_vars

# CODE_DIR is already set via ENV in Dockerfile, default to /comfy
CODE_DIR=${CODE_DIR:-/comfy}

echo "[entrypoint-installed] Using pre-installed code at $CODE_DIR"
echo "[entrypoint-installed] Skipping git operations and dependency installation"

echo "Delegating boot to repo script: scripts/container_boot-installed.sh"
exec pixi run bash -lc "cd '$CODE_DIR' && bash ./scripts/container_boot-installed.sh"

