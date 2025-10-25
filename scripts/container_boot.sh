#!/bin/bash
# Container boot script that runs inside the cloned repository (CODE_DIR)
#
# Responsibilities:
# - Install dependencies using repo-provided installers or requirements.txt
# - Ensure models persistence: seed configs and link /comfy/models -> /models
# - Start Supervisor which launches ComfyUI

set -e

export PYTHONUNBUFFERED=1

# Load .env file if it exists in CODE_DIR
CODE_DIR=${CODE_DIR:-/comfy}
if [[ -f "$CODE_DIR/.env" ]]; then
    echo "[container_boot] Loading environment variables from .env"
    set -a  # automatically export all variables
    source "$CODE_DIR/.env"
    set +a
fi

# Defaults (can be overridden via env)
CODE_DIR=${CODE_DIR:-/comfy}
REQUIREMENTS_PATH=${REQUIREMENTS_PATH:-requirements.txt}

echo "[container_boot] Running in $CODE_DIR"

# Prefer repo-owned installer to keep logic versioned in the repo
if [[ -f "$CODE_DIR/scripts/install_requirements.sh" ]]; then
    echo "[container_boot] Running install_requirements.sh via Pixi ..."
    (cd "$CODE_DIR" && pixi run bash -lc './scripts/install_requirements.sh')
elif [[ -f "$CODE_DIR/$REQUIREMENTS_PATH" ]]; then
    echo "[container_boot] Installing requirements from $REQUIREMENTS_PATH via Pixi ..."
    pixi run python -m pip install -r "$CODE_DIR/$REQUIREMENTS_PATH"
    if [[ -d "$CODE_DIR/custom_nodes" ]]; then
        echo "[container_boot] Installing custom_nodes requirements ..."
        find "$CODE_DIR/custom_nodes" -name 'requirements.txt' -type f -print0 | while IFS= read -r -d '' f; do
            pixi run python -m pip install -r "$f"
        done
    fi
fi

# Install proxy dependencies
if [[ -d "$CODE_DIR/proxy" ]]; then
    echo "[container_boot] Installing proxy dependencies..."
    (cd "$CODE_DIR/proxy" && bun install)
fi

# Ensure models mount and symlink
mkdir -p /models
if [[ -d "$CODE_DIR/models/config" && ! -d "/models/config" ]]; then
    echo "[container_boot] Seeding /models/config from repo ..."
    cp -a "$CODE_DIR/models/config" /models/
fi
if [[ -d "$CODE_DIR/models/configs" && ! -d "/models/configs" ]]; then
    echo "[container_boot] Seeding /models/configs from repo ..."
    cp -a "$CODE_DIR/models/configs" /models/
fi
# Use atomic symlink creation to avoid race conditions
if [[ -e "$CODE_DIR/models" && ! -L "$CODE_DIR/models" ]]; then
    rm -rf "$CODE_DIR/models"
fi
ln -sfn /models "$CODE_DIR/models.tmp"
mv -Tf "$CODE_DIR/models.tmp" "$CODE_DIR/models"

echo "[container_boot] Boot complete; starting Supervisor"
exec supervisord -c /etc/supervisord.conf


