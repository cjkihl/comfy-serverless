#!/bin/bash
# Comfy Serverless container entrypoint
#
# Responsibilities:
# - Export selected environment variables for interactive shells
# - Clone or update the application repository into CODE_DIR
# - Install Python dependencies using the repo's installers (preferred) or requirements.txt
# - Ensure models persistence by linking /comfy/models -> /models (RunPod volume)
# - Start Supervisor which launches ComfyUI
#
# Configurable environment variables (with defaults):
# - REPO_URL: Git repo to pull (default: https://github.com/cjkihl/comfy-serverless.git)
# - CODE_DIR: Filesystem location for the repo (default: /comfy)
# - REPO_REF: Optional branch/tag/commit to checkout (default: current remote head)
# - REQUIREMENTS_PATH: Path to requirements in repo (default: requirements.txt)
# - GITHUB_TOKEN: Used only if cloning private repos (optional)
# - COMFY_ARGS: Extra args passed to ComfyUI when it starts
set -e # Exit the script if any statement returns a non-true return value

export PYTHONUNBUFFERED=1 # Flush Python output directly to logs

# Load .env file if it exists
if [[ -f "/workspace/.env" ]]; then
    echo "Loading environment variables from .env"
    set -a  # automatically export all variables
    source /workspace/.env
    set +a
fi

# No SSH setup required: RunPod provides SSH out-of-the-box

# Persist a curated subset of environment variables for interactive shells
export_env_vars() {
    echo "Exporting environment variables..."
    printenv | grep -E '^RUNPOD_|^PATH=|^_=' | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /etc/rp_environment
    echo 'source /etc/rp_environment' >> ~/.bashrc
}

echo "Pod Started"
export_env_vars

# --- Clone/update repo and install dependencies ---
# REPO_URL: which repository to pull on container start.
# CODE_DIR: where the repo will live in the container filesystem.
# REPO_REF: optional branch/tag/sha to checkout.
REPO_URL=${REPO_URL:-https://github.com/cjkihl/comfy-serverless.git}
CODE_DIR=${CODE_DIR:-/comfy}
REPO_REF=${REPO_REF:-}
REQUIREMENTS_PATH=${REQUIREMENTS_PATH:-requirements.txt}

AUTH_URL="$REPO_URL"
# If GITHUB_TOKEN is provided (private repos), inject it in the URL for auth.
if [[ -n "${GITHUB_TOKEN:-}" && "$REPO_URL" =~ ^https://github.com/ ]]; then
    AUTH_URL="https://${GITHUB_TOKEN}@${REPO_URL#https://}"
fi

echo "[entrypoint] Using repo: $REPO_URL (ref: ${REPO_REF:-<default>})"

# Clone or update repository
if [[ -d "$CODE_DIR/.git" ]]; then
    echo "[entrypoint] Repo exists, updating..."
    git -C "$CODE_DIR" fetch --depth=1 origin
    CURRENT_BRANCH=$(git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo master)
    git -C "$CODE_DIR" reset --hard "origin/${CURRENT_BRANCH}"
    git -C "$CODE_DIR" clean -df
else
    echo "[entrypoint] Cloning into $CODE_DIR ..."
    rm -rf "$CODE_DIR"
    git clone --depth=1 "$AUTH_URL" "$CODE_DIR"
fi

# Checkout specific ref if provided
if [[ -n "$REPO_REF" ]]; then
    echo "[entrypoint] Checking out $REPO_REF ..."
    git -C "$CODE_DIR" checkout "$REPO_REF"
fi

# Prefer the repo-owned installer so logic remains in your repository
if [[ -f "$CODE_DIR/scripts/install_requirements.sh" ]]; then
    echo "[entrypoint] Running repo installer script install_requirements.sh via Pixi ..."
    (cd "$CODE_DIR" && pixi run bash -lc './scripts/install_requirements.sh')
elif [[ -f "$CODE_DIR/$REQUIREMENTS_PATH" ]]; then
    echo "[entrypoint] Installing requirements from $REQUIREMENTS_PATH via Pixi ..."
    pixi run python -m pip install -r "$CODE_DIR/$REQUIREMENTS_PATH"
    if [[ -d "$CODE_DIR/custom_nodes" ]]; then
        echo "[entrypoint] Installing custom_nodes requirements ..."
        find "$CODE_DIR/custom_nodes" -name 'requirements.txt' -type f -print0 | while IFS= read -r -d '' f; do
            pixi run python -m pip install -r "$f"
        done
    fi
fi

echo "Delegating boot to repo script: scripts/container_boot.sh"
exec pixi run bash -lc "cd '$CODE_DIR' && bash ./scripts/container_boot.sh"
