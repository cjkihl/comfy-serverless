Generic container that clones the repo at start and installs deps via Pixi + pip.

GPU build/run:

```bash
docker build -f docker/Dockerfile --target gpu -t comfy:gpu .
docker run --gpus all -p 8188:8188 \
  -e REPO_URL=https://github.com/cjkihl/comfy-serverless.git \
  comfy:gpu
```

DEV (CPU) build/run:

```bash
docker build -f docker/Dockerfile --target dev -t comfy:dev .
docker run -p 8188:8188 comfy:dev
```

Environment variables:

- REPO_URL (default: https://github.com/cjkihl/comfy-serverless.git)
- REPO_REF (optional branch/tag/sha)
- CODE_DIR (default: /comfy)
- REQUIREMENTS_PATH (default: requirements.txt)
- COMFY_ARGS (default: --listen 0.0.0.0 --port 8188)

