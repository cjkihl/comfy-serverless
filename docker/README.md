Generic container that clones the repo at start and installs deps via Pixi + pip.

GPU build/run:

```bash
docker build -f docker/Dockerfile --target gpu -t comfy:gpu .
docker run --gpus all -p 8188:8188 comfy:gpu
```

DEV (CPU) build/run:

```bash
docker build -f docker/Dockerfile --target dev -t comfy:dev .
docker run -p 8188:8188 comfy:dev
```

Environment variables:

- REPO_REF (optional branch/tag/sha)
- COMFY_ARGS (default: --listen 0.0.0.0 --port 8188)

## COMFY_ARGS Configuration

The `COMFY_ARGS` environment variable allows you to pass command-line arguments to ComfyUI. This is particularly useful for controlling model memory management and preventing model unloading between executions.

### Preventing Model Unloading

If you're using the same model repeatedly and experiencing slow execution times (50+ seconds) due to model reloading, you can use VRAM management flags:

**`--highvram`** (Recommended for most cases):
- Keeps models in GPU memory instead of unloading to CPU after use
- Prevents the 50-second reload delay when reusing the same model
- Use when you have sufficient GPU memory and want consistent fast execution

**`--gpu-only`** (More aggressive):
- Stores everything on GPU including text encoders/CLIP models
- Uses more VRAM but ensures nothing is offloaded to CPU
- Use when you have plenty of GPU memory and want maximum performance

### Usage Examples

**Docker command line:**
```bash
# Prevent model unloading with --highvram
docker run --gpus all -p 8188:8188 \
  -e COMFY_ARGS="--listen 0.0.0.0 --port 8188 --highvram" \
  comfy:gpu

# More aggressive: keep everything on GPU
docker run --gpus all -p 8188:8188 \
  -e COMFY_ARGS="--listen 0.0.0.0 --port 8188 --gpu-only" \
  comfy:gpu

# Combine with other flags
docker run --gpus all -p 8188:8188 \
  -e COMFY_ARGS="--listen 0.0.0.0 --port 8188 --highvram --verbose DEBUG" \
  comfy:gpu
```

**RunPod / Vast.ai:**
Set `COMFY_ARGS` in the environment variables section of your pod/template:

```
COMFY_ARGS=--listen 0.0.0.0 --port 8188 --highvram
```

Or for more aggressive GPU usage:
```
COMFY_ARGS=--listen 0.0.0.0 --port 8188 --gpu-only
```

### When to Use Each Flag

- **Default (no flag)**: Models are unloaded to CPU after use. Use when you have limited VRAM or switch between many different models.

- **`--highvram`**: Models stay in GPU memory. Use when:
  - You repeatedly use the same model(s)
  - You have sufficient GPU memory (8GB+ recommended)
  - You want consistent fast execution (<1 second vs 50+ seconds)

- **`--gpu-only`**: Everything stays on GPU. Use when:
  - You have plenty of GPU memory (16GB+ recommended)
  - You want maximum performance
  - You're using the same models repeatedly

### Other Useful COMFY_ARGS Options

- `--lowvram`: Split the unet in parts to use less VRAM (for limited GPU memory)
- `--normalvram`: Force normal VRAM use if lowvram gets automatically enabled
- `--cpu-vae`: Run the VAE on the CPU to save GPU memory
- `--verbose DEBUG`: Enable debug logging to troubleshoot issues

See ComfyUI's CLI arguments documentation for the full list of available options.

