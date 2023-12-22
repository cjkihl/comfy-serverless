import time
from typing import Optional
from aiohttp import web
from pydantic import BaseModel, ValidationError
from nodes import (
    NODE_CLASS_MAPPINGS,
    CLIPTextEncode,
    EmptyLatentImage,
    VAEEncode,
    CheckpointLoaderSimple,
    init_custom_nodes,
)
import sl.cd
import sl.img
import sl.stats
from comfy.cli_args import args
import folder_paths

routes = web.RouteTableDef()


@routes.get("/")
async def status(request):
    return web.json_response(sl.stats.get_stats())


class GenerateData(BaseModel):
    seed: int
    steps: int
    cfg_scale: float
    width: int
    height: int
    restore_faces: bool
    negative_prompt: str
    prompt: str
    sampler: str
    img: Optional[str] = None


@routes.post("/v1/generate")
async def text2img(request):
    start_time = time.time()

    data = await request.json()

    try:
        d = GenerateData(**data)
    except ValidationError as e:
        return web.Response(status=400, text=str(e))

    # Load model
    checkpoint_loader = CheckpointLoaderSimple()
    checkpoint_name = folder_paths.get_filename_list("checkpoints")[0]
    model, clip, vae = checkpoint_loader.load_checkpoint(checkpoint_name)

    # CLIP Text encoder
    clip_encoder = CLIPTextEncode()
    (positive,) = clip_encoder.encode(clip, d.prompt)
    (negative,) = clip_encoder.encode(clip, d.negative_prompt)

    if d.img is None:
        n = EmptyLatentImage()
        (latent,) = n.generate(width=512, height=512)
    else:
        latent, _ = sl.img.load_base64_image(d.img)
        vae_encoder = VAEEncode()
        (latent,) = vae_encoder.encode(vae, latent)

    samples = sl.cd.text2img_sampler(
        model,
        positive,
        negative,
        latent=latent,
        seed=d.seed,
        steps=d.steps,
        cfg=d.cfg_scale,
        sampler_name=d.sampler,
    )
    decoded = vae.decode(samples)
    print("VAE decoded")
    sl.img.save_image(
        decoded,
        filename_prefix="CD",
    )
    end_time = time.time()

    return web.json_response({"time": end_time - start_time, **d.model_dump()})


class UpscaleData(BaseModel):
    seed: int
    steps: int
    cfg_scale: float
    width: int
    height: int
    restore_faces: bool
    negative_prompt: str
    prompt: str
    sampler: str
    img: str


@routes.post("/v1/upscale")
async def img2img(request):
    start_time = time.time()
    data = await request.json()

    try:
        d = UpscaleData(**data)
    except ValidationError as e:
        return web.Response(status=400, text=str(e))

    # Load model
    checkpoint_loader = CheckpointLoaderSimple()
    checkpoint_name = folder_paths.get_filename_list("checkpoints")[0]
    model, clip, vae = checkpoint_loader.load_checkpoint(checkpoint_name)

    # Load Upscaler
    class_def = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]
    obj = class_def()
    upscale_name = folder_paths.get_filename_list("upscale_models")[0]
    (upscale_model,) = getattr(obj, class_def.FUNCTION)(upscale_name)

    # CLIP Text encoder
    clip_encoder = CLIPTextEncode()
    (positive,) = clip_encoder.encode(clip, d.prompt)
    (negative,) = clip_encoder.encode(clip, d.negative_prompt)

    (latent, _mask) = sl.img.load_base64_image(d.img)

    # (upscaled) = ImageUpscaleWithModel

    class_def = NODE_CLASS_MAPPINGS["UltimateSDUpscale"]
    obj = class_def()
    (upscaled,) = getattr(obj, class_def.FUNCTION)(
        image=latent,
        model=model,
        upscale_model=upscale_model,
        positive=positive,
        negative=negative,
        vae=vae,
        upscale_by=2,
        seed=d.seed,
        steps=d.steps,
        cfg=d.cfg_scale,
        sampler_name=d.sampler,
        scheduler="normal",
        denoise=0.1,
        mode_type="Linear",
        tile_width=512,
        tile_height=512,
        mask_blur=8,
        tile_padding=32,
        seam_fix_mode="None",
        seam_fix_denoise=0.0,
        seam_fix_mask_blur=0.0,
        seam_fix_width=0,
        seam_fix_padding=0,
        force_uniform_tiles=False,
        tiled_decode=False,
    )

    sl.img.save_image(
        upscaled,
        filename_prefix="CD",
    )

    end_time = time.time()

    return web.json_response(
        {
            "time": end_time - start_time,
        }
    )


def create_cors_middleware(allowed_origin: str):
    @web.middleware
    async def cors_middleware(request: web.Request, handler):
        if request.method == "OPTIONS":
            # Pre-flight request. Reply successfully:
            response = web.Response()
        else:
            response = await handler(request)

        response.headers["Access-Control-Allow-Origin"] = allowed_origin
        response.headers[
            "Access-Control-Allow-Methods"
        ] = "POST, GET, DELETE, PUT, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

    return cors_middleware


def main():
    init_custom_nodes()
    app = web.Application()
    print("Starting server on port", args.port)

    app.add_routes(routes)
    app.middlewares.append(create_cors_middleware("*"))
    web.run_app(app, host=args.listen, port=args.port, keepalive_timeout=600)


if __name__ == "__main__":
    main()
