import random
import time
from aiohttp import web
from marshmallow import Schema, fields, ValidationError
import torch

from nodes import (
    NODE_CLASS_MAPPINGS,
    CLIPTextEncode,
    EmptyLatentImage,
    CheckpointLoaderSimple,
    KSampler,
    SaveImage,
    VAEDecode,
    init_custom_nodes,
)
from sl import img, stats
from comfy.cli_args import args
import folder_paths

routes = web.RouteTableDef()


@routes.get("/")
async def status(_):
    return web.json_response(stats.get_stats())


class GenerateSchema(Schema):
    seed = fields.Int(default=-1)
    steps = fields.Int(default=20)
    cfg_scale = fields.Float(default=7)
    width = fields.Int(default=512)
    height = fields.Int(default=512)
    restore_faces = fields.Bool(default=False)
    negative_prompt = fields.Str(required=True)
    prompt = fields.Str(required=True)
    batch_size = fields.Int(default=1)
    sampler = fields.Str(default="Euler")
    test = fields.Bool(default=False)


@routes.post("/v1/generate")
async def text2img(request):
    start_time = time.time()

    data = await request.json()
    print(data)
    schema = GenerateSchema()
    try:
        d = schema.dump(schema.load(data))
    except ValidationError as err:
        print(err.messages)
        return web.json_response({"error": True, "message": err.messages}, status=400)

    if d["test"] is True:
        return web.json_response({"time": 0.0, "img": "", **d})

    # Load model
    checkpoint_loader = CheckpointLoaderSimple()
    checkpoint_name = folder_paths.get_filename_list("checkpoints")[0]
    print(checkpoint_name)
    model, clip, vae = checkpoint_loader.load_checkpoint(checkpoint_name)
    print("Checkpoint Loaded")

    # CLIP Text encoder
    with torch.no_grad():
        clip_encoder = CLIPTextEncode()
        (positive,) = clip_encoder.encode(clip, d["prompt"])
        (negative,) = clip_encoder.encode(clip, d["negative_prompt"])
        print("CLIP Encoded")

    n = EmptyLatentImage()
    (latent,) = n.generate(
        width=d["width"], height=d["height"], batch_size=d["batch_size"]
    )
    print("Latent Image Generated")

    if d["seed"] == -1:
        d["seed"] = random.randint(0, 0xFFFFFFFFFFFFFFFF)

    s = KSampler()
    (samples,) = s.sample(
        model,
        seed=d["seed"],
        steps=d["steps"],
        cfg=d["cfg_scale"],
        sampler_name=d["sampler"],
        scheduler="normal",
        positive=positive,
        negative=negative,
        latent_image=latent,
        denoise=1.0,
    )
    print("Samples Generated")

    decoder = VAEDecode()
    (decoded,) = decoder.decode(vae, samples)
    print("VAE decoded")

    with torch.no_grad():
        img_saver = SaveImage()
        img_saver.save_images(decoded, filename_prefix="CD")

    print("Images Saved")

    images = img.images_to_base64(decoded)
    print("Images Base 64 Encoded")

    end_time = time.time()

    r = {"time": end_time - start_time, "images": images, **d}
    print(r)
    return web.json_response(r)


class UpscaleSchema(Schema):
    seed = fields.Int(default=-1)
    steps = fields.Int(default=20)
    cfg_scale = fields.Float()
    upscale_by = fields.Int(default=2)
    restore_faces = fields.Bool(default=False)
    negative_prompt = fields.Str(required=True)
    prompt = fields.Str(required=True)
    sampler = fields.Str(default="Euler")
    denoising_strength = fields.Float(default=0.3)
    img = fields.Str(required=True)
    test = fields.Bool(default=False)


@routes.post("/v1/upscale")
async def img2img(request):
    start_time = time.time()
    data = await request.json()
    schema = UpscaleSchema()

    try:
        d = schema.load(data)
    except ValidationError as err:
        print(err.messages)
        return web.json_response({"error": True, "message": err.messages}, status=400)

    if d["test"] is True:
        return web.json_response({"time": 0.0, "img": "", **schema.dump(d)})

    # Load model
    checkpoint_loader = CheckpointLoaderSimple()
    checkpoint_name = folder_paths.get_filename_list("checkpoints")[0]
    print(checkpoint_name)
    model, clip, vae = checkpoint_loader.load_checkpoint(checkpoint_name)
    print("Checkpoint Loaded")

    # Load Upscaler
    class_def = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]
    obj = class_def()
    upscale_name = folder_paths.get_filename_list("upscale_models")[0]
    (upscale_model,) = getattr(obj, class_def.FUNCTION)(upscale_name)

    # CLIP Text encoder
    clip_encoder = CLIPTextEncode()
    (positive,) = clip_encoder.encode(clip, d["prompt"])
    (negative,) = clip_encoder.encode(clip, d["negative_prompt"])

    (latent, _mask) = img.base64_to_image(d["img"])

    class_def = NODE_CLASS_MAPPINGS["UltimateSDUpscale"]
    obj = class_def()
    (upscaled,) = getattr(obj, class_def.FUNCTION)(
        image=latent,
        model=model,
        upscale_model=upscale_model,
        positive=positive,
        negative=negative,
        vae=vae,
        upscale_by=d["upscale_by"],
        seed=d["seed"],
        steps=d["steps"],
        cfg=d["cfg_scale"],
        sampler_name=d["sampler"],
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

    img.save_image(
        upscaled,
        filename_prefix="CD",
    )

    images = img.images_to_base64(upscaled)

    end_time = time.time()

    r = {"time": end_time - start_time, "images": images, **d}
    print(r)
    return web.json_response(r)


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
