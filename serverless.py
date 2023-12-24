import time
from aiohttp import web
from marshmallow import Schema, fields
import torch

from nodes import (
    init_custom_nodes,
)
from comfy.cli_args import args
from sl import stats
from sl.cd import (
    encode_clip,
    load_loras,
    load_model,
    sample,
    save_images,
    upscale,
    validate,
)

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
    loras = fields.List(fields.Str(), default=[])


@routes.post("/v1/generate")
async def text2img(request):
    start_time = time.time()

    (d, r) = await validate(GenerateSchema, request)
    if r is not None:
        return r

    with torch.inference_mode():
        model, clip, vae = load_model()
        model, clip = load_loras(d["loras"], model, clip)
        (positive, negative) = encode_clip(clip, d)
        decoded = sample(model, d, positive, negative, vae)
        images = save_images(decoded)

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
    loras = fields.List(fields.Str(), default=[])


@routes.post("/v1/upscale")
async def img2img(request):
    start_time = time.time()
    (d, r) = await validate(UpscaleSchema, request)
    if r is not None:
        return r

    with torch.inference_mode():
        (model, clip, vae) = load_model()
        model, clip = load_loras(d["loras"], model, clip)
        (positive, negative) = encode_clip(clip, d)
        upscaled = upscale(model, d, positive, negative, vae)
        images = save_images(upscaled)

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
