import time
from aiohttp import web
from marshmallow import Schema, fields, validate
import torch
from comfy.samplers import SAMPLER_NAMES, SCHEDULER_NAMES
from comfy.cli_args import args

from server import PromptServer


class DummyPromptServer:
    client_id = None
    routes = web.RouteTableDef()

    def send_sync(self, event, data, sid=None):
        pass

    def add_on_prompt_handler(self, handler):
        pass


## Mock PromptServer so custom nodes will not crash
PromptServer.instance = DummyPromptServer()

from nodes import (
    init_custom_nodes,
)
from sl import stats
from sl.cd import (
    encode_clip,
    load_loras,
    load_model,
    restore_faces,
    sample,
    save_images,
    upscale,
    run_validation,
)

routes = web.RouteTableDef()


@routes.get("/")
async def status(_):
    return web.json_response(stats.get_stats())


class GenerateSchema(Schema):
    seed = fields.Int(dump_default=-1)
    steps = fields.Int(dump_default=20)
    cfg_scale = fields.Float(dump_default=7)
    width = fields.Int(dump_default=512)
    height = fields.Int(dump_default=512)
    restore_faces = fields.Bool(dump_default=False)
    face_prompt = fields.Str()
    face_loras = fields.List(fields.Str(), dump_default=[])
    negative_prompt = fields.Str(required=True)
    prompt = fields.Str(required=True)
    batch_size = fields.Int(dump_default=1)
    sampler = fields.Str(dump_default="euler", validate=validate.OneOf(SAMPLER_NAMES))
    scheduler = fields.Str(
        dump_default="normal", validate=validate.OneOf(SCHEDULER_NAMES)
    )
    test = fields.Bool(dump_default=False)
    loras = fields.List(fields.Str(), dump_default=[])


@routes.post("/v1/generate")
async def text2img(request):
    start_time = time.time()

    (d, r) = await run_validation(GenerateSchema, request)
    if r is not None:
        return r

    with torch.inference_mode():
        model, clip, vae = load_model()
        model, clip = load_loras(d["loras"], model, clip)
        negative = encode_clip(clip, d["negative_prompt"])
        positive = encode_clip(clip, d["prompt"])
        (decoded, seeds) = sample(model, d, positive, negative, vae)
        if "restore_faces" in d:
            if "face_prompt" in d:
                (positive) = encode_clip(clip, d["face_prompt"])
            if "face_loras" in d:
                model, clip = load_loras(d["face_loras"], model, clip)
            images = restore_faces(decoded, model, clip, vae, positive, negative)
        images = save_images(decoded)
    end_time = time.time()

    r = {"time": end_time - start_time, "info": d, "seeds": seeds}
    print(r)
    r["images"] = images
    return web.json_response(r)


class UpscaleSchema(Schema):
    seed = fields.Int(dump_default=-1)
    steps = fields.Int(dump_default=20)
    cfg_scale = fields.Float()
    upscale_by = fields.Int(dump_default=2)
    restore_faces = fields.Bool(dump_default=False)
    negative_prompt = fields.Str(required=True)
    prompt = fields.Str(required=True)
    sampler = fields.Str(dump_default="euler", validate=validate.OneOf(SAMPLER_NAMES))
    scheduler = fields.Str(
        dump_default="normal", validate=validate.OneOf(SCHEDULER_NAMES)
    )
    denoising_strength = fields.Float(dump_default=0.3)
    image = fields.Str(required=True)
    test = fields.Bool(dump_default=False)
    loras = fields.List(fields.Str(), dump_default=[])


@routes.post("/v1/upscale")
async def img2img(request):
    start_time = time.time()
    (d, r) = await run_validation(UpscaleSchema, request)
    if r is not None:
        return r

    with torch.inference_mode():
        (model, clip, vae) = load_model()
        model, clip = load_loras(d["loras"], model, clip)
        negative = encode_clip(clip, d["negative_prompt"])
        positive = encode_clip(clip, d["prompt"])
        (upscaled, seeds) = upscale(model, d, positive, negative, vae)
        images = save_images(upscaled)

    end_time = time.time()

    r = {"time": end_time - start_time, "info": d, "seeds": seeds}
    print(r)
    r["images"] = images
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
