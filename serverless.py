import asyncio
import time
from aiohttp import web
from marshmallow import Schema, fields, validate
import torch
from comfy.samplers import SAMPLER_NAMES, SCHEDULER_NAMES
from comfy.cli_args import args

from server import PromptServer
from sl.async_execute import recursive_execute_async


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
    encode_clip_with_loras,
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


async def async_generator_handler():
    for i in range(5):
        output = f"Generated async token output {i}"
        yield output
        await asyncio.sleep(1)


@routes.get("/test")
async def test(request):
    response = web.StreamResponse()
    # Prepare the response headers
    response.headers["Content-Type"] = "text/plain"
    await response.prepare(request)

    # Use the async_generator_handler to stream the result
    async for data in async_generator_handler():
        await response.write(data.encode("utf-8"))

    # Indicate that the body is complete
    await response.write_eof()
    return response


class ExecuteSchema(Schema):
    prompt = fields.Dict(required=True)
    client_id = fields.Str(required=True)
    output = fields.Str(required=True)
    test = fields.Bool(dump_default=False)


object_storage = {}


@routes.post("/v1/execute")
async def execute(request):
    (d, r) = await run_validation(ExecuteSchema, request)
    if r is not None:
        return r

    response = web.StreamResponse()
    response.enable_chunked_encoding()
    # Prepare the response headers
    response.headers["Content-Type"] = "text/plain"

    await response.prepare(request)

    executed = set()
    outputs = {}
    outputs_ui = {}

    async def callback(event, data, sid=None):
        # Write to the stream
        message = f"Event: {event}, Data: {data}, SID: {sid}\n"
        print(message)
        await response.write(message.encode(data))
        await response.drain()

    client_id = d["client_id"]

    await recursive_execute_async(
        callback,
        prompt=d["prompt"],
        outputs=outputs,
        current_item=d["output"],
        extra_data={"client_id": client_id},
        executed=executed,
        prompt_id=0,
        outputs_ui=outputs_ui,
        object_storage=globals()["object_storage"],
    )

    # Indicate that the body is complete
    await response.write_eof()
    return response


class GenerateSchema(Schema):
    seed = fields.Int(dump_default=-1)
    steps = fields.Int(dump_default=20)
    cfg_scale = fields.Float(dump_default=7)
    width = fields.Int(dump_default=512)
    height = fields.Int(dump_default=512)
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


def generate_pass(model, clip, vae, d):
    negative = encode_clip(clip, d["negative_prompt"])
    (positive, model, clip) = encode_clip_with_loras(model, clip, d["prompt"])
    (decoded, seeds) = sample(model, d, positive, negative, vae)
    return (decoded, seeds)


def face_restoration_pass(image, model, clip, vae, d):
    print("Restoring faces")
    negative = encode_clip(clip, d["face_neg_prompt"])
    positive = encode_clip_with_loras(
        model,
        clip,
        d["face_prompt"],
    )
    return restore_faces(
        image,
        model,
        clip,
        vae,
        positive,
        negative,
    )


@routes.post("/v1/generate")
async def text2img(request):
    start_time = time.time()

    (d, r) = await run_validation(GenerateSchema, request)
    if r is not None:
        return r

    with torch.inference_mode():
        model, clip, vae = load_model()
        (decoded, seeds) = generate_pass(model, clip, vae, d)
        # If we have a separate face prompt, run a second pass to restore faces
        if "face_prompt" in d:
            decoded = face_restoration_pass(decoded, model, clip, vae, d)
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
        negative = encode_clip_with_loras(model, clip, d["negative_prompt"])
        positive = encode_clip_with_loras(model, clip, d["prompt"])
        (upscaled, seeds) = upscale(model, d, positive, negative, vae)
        images = save_images(upscaled)

    end_time = time.time()

    # remove image from d, we don't want it to bloat the output
    d.pop("image", None)
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
