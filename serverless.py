import asyncio
import json
import comfy.options
from comfy.utils import set_progress_bar_global_hook

comfy.options.enable_args_parsing()
from aiohttp import web
from marshmallow import Schema, fields
import torch
from comfy.cli_args import args
from comfy.model_management import (
    cleanup_models,
    throw_exception_if_processing_interrupted,
)

from server import PromptServer
from sl.async_execute import cd_recursive_execute_async
from sl import stats
from sl.cd import (
    run_validation,
)


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

object_storage = {}

routes = web.RouteTableDef()


@routes.get("/")
async def status(_):
    return web.json_response(stats.get_stats())


class ExecuteSchema(Schema):
    prompt = fields.Dict(required=True)
    client_id = fields.Str(required=True)
    output = fields.Str(required=True)
    test = fields.Bool(dump_default=False)


messageList = []


@routes.post("/v1/execute")
async def execute(request):
    (d, r) = await run_validation(ExecuteSchema, request)
    if r is not None:
        return r

    print("Executing request", d["client_id"])

    response = web.StreamResponse()
    response.enable_chunked_encoding()
    # Prepare the response headers
    response.headers["Content-Type"] = "text/plain"

    await response.prepare(request)

    executed = set()
    outputs = {}
    outputs_ui = {}

    async def callback(data: dict):
        print(data)
        data_str = json.dumps(data) + "\n"
        await response.write(data_str.encode("utf-8"))

    # def hook(value, total, preview_image):
    #     throw_exception_if_processing_interrupted()
    #     print(f"Progress Hook: {value}/{total}")
    #     asyncio.create_task(
    #         callback(
    #             {
    #                 "status": "loading",
    #                 "type": "ksampler",
    #                 "value": value,
    #                 "total": total,
    #             }
    #         )
    #     )
    #     # if preview_image is not None:
    #     #     pass

    # set_progress_bar_global_hook(hook)

    client_id = d["client_id"]
    try:
        with torch.inference_mode():
            cleanup_models()
            await cd_recursive_execute_async(
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
        print("Done executing")
    except Exception as e:
        print(f"An error occurred: {e}")
        # Handle the error as needed...
    finally:
        try:
            await response.write_eof()
        except Exception as e:
            # Handle the error...
            print(f"Error when closing response: {e}")
    return response


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
    web.run_app(app, host=args.listen, port=args.port, keepalive_timeout=60)


if __name__ == "__main__":
    main()
