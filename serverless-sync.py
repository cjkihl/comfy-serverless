import json
import comfy.options
from comfy.utils import set_progress_bar_global_hook
from aiohttp import web

comfy.options.enable_args_parsing()

from flask import Flask, Response, request

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

routes = web.RouteTableDef()
app = Flask(__name__)


@app.route("/")
async def status():
    return web.json_response(stats.get_stats())


class ExecuteSchema(Schema):
    prompt = fields.Dict(required=True)
    client_id = fields.Str(required=True)
    output = fields.Str(required=True)
    test = fields.Bool(dump_default=False)


@app.post("/v1/execute", methods=["POST"])
def execute():
    data = request.json()
    schema = ExecuteSchema()
    d = schema.dump(schema.load(data))
    if d["test"] is True:
        return d

    print("Executing request", d["client_id"])

    executed = set()
    outputs = {}
    outputs_ui = {}

    async def callback(data: dict):
        print(data)
        data_str = json.dumps(data) + "\n"
        yield data_str.encode("utf-8")

    client_id = d["client_id"]

    with torch.inference_mode():
        cleanup_models()
        cd_recursive_execute_async(
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
    return Response(generate(), mimetype="text/plain")


def main():
    init_custom_nodes()
    app = web.Application()
    print("Starting server on port", args.port)

    app.add_routes(routes)
    app.middlewares.append(create_cors_middleware("*"))
    web.run_app(app, host=args.listen, port=args.port, keepalive_timeout=60)


if __name__ == "__main__":
    main()
