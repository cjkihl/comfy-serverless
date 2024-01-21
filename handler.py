import torch
from comfy.model_management import cleanup_models
import comfy.options
from nodes import init_custom_nodes
from sl.sync_execute import cd_recursive_execute_sync

comfy.options.enable_args_parsing(False)

from aiohttp import web
from marshmallow import Schema, fields
from server import PromptServer
import runpod


class DummyPromptServer:
    client_id = None
    routes = web.RouteTableDef()

    def send_sync(self, event, data, sid=None):
        pass

    def add_on_prompt_handler(self, handler):
        pass


## Mock PromptServer so custom nodes will not crash
PromptServer.instance = DummyPromptServer()

init_custom_nodes()


class ExecuteSchema(Schema):
    prompt = fields.Dict(required=True)
    client_id = fields.Str(required=True)
    output = fields.Str(required=True)
    test = fields.Bool(dump_default=False)


object_storage = {}
schema = ExecuteSchema()


def handler(job):
    try:
        job_input = job["input"]

        executed = set()
        outputs = {}
        outputs_ui = {}

        print("Received request", job_input)
        d = schema.dump(schema.load(job_input))
        if d["test"] is True:
            yield d

        print("Executing request", d["client_id"])
        with torch.inference_mode():
            cleanup_models()
            for r in cd_recursive_execute_sync(
                prompt=d["prompt"],
                outputs=outputs,
                current_item=d["output"],
                extra_data={"client_id": d["client_id"]},
                executed=executed,
                prompt_id=0,
                outputs_ui=outputs_ui,
                object_storage=globals()["object_storage"],
            ):
                print("Yielding result", r)
                yield r
            print("Done executing")
    except Exception as e:
        print("Exception", e)
        yield {"status": "error", "message": str(e)}


# def test_handler(job):
#     for count in range(3):
#         result = f"This is the {count} generated output."
#         yield result


runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
