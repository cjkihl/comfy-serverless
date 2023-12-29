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


class ExecuteSchema(Schema):
    prompt = fields.Dict(required=True)
    client_id = fields.Str(required=True)
    output = fields.Str(required=True)
    test = fields.Bool(dump_default=False)


object_storage = {}
schema = ExecuteSchema()

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.


async def generator_handler(job):
    """Handler function that will be used to process jobs."""
    job_input = job["input"]

    storage = globals()["object_storage"]

    executed = set()
    outputs = {}
    outputs_ui = {}

    d = schema.dump(schema.load(job_input))
    if d["test"] is True:
        yield d


runpod.serverless.start({"handler": generator_handler, "return_aggregate_stream": True})
