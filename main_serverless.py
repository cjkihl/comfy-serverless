import json
import logging

from comfy.cli_args import args, LatentPreviewMethod
from sl.callback import set_callback_hook
args.preview_method = LatentPreviewMethod.NoPreviews

from comfy.model_management import cleanup_models, throw_exception_if_processing_interrupted
from comfy.utils import set_progress_bar_global_hook
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Optional, Tuple
from queue import Queue
from threading import Thread
from threading import Lock
from PIL import Image
import base64
import io
from aiohttp import web
from server import PromptServer
from sl.execute import recursive_execute
import torch

class ThreadSafeDict:
    def __init__(self):
        self.dict = {}
        self.lock = Lock()

    def __setitem__(self, key, value):
        with self.lock:
            self.dict[key] = value
    
    def get(self, key, default=None):
        with self.lock:
            return self.dict.get(key, default)

    def set(self, key, value):
        with self.lock:
            self.dict[key] = value

    def delete(self, key):
        with self.lock:
            del self.dict[key]

object_storage = ThreadSafeDict()

routes = web.RouteTableDef()

class DummyPromptServer:
    client_id = None
    routes = routes

    def send_sync(self, event, data, sid=None):
        pass

    def add_on_prompt_handler(self, handler):
        pass

## Mock PromptServer so custom nodes will not crash
PromptServer.instance = DummyPromptServer() # type: ignore

from nodes import (
    init_external_custom_nodes,init_builtin_extra_nodes
)
init_builtin_extra_nodes()
if not args.disable_all_custom_nodes:
    init_external_custom_nodes(["ComfyUI-Manager"])
else:
    logging.info("Skipping loading of custom nodes")
    
app = FastAPI()

# The callback function
def callback(data):
    if data is not None:
        yield data + "\n"
    else:
        yield "End of stream\n"



class ExecuteSchema(BaseModel):
    prompt: Dict
    client_id: str
    output: str
    test: Optional[bool] = False

class CallbackData(BaseModel):
    status: str
    type: str
    node_id: str
    progress: int
    total: int
    preview_image: Optional[str] = None

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/v1/execute")
def main(data: ExecuteSchema):

    if(data.test):
        return data.model_dump_json()
    print({"input": data.model_dump_json()})
    print("Executing request", data.client_id)

    with torch.inference_mode():
        cleanup_models()

    return StreamingResponse(stream_generator(data), media_type="text/plain")

def stream_generator(input: ExecuteSchema):
    q = Queue()

    def callback(d: dict):
        q.put(d)

     # Start a new thread that runs the execute function
    thread = Thread(target=execute, args=(input, callback))
    thread.start()

    # End the stream when None is received
    while True:
        result = q.get()
        data_str = json.dumps(result)
        print(data_str)
        yield data_str + "\n"
        if result["status"] == "success" or result["status"] == "error":
            print("Done executing")
            break

def img_to_base64(image: Tuple[str, Image.Image], mime_type: str = "WEBP"):
        buffer = io.BytesIO()
        print(image[0])
        image[1].save(buffer, format=mime_type, quality=20)
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes)
        return img_base64.decode("utf-8")

def execute(data: ExecuteSchema, cb):
    executed = set()
    outputs = {}
    outputs_ui = {}

    # Hook for sampler progress which is different from the callback
    def hook(value, total, preview_image):
        throw_exception_if_processing_interrupted()
        cb({"status": "loading", "type": "Preview", "node_id": "sampler", "progress": value, "total": total, "previews": [img_to_base64(preview_image)] if preview_image else None })

    set_progress_bar_global_hook(hook)

    def callback_hook(args: object):
        cb(args)
    set_callback_hook(callback_hook)

    try:
        with torch.inference_mode():
            recursive_execute(cb,
                data.prompt,
                outputs,
                data.output,
                {"client_id": data.client_id},
                executed,
                0,
                outputs_ui,
                globals()["object_storage"])
        print("Done executing")
    except Exception as e:
        print(f"An error occurred: {e}")
        cb({"status": "error", "type": "error", "error": str(e)})
    
