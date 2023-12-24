from ast import Dict
import random
from typing import List
from marshmallow import ValidationError
from aiohttp import web
from torch import Tensor
from comfy.sd import load_lora_for_models
from comfy.utils import load_torch_file

from nodes import (
    NODE_CLASS_MAPPINGS,
    CLIPTextEncode,
    CheckpointLoaderSimple,
    EmptyLatentImage,
    KSampler,
    SaveImage,
    VAEDecode,
)
from sl import img
import folder_paths


nodes = []


async def validate(schema_def, request):
    data = await request.json()
    schema = schema_def()
    print({"input_data": data})
    try:
        d = schema.dump(schema.load(data))
        if d["test"] is True:
            return (None, web.json_response(d))
    except ValidationError as err:
        print(err.messages)
        return (
            None,
            web.json_response({"error": True, "message": err.messages}, status=400),
        )
    return (d, None)


def load_model():
    checkpoint_loader = CheckpointLoaderSimple()
    l = folder_paths.get_filename_list("checkpoints")
    if not l:
        raise FileNotFoundError("No checkpoints found")
    n = l[0]
    print(n)
    model, clip, vae = checkpoint_loader.load_checkpoint(n)
    print("Checkpoint Loaded")
    return (model, clip, vae)


# Lora cache
loaded_loras: Dict[str, Dict[str, Tensor]] = {}


def load_loras(names: List[str] | None, model, clip):
    for n in names:
        lora_name, strength = (n.split(":") + [1])[:2]
        lora = loaded_loras.get(lora_name)
        if lora is None:
            lora_path = folder_paths.get_full_path("loras", lora_name)
            print("Loading lora", lora_path)
            loaded_loras[lora_name] = lora = load_torch_file(lora_path, safe_load=True)
        model, clip = load_lora_for_models(model, clip, lora, strength, strength)
    return (model, clip)


def load_upscaler():
    class_def = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]
    obj = class_def()
    l = folder_paths.get_filename_list("upscale_models")
    if not l:
        raise FileNotFoundError("No upscale models found")
    n = l[0]
    print(n)
    (upscale_model,) = getattr(obj, class_def.FUNCTION)(n)
    print("Upscale Model Loaded")
    return (upscale_model,)


def encode_clip(clip, d):
    print("Encoding CLIP")
    clip_encoder = CLIPTextEncode()
    (positive,) = clip_encoder.encode(clip, d["prompt"])
    (negative,) = clip_encoder.encode(clip, d["negative_prompt"])
    print("CLIP Encoded")
    return (positive, negative)


def save_images(images):
    img_saver = SaveImage()
    img_saver.save_images(images, filename_prefix="CD")
    print("Images Saved")
    images = img.images_to_base64(images)
    print("Images Base 64 Encoded")
    return images


def sample(model, d, positive, negative, vae):
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
    return decoded


def upscale(model, d, positive, negative, vae):
    (upscale_model,) = load_upscaler()
    (latent, _) = img.base64_to_image(d["img"])
    print("Start Upscaling")
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
    print("Upscaling done")
    return upscaled
