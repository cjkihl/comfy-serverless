import random
from typing import List
from marshmallow import ValidationError
from aiohttp import web
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP, load_lora_for_models
from comfy.utils import load_torch_file

from nodes import (
    NODE_CLASS_MAPPINGS,
    CheckpointLoaderSimple,
    EmptyLatentImage,
    KSampler,
    SaveImage,
    VAEDecode,
)
from sl import img
import folder_paths


node_cache = {}


async def run_validation(schema_def, request):
    data = await request.json()
    schema = schema_def()
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

    d_copy = d.copy()
    d_copy.pop("image", None)
    print({"input": d_copy})
    return (d, None)


model_cache = {}


def load_model():
    l = folder_paths.get_filename_list("checkpoints")
    if not l:
        raise FileNotFoundError("No checkpoints found")
    n = l[0]
    print(n)
    global model_cache
    if n in model_cache:
        return model_cache[n]

    checkpoint_loader = CheckpointLoaderSimple()
    model, clip, vae = checkpoint_loader.load_checkpoint(n)
    model_cache[n] = (model, clip, vae)
    print("Checkpoint Loaded")
    return (model, clip, vae)


# Lora cache
lora_cache = {}


def load_loras(names: List[str] | None, model: ModelPatcher | None, clip: CLIP | None):
    global lora_cache
    if names is None:
        return (model, clip)

    model_with_loras = model
    clip_with_loras = clip
    for n in names:
        parts = n.split(":")
        lora_name = parts[0]
        strength_model = 1.0 if len(parts) < 2 else float(parts[1])
        strength_clip = 1.0 if len(parts) < 3 else float(parts[2])
        print("Applying lora", lora_name, strength_model, strength_clip)
        lora = lora_cache.get(lora_name)
        if lora is None:
            lora_path = folder_paths.get_full_path("loras", lora_name + ".safetensors")
            if lora_path is None:
                raise FileNotFoundError("Lora not found", lora_name)
            print("Loading lora", lora_name, strength_model, strength_clip)
            lora_cache[lora_name] = lora = load_torch_file(lora_path, safe_load=True)

        (m, c) = load_lora_for_models(
            model_with_loras, clip_with_loras, lora, strength_model, strength_clip
        )
        model_with_loras = m
        clip_with_loras = c

    return (model_with_loras, clip_with_loras)


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


def encode_clip(clip: CLIP | None, text: str):
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


def encode_clip_with_loras(model: ModelPatcher | None, clip: CLIP | None, text: str):
    class_def = NODE_CLASS_MAPPINGS["CLIPTextEncodeLoras"]
    obj = class_def()
    print("Encoding CLIP with Loras", text)
    (cond, model, clip) = getattr(obj, class_def.FUNCTION)(model, clip, text)
    print("CLIP Encoded")
    return (cond, model, clip)
    # tokens = clip.tokenize(text)
    # cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    #  return [[cond, {"pooled_output": pooled}]]

    # if clip_encoder is None:
    #     clip_encoder = CLIPTextEncode()
    # (cond,) = clip_encoder.encode(clip, text)
    # return cond


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
    return (decoded, [d["seed"]])


def restore_faces(
    image,
    model,
    clip,
    vae,
    positive,
    negative,
):
    # Load BBox Detector
    class_def = NODE_CLASS_MAPPINGS["UltralyticsDetectorProvider"]
    obj = class_def()
    (bbox, *_) = getattr(obj, class_def.FUNCTION)("bbox/face_yolov8m.pt")
    # Load Dace Detailer
    class_def = NODE_CLASS_MAPPINGS["FaceDetailer"]
    obj = class_def()
    seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
    (decoded, *_) = getattr(obj, class_def.FUNCTION)(
        image=image,
        model=model,
        clip=clip,
        vae=vae,
        bbox_detector=bbox,
        guide_size=256,
        guide_size_for=True,
        max_size=512,
        seed=seed,
        steps=40,
        cfg=5.0,
        sampler_name="dpmpp_2m",
        scheduler="karras",
        positive=positive,
        negative=negative,
        denoise=0.5,
        feather=5,
        noise_mask=True,
        force_inpaint=True,
        bbox_threshold=0.5,
        bbox_dilation=10,
        bbox_crop_factor=3.0,
        sam_detection_hint="center-1",
        sam_dilation=0,
        sam_threshold=0.93,
        sam_bbox_expansion=0,
        sam_mask_hint_threshold=0.7,
        sam_mask_hint_use_negative=False,
        drop_size=10,
        wildcard="",
        cycle=1,
    )
    return decoded


def upscale(model, d, positive, negative, vae):
    (upscale_model,) = load_upscaler()
    (latent, _) = img.base64_to_image(d["image"])
    if d["seed"] == -1:
        d["seed"] = random.randint(0, 0xFFFFFFFFFFFFFFFF)

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
        denoise=d["denoising_strength"],
        mode_type="Linear",
        tile_width=512,
        tile_height=512,
        mask_blur=8,
        tile_padding=32,
        seam_fix_mode="Band Pass",
        seam_fix_denoise=1.0,
        seam_fix_mask_blur=8.0,
        seam_fix_width=64,
        seam_fix_padding=16,
        force_uniform_tiles=False,
        tiled_decode=False,
    )
    print("Upscaling done")
    return (upscaled, [d["seed"]])
