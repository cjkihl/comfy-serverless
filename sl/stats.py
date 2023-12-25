import os
import sys
from comfy import model_management
from nodes import NODE_CLASS_MAPPINGS


def get_stats():
    device = model_management.get_torch_device()
    device_name = model_management.get_torch_device_name(device)
    vram_total, torch_vram_total = model_management.get_total_memory(
        device, torch_total_too=True
    )
    vram_free, torch_vram_free = model_management.get_free_memory(
        device, torch_free_too=True
    )
    system_stats = {
        "system": {
            "os": os.name,
            "python_version": sys.version,
            "embedded_python": os.path.split(os.path.split(sys.executable)[0])[1]
            == "python_embeded",
        },
        "devices": [
            {
                "name": device_name,
                "type": device.type,
                "index": device.index,
                "vram_total": vram_total,
                "vram_free": vram_free,
                "torch_vram_total": torch_vram_total,
                "torch_vram_free": torch_vram_free,
            }
        ],
        "nodes": list(NODE_CLASS_MAPPINGS.keys()),
    }
    return system_stats
