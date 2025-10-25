from .nodes_image_s3 import NODE_CLASS_MAPPINGS as s3_mappings, NODE_DISPLAY_NAME_MAPPINGS as s3_display
from .nodes_image_base64 import NODE_CLASS_MAPPINGS as base64_mappings, NODE_DISPLAY_NAME_MAPPINGS as base64_display

NODE_CLASS_MAPPINGS = {**s3_mappings, **base64_mappings}
NODE_DISPLAY_NAME_MAPPINGS = {**s3_display, **base64_display}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

