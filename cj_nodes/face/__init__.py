from .nodes_cj_face import NODE_CLASS_MAPPINGS as cj_face_mappings, NODE_DISPLAY_NAME_MAPPINGS as cj_face_display
from .nodes_cj import NODE_CLASS_MAPPINGS as cj_mappings, NODE_DISPLAY_NAME_MAPPINGS as cj_display

NODE_CLASS_MAPPINGS = {**cj_face_mappings, **cj_mappings}
NODE_DISPLAY_NAME_MAPPINGS = {**cj_face_display, **cj_display}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

