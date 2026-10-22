"""ComfyUI-Marigold-V2-Tiled-Depth-Vathography — tiled Marigold V2 depth for ComfyUI.
Copyright (C) 2026 Karim Joseph Nassar (Vathography). AGPL-3.0-only; output is free to use (see LICENSE)."""

from . import nf4_loader
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **nf4_loader.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **nf4_loader.NODE_DISPLAY_NAME_MAPPINGS}

print(f"[ComfyUI-Marigold-V2-Tiled-Depth-Vathography] loaded: {list(NODE_CLASS_MAPPINGS)}", flush=True)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
