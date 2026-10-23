<img width="1280" height="320" alt="A Vathography artwork cut into tiles" src="https://github.com/user-attachments/assets/1284012a-071c-4efd-8076-9442b8c4c066" />


# ComfyUI-Marigold-V2-Tiled-Depth-Vathography

Custom node pack for full-resolution depth maps from high-resolution photographs, estimated tile by tile, for the Vathography art practice. This repo is for the artistic application of the Marigold V2 model and is not affiliated with the original authors.

**The Vathography practice:** a photograph is distilled to a single element: the volume of the scene, creating a unique artwork, composed at capture, focused and shaped in post by the artist.

**Vathography tutorial:** https://youtu.be/DVYgwRaTKkA

**Vathography website:** https://vathography.com/

The depth maps generated with the nodes in this repo are meant to be shaped with the Vathography Blender add-on.

**Vathography Blender add-on:** https://vathography.gumroad.com/l/addon

---

## The nodes

**Marigold V2 Tiled Depth - Vathography** estimates the depth map of a high-resolution photograph by tiling it, keeping the photograph's full resolution. It works with two Marigold V2 models and recognises which one is loaded:

- **Base model:** the Marigold V2 depth model.
- **See-through model:** estimates the depth behind glass (windows, windshields, glass roofs) instead of stopping at the glass. Loaded with **Marigold V2 See-through Loader - Vathography**, on the 4-bit backbone the model was trained with.

<img width="1920" height="1080" alt="Photograph, base model and see-through model" src="https://github.com/user-attachments/assets/9cb1a8c0-1fac-4795-809c-dc7b7c5890b6" />

**Depth Relief Increase - Vathography** works like local contrast for depth maps: it makes the small shapes stand out (folds, faces, carvings) and shrinks the big depth jumps between objects, while keeping edges sharp. It uses a Local Laplacian filter. I recommend using this as an optional final step, after having shaped the depth map with the Blender add-on, to make the details and volumes pop. Load the shaped artwork you exported with Blender, set `input` to match, and save in the same format with Save Image (Advanced): EXR with input_color_space set to linear, PNG with bit_depth 16-bit. `level` 1 (subtle) to 6 (strongest). High levels can show terraces on smooth slopes.

<img width="1920" height="1080" alt="Base depth map and relief increase" src="https://github.com/user-attachments/assets/edb865f6-6586-4462-882e-c035e1198fa3" />

**Install:** search *Marigold V2 Tiled Depth Vathography* in ComfyUI Manager, or clone into `ComfyUI/custom_nodes/`:

```
git clone https://github.com/vathography/ComfyUI-Marigold-V2-Tiled-Depth-Vathography
```

No extra Python packages required. Needs ComfyUI v0.36.0 or newer.

**Models:** 2 models can be used, the base model and the see-through model.

| Model | Model files |
|---|---|
| Base model | [Comfy-Org/marigold-v2-0](https://huggingface.co/Comfy-Org/marigold-v2-0), installed following the official guide: https://docs.comfy.org/tutorials/utility/marigold-v2 |
| See-through model | [Vathography/marigold-v2-see-through-comfyui](https://huggingface.co/Vathography/marigold-v2-see-through-comfyui) (converted from Marigold V2, Apache-2.0), plus the depth conditioning from [Comfy-Org/marigold-v2-0](https://huggingface.co/Comfy-Org/marigold-v2-0) |

**Use:** open the base model or the see-through example workflow in `example_workflows/`, load a photograph, run.

**GPU:** Estimated minimum requirements for both models: 8 GB VRAM + 24 GB RAM.

**License:** AGPL-3.0-only. Free nodes; the depth maps they produce are free to use, for personal and commercial work. See `LICENSE`.

**What's new in 1.2.0:** the Depth Relief Increase node.

**What's new in 1.1.0:** see-through depth (the loader node, its example workflow and model files). The straight-line fit for steep tiles now starts at 68× instead of 75×, so a few depth maps can differ slightly from 1.0.0.

By Karim Joseph Nassar, developed with Claude Code.

---

## Marigold V2 Credit and Links

- **Marigold V2 ComfyUI official guide:** https://docs.comfy.org/tutorials/utility/marigold-v2
- **The ComfyUI implementation** — native Marigold V2 support by the ComfyUI team: [models for ComfyUI](https://huggingface.co/Comfy-Org/marigold-v2-0)
- **See-through model files for ComfyUI** — converted from Marigold V2 by Vathography, Apache-2.0: [Vathography/marigold-v2-see-through-comfyui](https://huggingface.co/Vathography/marigold-v2-see-through-comfyui)
- **The model** — Marigold V2 by huawei-bayerlab, Apache-2.0: [code](https://github.com/huawei-bayerlab/marigold-v2) · [weights](https://huggingface.co/huawei-bayerlab/marigold-v2-0)

Thank you to the Marigold V2 team: Igor Pavlovic, Thiemo Wandel, Anton Obukhov, Luca Bartolomei, Andrey Davydov, Fabio Tosi, Matteo Poggi, Sabine Süsstrunk, Dengxin Dai.
https://huggingface.co/spaces/huawei-bayerlab/marigold-v2-web
