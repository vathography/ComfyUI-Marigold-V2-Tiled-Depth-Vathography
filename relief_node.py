import time

import numpy as np
import torch

import comfy.model_management
import comfy.utils

from . import relief as RL

GREY_TOL = 1e-6                # max channel difference for an IMAGE to count as a grey depth map
WORK_BYTES_PER_PX = 48         # GPU working set of the filter, measured shape: about 12 float32 copies of the map


class VathographyDepthReliefIncrease:
    CATEGORY = "image/geometry estimation"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    OUTPUT_TOOLTIPS = ("The picture with its relief increased, same size. An EXR input comes out linear: save it with "
                       "Save Image (Advanced) as exr with input_color_space 'linear'. A PNG input comes out sRGB: save it "
                       "as png, bit_depth 16-bit. A raw depth map comes out stretched to 0..1.",)
    DESCRIPTION = ("Works like local contrast for depth maps: makes the small shapes stand out (folds, faces, carvings) "
                   "and shrinks the big depth jumps between objects, while keeping edges sharp (a Local Laplacian "
                   "filter). Meant as a final step on a depth map shaped with the Vathography Blender add-on: load its "
                   "PNG or EXR export and set 'input' to match. Blacks and whites of an export stay where they were.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("IMAGE", {"tooltip": "A grey picture: the Blender add-on's export (PNG or EXR, grey ramp) "
                                               "or a raw depth map. Colour images are refused."}),
                "input": (list(RL.SOURCES), {"default": RL.SOURCES[0],
                                             "tooltip": "What is loaded. 'Blender export, EXR (linear)': the add-on's "
                                                        "EXR. 'Blender export, PNG (sRGB)': the add-on's PNG (8 or 16-bit). "
                                                        "Both are processed as they look in Blender, with blacks and "
                                                        "whites kept. 'Raw depth map': a 0..1 depth map straight from "
                                                        "Marigold V2 Tiled Depth - Vathography, stretched to 0..1 "
                                                        "afterwards."}),
                "level": ("INT", {"default": 1, "min": 1, "max": 6, "step": 1,
                                  "tooltip": "Strength, 1 = subtle … 6 = strongest. Each level widens the depth range "
                                             "treated as 'small relief' (1: 0.015, 2: 0.03, 3: 0.05, 4: 0.07, 5: 0.09, "
                                             "6: 0.11, in the values of a 0..1 picture; the input is expected in "
                                             "0..1). High levels can show terraces on smooth slopes. Lower levels "
                                             "take longer (more samples)."}),
            },
        }

    def execute(self, depth, input, level):
        log = lambda *a: print(*a, flush=True)
        T0 = time.time()
        sigma_r = RL.LEVELS[int(level)]
        log(f"[relief] ENTRY depth {tuple(depth.shape)} {depth.dtype} range {float(depth.min()):.4f}.."
            f"{float(depth.max()):.4f} | input '{input}' | level {level} (sigma_r {sigma_r}, alpha {RL.ALPHA}, beta {RL.BETA}, "
            f"spacing sigma_r/{RL.SPACING_DIV:g}, floor {RL.FLOOR_ABS})")
        dev = comfy.model_management.get_torch_device()
        outs = []
        for b in range(depth.shape[0]):
            img = depth[b].float().cpu().numpy()
            if img.shape[-1] >= 3:
                spread = float(max(np.abs(img[..., 0] - img[..., 1]).max(), np.abs(img[..., 0] - img[..., 2]).max()))
                if spread > GREY_TOL:
                    raise ValueError(f"image {b} is a colour image (channels differ by up to {spread:.4f}): this node "
                                     "takes a grey depth map, such as the Marigold V2 Tiled Depth output.")
            d = np.ascontiguousarray(img[..., 0], dtype=np.float32)
            H, W = d.shape
            if not np.isfinite(d).all():
                raise ValueError(f"image {b} contains NaN or infinite values")
            comfy.model_management.free_memory(H * W * WORK_BYTES_PER_PX, dev)
            pbar = comfy.utils.ProgressBar(1)

            def tick(done, total):
                comfy.model_management.throw_exception_if_processing_interrupted()
                pbar.update_absolute(done, total)

            t = time.time()
            try:
                r, info = RL.relief(d, sigma_r, input, device=dev, tick=tick)
            except comfy.model_management.OOM_EXCEPTION as e:
                comfy.model_management.soft_empty_cache()
                raise RuntimeError(f"the GPU ran out of memory on a {W}x{H} map ({str(e)[:120]}); about "
                                   f"{H * W * WORK_BYTES_PER_PX / 2 ** 30:.1f} GB are needed. Close other programs "
                                   "using the GPU and run again.") from e
            comfy.model_management.soft_empty_cache()
            log(f"[relief] image {b}: {W}x{H} | in {info['in_range'][0]:.4f}..{info['in_range'][1]:.4f} | "
                f"{info['samples']} samples x {info['levels']} levels, spacing {info['spacing']:.6f}, floor "
                f"{info['floor']:.6f} | filtered {info['raw_range'][0]:.4f}..{info['raw_range'][1]:.4f} -> 0..1 | "
                f"{time.time() - t:.1f} s on {dev}")
            if "levels_kept" in info:
                lv = info["levels_kept"]
                log(f"[relief] image {b}: levels kept (display units): black {lv['in_lo']:.5f} / white {lv['in_hi']:.5f} "
                    f"<- relief {lv['got_lo']:.5f} / {lv['got_hi']:.5f} on {lv['n_lo']} / {lv['n_hi']} px; halo clipped "
                    f"{100 * lv['clipped']:.3f} % | out {float(r.min()):.4f}..{float(r.max()):.4f}")
            outs.append(torch.from_numpy(r))
        out = torch.stack(outs)[..., None].repeat(1, 1, 1, 3).to(comfy.model_management.intermediate_device())
        log(f"[relief] EXIT depth {tuple(out.shape)} range {float(out.min()):.3f}..{float(out.max()):.3f} | "
            f"total {time.time() - T0:.1f} s")
        return (out,)


NODE_CLASS_MAPPINGS = {"VathographyDepthReliefIncrease": VathographyDepthReliefIncrease}
NODE_DISPLAY_NAME_MAPPINGS = {"VathographyDepthReliefIncrease": "Depth Relief Increase - Vathography"}
