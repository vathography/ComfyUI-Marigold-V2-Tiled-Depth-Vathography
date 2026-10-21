import os
import time

import numpy as np
import torch

import comfy.model_management
import comfy.utils

from . import native_backend as NB
from . import tiled_core as TC

TILE = 1536
MIN_VRAM_TEXT = ("an estimated minimum is about 8 GB of GPU memory plus about 24 GB of system RAM (small cards stream "
                 "the model: similar result, slower)")


def image_to_bgr_u8(image):
    x = image[0, :, :, :3].detach().cpu().float().numpy()
    u8 = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(u8[:, :, ::-1])


def run_pipeline(photo, infer, tile=TILE, log=print, progress=None, merge_device="cpu", m1_budget=TC.M1_BUDGET_MP):
    H, W = photo.shape[:2]
    info = dict(canvas=[W, H], tile=tile)

    mw, mh, mp = TC.pin_for16(W, H, m1_budget)
    if mw is None:
        raise ValueError(f"{W}x{H}: no global grid fits the {m1_budget} MP budget")
    m1_native = TC.build_m1_input(photo, mw, mh, TC.load_decoy(TC.M1_DECOY_SCALE))
    iw, ih = m1_native.shape[1], m1_native.shape[0]
    assert iw % 16 == 0 and ih % 16 == 0, (iw, ih)
    info.update(m1_grid=[mw, mh], m1_input=[iw, ih], m1_mp=round(mp, 2))
    log(f"[tiled] photo {W}x{H} | global {mw}x{mh} + {TC.M1_STRIP}-px strip = {iw}x{ih} ({mp:.2f} MP, /16 grid)")
    hand_h = TC.load_decoy(TC.M1_DECOY_SCALE)["obj"].shape[0]
    if hand_h + TC.DECOY_BOTTOM > mh:
        log(f"[tiled] wide photo: the global pass's hand is scaled down to fit ({hand_h} -> {mh - 2 * TC.DECOY_BOTTOM} px tall)")

    single = W < tile or H < tile
    if single:
        plan, inputs = None, [m1_native]
        info["mode"] = f"single pass (photo smaller than one {tile}px tile)"
        log(f"[tiled] {info['mode']}: the global pass alone, no tiles")
    else:
        plan = TC.make_plan(W, H, tile, TC.OVERLAP)
        tins = TC.build_tile_inputs(photo, plan, TC.load_decoy(TC.TILE_DECOY_SCALE))
        inputs = [m1_native] + tins
        del tins
        info["mode"] = f"tiled {plan['cols']}x{plan['rows']} = {len(plan['tiles'])} tiles"
        log(f"[tiled] plan: tile {tile} (+ {TC.TILE_STRIP}-px decoy strip) overlap min {TC.OVERLAP} -> {info['mode']} | "
            f"achieved overlap {plan['achieved_overlap_x']:.0f} / {plan['achieved_overlap_y']:.0f} px")

    t0 = time.time()
    preds = infer(inputs, progress)
    info["infer_s"] = round(time.time() - t0, 1)
    assert len(preds) == len(inputs), (len(preds), len(inputs))
    del inputs
    dump = os.environ.get("VATHOGRAPHY_TILED_DUMP")         # diagnostic only: raw predictions as .npy, nothing else changes
    if dump:
        os.makedirs(dump, exist_ok=True)
        names = ["m1"] + ([t["name"] for t in plan["tiles"]] if plan else [])
        for nm, p in zip(names, preds):
            np.save(os.path.join(dump, nm + ".npy"), p)
        log(f"[tiled] DUMP: {len(names)} raw predictions -> {dump}")

    # ---- the global: strip cropped, resized to the canvas
    p1 = preds[0]
    assert p1.shape == (ih, iw), (p1.shape, (ih, iw))
    m1 = TC.crop_strip(p1, mw)
    G = TC.global_from_m1(m1, W, H)
    info["m1_near_rail_pct"] = round(100 * float((m1 <= -TC.SAT_THR).mean()), 3)
    log(f"[tiled] global prediction {mw}x{mh} range {m1.min():.4f}..{m1.max():.4f} | near rail {info['m1_near_rail_pct']} % "
        f"-> resized to the photo {W}x{H}, range {G.min():.4f}..{G.max():.4f}")

    if single:
        S = G
    else:
        tp = [TC.crop_strip(p, tile) for p in preds[1:]]
        del preds
        Tm, rows = TC.mosaic(G, plan, tp, log=log)
        del tp
        S, minfo = TC.laplacian_merge(G, Tm, TC.RADIUS, merge_device)
        if dump:
            np.save(os.path.join(dump, "merged_r32.npy"), S)
        del Tm
        info["tiles_near_rail_max_pct"] = round(100 * max(r["near"] for r in rows), 3)
        info["affine_fallback"] = [f"{r['name']} ({r['steep_x_median']:.0f}x median)" for r in rows if r["affine_fallback"]]
        if info["affine_fallback"]:
            log(f"[tiled] steep-segment fallback (straight-line fit) on: {', '.join(info['affine_fallback'])}")
        log(f"[tiled] merge r{TC.RADIUS} on {merge_device}: {minfo['levels']} levels, residual {minfo['residual_px']} px | "
            f"range {S.min():.4f}..{S.max():.4f}")
    out, sinfo = TC.stretch_near_bright(S)
    info["stretch"] = dict(lo=round(sinfo["lo"], 5), hi=round(sinfo["hi"], 5))
    log(f"[tiled] stretch 0.5 / 99.5 %: {sinfo['lo']:.4f}..{sinfo['hi']:.4f} -> 0..1, near = bright | "
        f"clipped {100 * sinfo['clipped_low']:.2f} % / {100 * sinfo['clipped_high']:.2f} %")
    return out, info


class VathographyMarigoldTiledDepth:
    CATEGORY = "image/geometry estimation"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    OUTPUT_TOOLTIPS = ("Depth map, 0..1, near = bright, at the photo's size.",)
    DESCRIPTION = ("Tiled Marigold V2 depth: a global pass plus 1536-px tiles, merged so the global holds the shape "
                   "and the tiles add detail. Wire the Marigold V2 depth template's MODEL, VAE and CONDITIONING.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The photo (one image). Larger photos give more tiles and more detail."}),
                "model": ("MODEL", {"tooltip": "The Marigold V2 depth model AFTER 'ModelSamplingAuraFlow' "
                                               "(shift 1.73, sampling img_to_img_velocity), as in the template."}),
                "vae": ("VAE", {"tooltip": "The Marigold V2 depth VAE from the template."}),
                "conditioning": ("CONDITIONING", {"tooltip": "The Marigold V2 depth conditioning from the template "
                                                             "('ConditioningLoader')."}),
            },
        }

    def execute(self, image, model, vae, conditioning):
        log = lambda *a: print(*a, flush=True)
        T0 = time.time()
        log(f"[tiled] ENTRY image {tuple(image.shape)} {image.dtype} | tile {TILE}")
        if image.shape[0] != 1:
            raise ValueError(f"one image at a time: got a batch of {image.shape[0]}. Run the images one by one "
                             "(a batch would be stretched image by image).")
        sampling = NB.check_model(model)
        warnings = []
        if comfy.model_management.intermediate_dtype() == torch.float16:
            warnings.append("ComfyUI runs with --fp16-intermediates: the decoded predictions are fp16")
        photo = image_to_bgr_u8(image)
        log(f"[tiled] model sampling {sampling} | photo {photo.shape[1]}x{photo.shape[0]} -> 8-bit BGR")

        pbar = comfy.utils.ProgressBar(1)

        def progress(done, total):
            pbar.update_absolute(done, total)

        binfo = {}

        def infer(inputs, prog):
            try:
                preds, bi = NB.run(model, vae, conditioning, inputs, log=log, progress=prog)
            except comfy.model_management.OOM_EXCEPTION as e:
                comfy.model_management.soft_empty_cache()
                raise RuntimeError(
                    f"This GPU ran out of memory on one {TILE}-px tile ({str(e)[:120]}). Close other programs using the GPU "
                    f"and run again; {MIN_VRAM_TEXT}. Smaller tiles are not offered because they were judged clearly worse.") from e
            binfo.update(bi)
            return preds

        dev = comfy.model_management.get_torch_device()
        out, info = run_pipeline(photo, infer, TILE, log=log, progress=progress, merge_device=dev)
        notes = []
        if binfo.get("encode_fallback") or binfo.get("decode_fallback"):
            notes.append(f"the VAE ran tiled (not enough GPU memory) for {binfo['encode_fallback']} encode(s) / "
                         f"{binfo['decode_fallback']} decode(s): same result, slower run")
        depth = torch.from_numpy(out)[None, :, :, None].repeat(1, 1, 1, 3).to(comfy.model_management.intermediate_device())
        wall = time.time() - T0
        lines = [f"Marigold V2 Tiled Depth - Vathography — {info['mode']}",
                 f"photo {info['canvas'][0]}x{info['canvas'][1]} | tile {info['tile']} + decoy strip {TC.TILE_STRIP} | "
                 f"global {info['m1_grid'][0]}x{info['m1_grid'][1]} ({info['m1_mp']} MP)",
                 f"inference {info['infer_s']} s | total {wall:.1f} s",
                 f"stretch 0.5/99.5 %: {info['stretch']['lo']}..{info['stretch']['hi']} (log depth) -> 0..1, near = bright"]
        if "tiles_near_rail_max_pct" in info:
            lines.append(f"tiles: max near-rail {info['tiles_near_rail_max_pct']} % | "
                         f"global near-rail {info['m1_near_rail_pct']} %")
            lines.append("straight-line fit (steep segment) on: " + (", ".join(info["affine_fallback"]) or "none"))
        lines += [f"note: {n}" for n in notes]
        lines += [f"WARN: {w}" for w in warnings]
        log("[tiled] EXIT depth " + str(tuple(depth.shape)) + f" range {float(depth.min()):.3f}..{float(depth.max()):.3f}\n"
            + "\n".join("[tiled] " + ln for ln in lines))
        return (depth,)


NODE_CLASS_MAPPINGS = {"VathographyMarigoldTiledDepth": VathographyMarigoldTiledDepth}
NODE_DISPLAY_NAME_MAPPINGS = {"VathographyMarigoldTiledDepth": "Marigold V2 Tiled Depth - Vathography"}
