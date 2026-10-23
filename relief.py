import math

import numpy as np
import torch

from . import tiled_core as TC

# Relief increase = a fast Local Laplacian filter with beta on the depth (Aubry et al. 2014, Paris et al. 2011 remap):
# small relief is boosted (alpha < 1), large depth jumps between objects are compressed (beta < 1), then the result is
# stretched min/max back to 0..1. Settings chosen by eye on 0..1 depth maps.
ALPHA, BETA = 0.5, 0.4
SPACING_DIV = 48.0             # sample spacing = sigma_r / this (denser sampling = less banding on smooth gradients)
FLOOR_ABS = 0.001875           # boost floor in map units: below it the remap blends to identity (no banding cusp)
LEVELS = {1: 0.015, 2: 0.03, 3: 0.05, 4: 0.07, 5: 0.09, 6: 0.11}   # level -> sigma_r (map units)


def remap_beta(i, g, sigma_r, alpha, beta, floor=0.0):
    d = i - g
    ad = d.abs()
    small = sigma_r * (ad / sigma_r).pow(alpha)
    if alpha < 1.0 and floor > 0.0:
        t = ((ad - floor / 2.0) / (floor / 2.0)).clamp(0.0, 1.0)
        tau = t * t * (3.0 - 2.0 * t)
        small = tau * small + (1.0 - tau) * ad
    large = sigma_r + beta * (ad - sigma_r)
    return g + torch.sign(d) * torch.where(ad <= sigma_r, small, large)


def fast_llf_beta(L, alpha, beta, sigma_r, spacing_div, floor_steps, tick=None):
    h, w = L.shape[-2:]
    n = max(1, int(math.floor(math.log2(min(h, w) / TC.MIN_COARSE_DIM))))
    Lp, hw = TC.pad_to_divisible(L, 2 ** n)
    G = TC.gaussian_pyramid(Lp, n)
    lo, hi = float(Lp.min()), float(Lp.max())
    ns = int(math.ceil((hi - lo) / (sigma_r / spacing_div))) + 1
    step = (hi - lo) / (ns - 1)
    floor = step * floor_steps
    acc = [torch.zeros_like(t) for t in TC.laplacian_from_gaussian(G)]
    for j in range(ns):
        gam = lo + step * j
        gR = TC.gaussian_pyramid(remap_beta(Lp, gam, sigma_r, alpha, beta, floor), n)
        for l in range(n):
            hat = (1.0 - (G[l] - gam).abs() / step).clamp(0.0, 1.0)
            acc[l] = acc[l] + hat * (gR[l] - TC.pyrup(gR[l + 1]))
        if tick is not None:
            tick(j + 1, ns)
    res = G[-1].mean() + beta * (G[-1] - G[-1].mean())
    out = TC.collapse(acc, res)
    return TC.crop_back(out, hw), {"levels": n, "samples": ns, "spacing": step, "floor": floor}


def oetf_srgb_ext(a):
    """sRGB OETF (the Standard view on an sRGB display) on [0, 1], continued past it: the power branch above 1,
    mirrored below 0, so a value outside 0..1 is not clipped before the reshape."""
    a = np.asarray(a, dtype=np.float64)
    m = np.abs(a)
    y = np.where(m <= 0.0031308, 12.92 * m, 1.055 * np.power(m, 1.0 / 2.4) - 0.055)
    return (np.sign(a) * y).astype(np.float32)


def eotf_srgb_ext(v):
    """Inverse of `oetf_srgb_ext`."""
    v = np.asarray(v, dtype=np.float64)
    m = np.abs(v)
    y = np.where(m <= 0.04045, m / 12.92, np.power((m + 0.055) / 1.055, 2.4))
    return (np.sign(v) * y).astype(np.float32)


LEVEL_EPS = 1e-4          # display units: "the input's black" = pixels within this of its minimum (same for white)
LEVEL_MIN_SHARE = 1e-4    # fewer anchor pixels than this share of the frame -> the darkest / brightest 0.1 % instead


def keep_levels(r, disp):
    """Artistic pictures: blacks in stay blacks out, whites stay whites. The min/max stretch is set by a few edge-halo
    pixels, which would lift the black and pull the white down. Remap the stretched relief LINEARLY so its median over
    the input's own black pixels lands on the input's black and over its white pixels on its white; the halo beyond
    either end is clipped. Display units in and out."""
    lo_in, hi_in = float(disp.min()), float(disp.max())
    n = disp.size

    def anchor(mask, fallback):
        if int(mask.sum()) < max(1, int(LEVEL_MIN_SHARE * n)):
            mask = fallback
        return float(np.median(r[mask])), int(mask.sum())

    got_lo, n_lo = anchor(disp <= lo_in + LEVEL_EPS, disp <= np.percentile(disp, 0.1))
    got_hi, n_hi = anchor(disp >= hi_in - LEVEL_EPS, disp >= np.percentile(disp, 99.9))
    info = dict(in_lo=lo_in, in_hi=hi_in, got_lo=got_lo, got_hi=got_hi, n_lo=n_lo, n_hi=n_hi, clipped=0.0)
    if not got_hi - got_lo > 1e-6:                        # a flat picture: nothing to anchor on -> the stretch as is
        return r, info
    out = lo_in + (r.astype(np.float64) - got_lo) * ((hi_in - lo_in) / (got_hi - got_lo))
    info["clipped"] = float(((out < lo_in) | (out > hi_in)).mean())
    return np.clip(out, lo_in, hi_in).astype(np.float32), info


SOURCES = ("Blender export, EXR (linear)", "Blender export, PNG (sRGB)", "Raw depth map")


def relief(d, sigma_r, source, device="cpu", tick=None):
    """d float32 [H, W] as loaded -> (relief float32 [H, W] in the same encoding, info).
    EXR: through the sRGB curve, reshape, levels kept, curve undone -> linear again.
    PNG: the values are already the Standard view -> reshape + levels kept, output stays sRGB.
    Raw depth map: reshape + min/max stretch to 0..1."""
    if source == SOURCES[0]:
        disp = oetf_srgb_ext(d)
        r, info = reshape(disp, sigma_r, device=device, tick=tick)
        r, info["levels_kept"] = keep_levels(r, disp)
        return eotf_srgb_ext(r), info
    if source == SOURCES[1]:
        r, info = reshape(d, sigma_r, device=device, tick=tick)
        r, info["levels_kept"] = keep_levels(r, d)
        return r, info
    if source == SOURCES[2]:
        return reshape(d, sigma_r, device=device, tick=tick)
    raise ValueError(f"unknown input type {source!r}; choose one of {SOURCES}")


def reshape(d, sigma_r, device="cpu", tick=None):
    """d float32 [H, W] -> (relief float32 [H, W] in 0..1, info)."""
    lo_in, hi_in = float(d.min()), float(d.max())
    span = float(d.max() - d.min())          # computed in float32 (sets the sample count)
    if not span > 0.0:
        raise ValueError(f"the depth map is flat (every pixel = {lo_in:.6f}): there is no relief to increase")
    ns = int(np.ceil(span / (sigma_r / SPACING_DIV))) + 1
    fs = FLOOR_ABS / (span / (ns - 1))
    L = torch.from_numpy(np.ascontiguousarray(d, dtype=np.float32))[None, None].to(device)
    with torch.no_grad():
        o, info = fast_llf_beta(L, ALPHA, BETA, sigma_r, spacing_div=SPACING_DIV, floor_steps=fs, tick=tick)
    o = o[0, 0].cpu().numpy().astype(np.float32)
    del L
    lo, hi = float(o.min()), float(o.max())
    info.update(in_range=(lo_in, hi_in), raw_range=(lo, hi))
    return ((o - lo) / (hi - lo)).astype(np.float32), info
