import math

import numpy as np
import torch


# ----------------------------------------------------------------------------- area resize (uint8)
def _area_down_weights(n_src, n_dst):
    """Shrinking (n_src >= n_dst): output pixel d averages the source interval [d*s, (d+1)*s), s = n_src / n_dst,
    each source pixel weighted by its overlap with that interval (OpenCV's INTER_AREA for scale >= 1)."""
    s = n_src / n_dst
    W = np.zeros((n_dst, n_src), np.float64)
    for d in range(n_dst):
        a, b = d * s, (d + 1) * s
        i0, i1 = int(math.floor(a)), min(int(math.ceil(b)), n_src)
        for i in range(i0, i1):
            w = min(b, i + 1) - max(a, i)
            if w > 0:
                W[d, i] += w
        W[d] /= s
    return W


def _area_up_weights(n_src, n_dst):
    """OpenCV's INTER_AREA rule when an axis is not shrunk: sx = floor(dx*scale), fx = (dx+1) - (sx+1)*inv computed in
    double and STORED AS FLOAT32, fx = 0 if fx <= 0 else fx - floor(fx) (float32); weights (1-fx, fx) on (sx, sx+1),
    indices clamped. The precision order matters: in plain double this misses OpenCV by up to 175 levels at ratios
    like 39/25, where fx lands on an exact integer."""
    inv = n_dst / n_src
    scale = 1.0 / inv
    W = np.zeros((n_dst, n_src), np.float64)
    for dx in range(n_dst):
        sx = int(math.floor(dx * scale))
        fx = np.float32((dx + 1) - (sx + 1) * inv)
        fx = np.float32(0.0) if fx <= 0 else np.float32(fx - np.float32(math.floor(fx)))
        a, b = min(max(sx, 0), n_src - 1), min(max(sx + 1, 0), n_src - 1)
        W[dx, a] += 1.0 - float(fx)
        W[dx, b] += float(fx)
    return W


def resize_area_u8(img, w, h):
    """uint8 (H, W, C) -> (h, w, C), = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) within 1 level.
    OpenCV uses true area averaging only when BOTH axes shrink; otherwise its area-mode linear rule on both axes."""
    assert img.dtype == np.uint8 and img.ndim == 3, (img.dtype, img.shape)
    H, W = img.shape[:2]
    shrink = W >= w and H >= h
    Wy = (_area_down_weights if shrink else _area_up_weights)(H, h).astype(np.float32)
    Wx = (_area_down_weights if shrink else _area_up_weights)(W, w).astype(np.float32)
    out = np.empty((h, w, img.shape[2]), np.uint8)
    for c in range(img.shape[2]):                          # per channel, float32: ~0.1 GB for a 33 MP photo
        ch = img[:, :, c].astype(np.float32)
        r = Wy @ ch @ Wx.T
        out[:, :, c] = np.clip(np.rint(r), 0, 255).astype(np.uint8)
    return out


def fit_decoy(obj, alpha, max_h):
    """Shrink the decoy (uint8 obj + float alpha) to at most max_h px tall, keeping its aspect (area resize).
    Returns the inputs unchanged when they already fit, so every photo where the decoy fits is unaffected."""
    oh, ow = obj.shape[:2]
    if oh <= max_h:
        return obj, alpha
    nh = max(1, int(max_h))
    nw = max(1, int(round(ow * nh / oh)))
    Wy = _area_down_weights(oh, nh)
    Wx = _area_down_weights(ow, nw)
    small = resize_area_u8(obj, nw, nh)
    a = (Wy @ alpha.astype(np.float64) @ Wx.T).astype(np.float32)
    return small, a


# ----------------------------------------------------------------------------- dilate (binary mask)
def dilate_square(mask, radius):
    """= cv2.dilate(mask, np.ones((2r+1, 2r+1))) on a 0/1 mask (OpenCV's dilate border never wins)."""
    from scipy.ndimage import maximum_filter                 # scipy ships with ComfyUI core
    return maximum_filter(mask, size=2 * radius + 1, mode="constant", cval=0)


# ----------------------------------------------------------------------------- bicubic upscale (float32 map)
def resize_cubic_f32(m, w, h):
    """float32 (H, W) -> (h, w), = cv2.resize(m, (w, h), interpolation=cv2.INTER_CUBIC) to ≈ 2e-7: same kernel
    (A = -0.75), same pixel-centre mapping, clamped borders. Computed in float64 on the CPU (float32 misses OpenCV by
    up to 2.6e-4 on real maps), then cast."""
    t = torch.from_numpy(np.ascontiguousarray(m, dtype=np.float64))[None, None]
    out = torch.nn.functional.interpolate(t, size=(h, w), mode="bicubic", align_corners=False)
    return out[0, 0].numpy().astype(np.float32)
