import math
import os

import numpy as np
import torch

try:
    from . import imageops as IO                           # inside the pack
except ImportError:
    import imageops as IO

try:
    from kornia.geometry.transform import pyrdown, pyrup
except ImportError as e:
    raise ImportError(f"[tiled_core] kornia is required: {e}")

SAT_THR = 0.99                 # |p| >= this = a rail pixel (both rails), excluded from the fit
SAT_DILATE_PX = 8              # fit-exclusion dilation radius
FIT_CENTRAL = 0.80             # central fraction of the tile used for the fit
QUANTILE_KNOTS = 129
KNOT_Q = 0.005                 # end knots at the 0.5 / 99.5 % quantiles
OVERLAP = 512
RADIUS = 32
TILE_STRIP = 512               # decoy strip on the right of every tile
TILE_DECOY_SCALE = 3.5
M1_STRIP = 384                 # decoy strip on the right of the M1 (global) input
M1_DECOY_SCALE = 2.5
M1_BUDGET_MP = 3.2
M1_MAX_LONG_EDGE = 2048        # largest long edge tried for the global grid
DECOY_BOTTOM = 40              # px between the hand and the bottom of the strip

# What differs between the two Marigold V2 depth models. Only the global pass changes: the see-through model
# (Log-layered) reads glass better with a narrow bare strip (no hand) and a global long edge of at most 1920 px;
# a bigger global made its glass worse. The tiles are the same for both.
RECIPES = {
    "depth": dict(m1_strip=M1_STRIP, m1_decoy_scale=M1_DECOY_SCALE, m1_max_long_edge=M1_MAX_LONG_EDGE),
    "see-through": dict(m1_strip=64, m1_decoy_scale=0.0, m1_max_long_edge=1920),
}
STRETCH_LO, STRETCH_HI = 0.5, 99.5
STEEP_X_MEDIAN = 68.0             # steepest segment / median segment slope at or above which a tile uses a straight-line fit


def _log(log, msg):
    if log is not None:
        log(msg)


# ----------------------------------------------------------------------------- geometry


def pin_for16(w, h, budget_mp=M1_BUDGET_MP, strip=M1_STRIP, max_long_edge=M1_MAX_LONG_EDGE):
    budget = budget_mp * 1e6
    r16 = lambda v: max(16, int(round(v / 16.0)) * 16)
    for long_edge in range(max_long_edge, 255, -16):
        if h >= w:
            mh, mw = long_edge, r16(long_edge * w / h)
        else:
            mw, mh = long_edge, r16(long_edge * h / w)
        if (mw + strip) * mh <= budget:
            return mw, mh, (mw + strip) * mh / 1e6
    return None, None, None


def plan_axis(length, tile, overlap):
    if tile >= length:
        return [0], 0.0
    n = int(math.ceil((length - overlap) / float(tile - overlap)))
    n = max(n, 2)
    stride = (length - tile) / float(n - 1)
    starts = [int(round(i * stride)) for i in range(n)]
    return starts, float(tile - stride)


def make_plan(W, H, tile, overlap=OVERLAP):
    xs, ovx = plan_axis(W, tile, overlap)
    ys, ovy = plan_axis(H, tile, overlap)
    tiles = [dict(name=f"tile_r{j}_c{i}", i=i, j=j, x0=x, y0=y, x1=x + tile, y1=y + tile)
             for j, y in enumerate(ys) for i, x in enumerate(xs)]
    return dict(canvas=[W, H], tile=tile, overlap_widget=overlap, cols=len(xs), rows=len(ys),
                starts_x=xs, starts_y=ys, achieved_overlap_x=ovx, achieved_overlap_y=ovy, tiles=tiles)


# ----------------------------------------------------------------------------- decoy + inputs
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def load_decoy(scale):
    p = os.path.join(ASSETS, f"decoy_hand_x{scale:g}.npz")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"decoy asset missing: {p}")
    z = np.load(p)
    return dict(obj=z["obj"], alpha=z["alpha"], scale=float(z["scale"]))


def add_decoy_strip(content, decoy, strip_px):
    H = content.shape[0]
    s_ = np.full((H, strip_px, 3), tuple(int(v) for v in np.median(content.reshape(-1, 3), axis=0)), np.uint8)
    if decoy is not None:
        obj, alpha = decoy["obj"], decoy["alpha"]
        if obj.shape[0] + DECOY_BOTTOM > H:                    # very wide photos: a short input, the hand is scaled down to fit
            obj, alpha = IO.fit_decoy(obj, alpha, H - 2 * DECOY_BOTTOM)
        oh, ow = obj.shape[:2]
        assert ow <= strip_px and oh + DECOY_BOTTOM <= H, f"decoy {ow}x{oh} does not fit a {strip_px}-px strip on a {H}-px input"
        x0 = (strip_px - ow) // 2; y0 = H - oh - DECOY_BOTTOM
        roi = s_[y0:y0 + oh, x0:x0 + ow].astype(np.float32)
        s_[y0:y0 + oh, x0:x0 + ow] = (alpha[..., None] * obj.astype(np.float32) + (1 - alpha[..., None]) * roi).astype(np.uint8)
    return np.concatenate([content, s_], axis=1)


def build_m1_input(photo_bgr, mw, mh, decoy, strip_px=M1_STRIP):
    m1_in = IO.resize_area_u8(photo_bgr, mw, mh)
    return add_decoy_strip(m1_in, decoy, strip_px)


def build_tile_inputs(photo_bgr, plan, decoy, strip_px=TILE_STRIP):
    out = []
    for t in plan["tiles"]:
        content = photo_bgr[t["y0"]:t["y1"], t["x0"]:t["x1"]]
        out.append(add_decoy_strip(content, decoy, strip_px) if strip_px > 0 else content)
    return out


def crop_strip(pred, content_w):
    """Drop the decoy strip from a prediction before anything reads it."""
    return np.ascontiguousarray(pred[:, :content_w]).astype(np.float32)


# ----------------------------------------------------------------------------- fit + weights
def quantile_map(tv, gv, knot_q=KNOT_Q):
    q = np.linspace(knot_q, 1.0 - knot_q, QUANTILE_KNOTS)
    kt = np.quantile(tv, q); kg = np.quantile(gv, q)
    kt = np.maximum.accumulate(kt); kg = np.maximum.accumulate(kg)
    keep = np.concatenate([[True], np.diff(kt) > 1e-6])
    kt, kg = kt[keep], kg[keep]
    n = len(kt); m = n // 2
    sl = lambda i0, i1: float((kg[i1] - kg[i0]) / max(kt[i1] - kt[i0], 1e-6))
    return kt, kg, sl(max(m - 8, 0), min(m + 8, n - 1)), sl(0, min(4, n - 1)), sl(max(n - 5, 0), n - 1)


def apply_quantile_map(p, kt, kg):
    out = np.interp(p, kt, kg).astype(np.float32)
    lo = p < kt[0]; hi = p > kt[-1]
    if lo.any():
        s = (kg[1] - kg[0]) / max(kt[1] - kt[0], 1e-6); out[lo] = kg[0] + s * (p[lo] - kt[0])
    if hi.any():
        s = (kg[-1] - kg[-2]) / max(kt[-1] - kt[-2], 1e-6); out[hi] = kg[-1] + s * (p[hi] - kt[-1])
    return out


def huber_affine(p, g, valid):
    from scipy.optimize import least_squares            # scipy ships with ComfyUI core
    tv, gv = p[valid][::7], g[valid][::7]
    a0, b0 = np.polyfit(tv, gv, 1)
    res = least_squares(lambda x: x[0] * tv + x[1] - gv, x0=[a0, b0], loss="huber", f_scale=0.05)
    return float(res.x[0]), float(res.x[1])


def fit_tile(p, g, tile, steep_x_median=STEEP_X_MEDIAN):
    """One tile's prediction aligned onto the global's values. Returns (aligned, stats).
    steep_x_median > 0: a tile whose steepest map segment is >= steep_x_median x the median segment slope uses the
    affine fit."""
    cx0 = int(tile * (1 - FIT_CENTRAL) / 2); cx1 = tile - cx0
    sat = (np.abs(p) >= SAT_THR).astype(np.uint8)
    sat_d = IO.dilate_square(sat, SAT_DILATE_PX).astype(bool)
    valid = ~sat_d
    valid[:cx0, :] = False; valid[cx1:, :] = False; valid[:, :cx0] = False; valid[:, cx1:] = False
    kt, kg, s_mid, s_lo, s_hi = quantile_map(p[valid], g[valid], KNOT_Q)
    seg = np.sort(np.diff(kg) / np.maximum(np.diff(kt), 1e-6))
    ratio = float(seg[-1] / max(float(np.median(seg)), 1e-6)) if seg.size > 1 else 1.0     # steepest / median
    flagged = steep_x_median > 0 and ratio >= steep_x_median
    if flagged:
        fa, fb = huber_affine(p, g, valid)
        aligned = (fa * p + fb).astype(np.float32)
    else:
        aligned = apply_quantile_map(p, kt, kg)
    stats = dict(near=float((p <= -SAT_THR).mean()), far=float((p >= SAT_THR).mean()), n_fit=int(valid.sum()),
                 slope_mid=s_mid, slope_lo=s_lo, slope_hi=s_hi, steep_max=float(seg[-1]), steep_x_median=ratio,
                 affine_fallback=flagged, rms=float(np.sqrt(np.mean((aligned[valid] - g[valid]) ** 2))))
    return aligned, stats


def ramp_weights(tile, has_prev, has_next, ov_prev, ov_next):
    w = np.ones(tile, np.float32)
    if has_prev and ov_prev > 1:
        n = int(round(ov_prev)); t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        w[:n] = 0.5 - 0.5 * np.cos(np.pi * t)
    if has_next and ov_next > 1:
        n = int(round(ov_next)); t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        w[tile - n:] = np.minimum(w[tile - n:], 0.5 + 0.5 * np.cos(np.pi * t))
    return w


# ----------------------------------------------------------------------------- global, mosaic, merge, stretch
def global_from_m1(m1, W, H):
    """G = the M1 prediction (strip already cropped) resized to the canvas, cubic (= OpenCV INTER_CUBIC)."""
    return IO.resize_cubic_f32(m1.astype(np.float32), W, H)


def mosaic(G, plan, tile_preds, log=None, steep_x_median=STEEP_X_MEDIAN):
    """Fit every tile onto G and blend: T = (sum w*aligned + max(0, 1 - sum w)*G) / max(sum w, 1).
    tile_preds = strip-cropped predictions in plan order. Returns (Tm, per-tile stats)."""
    H, W = G.shape
    tile = plan["tile"]
    num = np.zeros((H, W), np.float32); den = np.zeros_like(num)
    rows = []
    for t, p in zip(plan["tiles"], tile_preds):
        assert p.shape == (tile, tile), (t["name"], p.shape)
        g = G[t["y0"]:t["y1"], t["x0"]:t["x1"]]
        aligned, st = fit_tile(p, g, tile, steep_x_median)
        i, j = t["i"], t["j"]
        wx = ramp_weights(tile, i > 0, i < plan["cols"] - 1, plan["achieved_overlap_x"], plan["achieved_overlap_x"])
        wy = ramp_weights(tile, j > 0, j < plan["rows"] - 1, plan["achieved_overlap_y"], plan["achieved_overlap_y"])
        wr = wy[:, None] * wx[None, :]
        w = wr
        sl = (slice(t["y0"], t["y1"]), slice(t["x0"], t["x1"]))
        num[sl] += w * aligned
        den[sl] += w
        st["name"] = t["name"]; rows.append(st)
        _log(log, f"[tiled_core] {t['name']:<12} near {100 * st['near']:6.2f} % far {100 * st['far']:6.2f} % | "
                  f"q-slope mid {st['slope_mid']:.4f} lo {st['slope_lo']:.4f} hi {st['slope_hi']:.4f} | steepest {st['steep_max']:.1f} "
                  f"({st['steep_x_median']:.1f}x median){' -> AFFINE FALLBACK' if st['affine_fallback'] else ''} | rms {st['rms']:.4f} | n_fit {st['n_fit']}")
    g_share = np.clip(1.0 - den, 0.0, 1.0)
    Tm = ((num + g_share * G) / np.maximum(den, 1.0)).astype(np.float32)
    _log(log, f"[tiled_core] mosaic {W}x{H}: global share > 50 % on {100 * float((g_share > 0.5).mean()):.2f} % | "
              f"range {Tm.min():.4f}..{Tm.max():.4f} | mosaic-vs-global rms {float(np.sqrt(np.mean((Tm - G) ** 2))):.4f}")
    return Tm, rows


def laplacian_merge(G, Tm, r=RADIUS, device="cpu"):
    Gt = torch.from_numpy(G)[None, None].to(device)
    Tt = torch.from_numpy(Tm)[None, None].to(device)
    n, _, _ = derive_levels(r, Tm.shape[0], Tm.shape[1])
    div = 2 ** n
    Gp, hw = pad_to_divisible(Gt, div); Tp, _ = pad_to_divisible(Tt, div)
    gG = gaussian_pyramid(Gp, n); gT = gaussian_pyramid(Tp, n)
    outp = collapse(laplacian_from_gaussian(gT), gG[n])
    S = crop_back(outp, hw)[0, 0].detach().cpu().numpy().astype(np.float32)
    return S, dict(levels=n, residual_px=div)


def stretch_near_bright(S, lo_pct=STRETCH_LO, hi_pct=STRETCH_HI):
    lo, hi = (float(v) for v in np.percentile(S, [lo_pct, hi_pct]))
    if not hi > lo:
        raise ValueError(f"the depth map is flat between its {lo_pct} and {hi_pct} % values ({lo:.6f})")
    n = np.clip((S - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    info = dict(lo=lo, hi=hi, clipped_low=float((S < lo).mean()), clipped_high=float((S > hi).mean()))
    return (1.0 - n).astype(np.float32), info


# ----------------------------------------------------------------------------- vendored Laplacian primitives
MIN_COARSE_DIM = 8


def pad_to_divisible(x, div):
    h, w = x.shape[-2:]
    ph = (div - h % div) % div
    pw = (div - w % div) % div
    if ph == 0 and pw == 0:
        return x, (h, w)
    return torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect"), (h, w)


def crop_back(x, hw):
    h, w = hw
    return x[..., :h, :w]


def gaussian_pyramid(x, levels):
    g = [x]
    for _ in range(levels):
        g.append(pyrdown(g[-1]))
    return g


def laplacian_from_gaussian(g):
    return [g[i] - pyrup(g[i + 1]) for i in range(len(g) - 1)]


def collapse(laps, residual):
    x = residual
    for lap in reversed(laps):
        x = lap + pyrup(x)
    return x


def derive_levels(radius_px, h, w):
    n_desired = max(1, int(math.ceil(math.log2(max(radius_px, 1.0)))) + 1)
    n_cap = max(1, int(math.floor(math.log2(max(min(h, w) / MIN_COARSE_DIM, 2.0)))))
    n = min(n_desired, n_cap)
    return n, n_desired, n < n_desired
