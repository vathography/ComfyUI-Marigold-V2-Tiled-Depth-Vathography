import numpy as np
import torch

from . import relief as RL

# Colour pictures: the add-on's colour ramp maps depth to a 1-D path of colours. The path is fitted from the picture
# itself (a curve through its colours in OkLab), each pixel gets its position t along it (0..1, arc length), the
# relief runs on t exactly as on a grey picture, and each pixel moves along the curve by its change in t. What is
# not on the curve (local colour pushes) is kept as the pixel's offset from it.
CURVE_POINTS = 512             # polyline resolution of the fitted curve
CLUSTERS = 48                  # k-means centroids the first ordering is built from
MIN_CLUSTER_SHARE = 0.002      # centroids holding fewer of the sampled pixels than this are left out of the ordering
SAMPLE = 200_000               # pixels the curve is fitted on (fixed seed: same picture -> same curve)
REFINE_ITERS = 4
OFF_DIST = 0.02                # OkLab distance from the curve that counts a pixel as off it
OFF_SHARE_MAX = 0.10           # more pixels off the curve than this -> the brightness fallback
CHUNK = 1 << 18                # pixels per projection chunk

_M1 = torch.tensor([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005]], dtype=torch.float64)
_M2 = torch.tensor([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                    [0.0259040371, 0.7827717662, -0.8086757660]], dtype=torch.float64)
_M2I = torch.linalg.inv(_M2)
_M1I = torch.linalg.inv(_M1)
_LUMA = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float64)


def to_oklab(lin):
    lms = lin @ _M1.to(lin).T
    return torch.sign(lms) * lms.abs().pow(1.0 / 3.0) @ _M2.to(lin).T


def from_oklab(lab):
    lms = (lab @ _M2I.to(lab).T).pow(3)
    return lms @ _M1I.to(lab).T


def _kmeans(x, k, iters=20):
    """Deterministic k-means: farthest-point init from the darkest point."""
    c = [x[torch.argmin(x[:, 0])]]
    d = torch.cdist(x, c[0][None]).squeeze(1)
    for _ in range(k - 1):
        c.append(x[torch.argmax(d)])
        d = torch.minimum(d, torch.cdist(x, c[-1][None]).squeeze(1))
    c = torch.stack(c)
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        n = torch.bincount(a, minlength=k)
        s = torch.zeros_like(c).index_add_(0, a, x)
        keep = n > 0
        c = torch.where(keep[:, None], s / n.clamp(min=1)[:, None], c)
    n = torch.bincount(torch.cdist(x, c).argmin(1), minlength=k)
    return c[n >= max(1, int(MIN_CLUSTER_SHARE * len(x)))]          # tiny clusters = stray colours, not the ramp


def _main_path(c, k=5):
    """Order the centroids: the cheapest path, over a nearest-neighbour graph, between the two centroids farthest
    apart along that graph. Edge cost = squared distance, so the path follows chains of nearby centroids (where the
    picture's colours are) instead of jumping across a bend of the ramp."""
    p = c.cpu().numpy().astype(np.float64)
    n = len(p)
    D = np.linalg.norm(p[:, None] - p[None], axis=-1) ** 2
    G = np.full((n, n), np.inf)
    nn = np.argsort(D, axis=1)[:, 1:k + 1]
    for i in range(n):
        G[i, nn[i]] = D[i, nn[i]]
    inside, best, parent = np.zeros(n, bool), np.full(n, np.inf), np.full(n, -1)
    best[0] = 0.0
    for _ in range(n):                                      # minimum spanning tree edges keep the graph connected
        u = int(np.argmin(np.where(inside, np.inf, best)))
        inside[u] = True
        if parent[u] >= 0:
            G[u, parent[u]] = D[u, parent[u]]
        upd = (~inside) & (D[u] < best)
        best[upd], parent[upd] = D[u][upd], u
    G = np.minimum(G, G.T)
    np.fill_diagonal(G, 0.0)
    nxt = np.tile(np.arange(n), (n, 1))
    for m in range(n):                                      # Floyd-Warshall with next-hop table
        via = G[:, m:m + 1] + G[m:m + 1, :]
        better = via < G
        G = np.where(better, via, G)
        nxt = np.where(better, nxt[:, m:m + 1], nxt)
    a, b = np.unravel_index(int(np.argmax(G)), G.shape)
    path = [a]
    while path[-1] != b:
        path.append(int(nxt[path[-1], b]))
    return c[torch.tensor(path, device=c.device)]


def _resample(poly, n):
    seg = (poly[1:] - poly[:-1]).norm(dim=1)
    cum = torch.cat([seg.new_zeros(1), seg.cumsum(0)])
    if not cum[-1] > 0:
        return poly[:1].repeat(n, 1)
    s = torch.linspace(0, float(cum[-1]), n, dtype=poly.dtype, device=poly.device)
    i = torch.searchsorted(cum, s, right=True).clamp(1, len(poly) - 1)
    u = ((s - cum[i - 1]) / (cum[i] - cum[i - 1]).clamp(min=1e-12))[:, None]
    return poly[i - 1] + u * (poly[i] - poly[i - 1])


def _project(x, poly):
    """Each colour -> (t in 0..1 along the polyline by arc length, distance to it). Continuous in t: the colour is
    projected onto the segment before and after its nearest point and the closer projection is kept."""
    seg = poly[1:] - poly[:-1]
    slen = seg.norm(dim=1)
    cum = torch.cat([slen.new_zeros(1), slen.cumsum(0)])
    total = float(cum[-1])
    n = len(poly)
    ts, ds = [], []
    for k in range(0, len(x), CHUNK):
        xc = x[k:k + CHUNK]
        i = torch.cdist(xc, poly).argmin(1)
        best_t = best_d = None
        for j in ((i - 1).clamp(0, n - 2), i.clamp(0, n - 2)):
            a, v = poly[j], seg[j]
            u = (((xc - a) * v).sum(1) / (v * v).sum(1).clamp(min=1e-20)).clamp(0, 1)
            dist = (xc - (a + u[:, None] * v)).norm(dim=1)
            t = (cum[j] + u * slen[j]) / total
            if best_d is None:
                best_t, best_d = t, dist
            else:
                closer = dist < best_d
                best_t, best_d = torch.where(closer, t, best_t), torch.where(closer, dist, best_d)
        ts.append(best_t)
        ds.append(best_d)
    return torch.cat(ts), torch.cat(ds)


def _at(poly, t):
    """The curve's colour at t (0..1, arc length)."""
    seg = (poly[1:] - poly[:-1]).norm(dim=1)
    cum = torch.cat([seg.new_zeros(1), seg.cumsum(0)])
    s = t.clamp(0, 1) * cum[-1]
    i = torch.searchsorted(cum, s.contiguous(), right=True).clamp(1, len(poly) - 1)
    u = ((s - cum[i - 1]) / (cum[i] - cum[i - 1]).clamp(min=1e-12))[:, None]
    return poly[i - 1] + u * (poly[i] - poly[i - 1])


def _extend_ends(poly, x):
    """Refinement pulls the ends inward (an end point is the mean of the colours around it), which would clamp the
    most extreme colours to t = 0 / 1. Extend each end along its last segment until every colour nearest to it
    projects inside the curve."""
    a = torch.cdist(x, poly).argmin(1)
    tips = []
    for end, nxt in ((0, 1), (len(poly) - 1, len(poly) - 2)):
        v = poly[end] - poly[nxt]
        v = v / v.norm().clamp(min=1e-12)
        pts = x[a == end]
        over = float(((pts - poly[end]) @ v).max()) if len(pts) else 0.0
        tips.append((poly[end] + over * v)[None] if over > 0 else poly[end:end + 0])
    return torch.cat([tips[0], poly, tips[1]])


def fit_curve(lab):
    """lab [N, 3] -> polyline [CURVE_POINTS, 3] through the picture's colours, t = 0 at its darker end."""
    g = torch.Generator(device="cpu").manual_seed(0)
    x = lab[torch.randperm(len(lab), generator=g)[:SAMPLE].to(lab.device)] if len(lab) > SAMPLE else lab
    poly = _resample(_main_path(_kmeans(x, CLUSTERS)), CURVE_POINTS)
    for _ in range(REFINE_ITERS):
        a = torch.cdist(x, poly).argmin(1)
        n = torch.bincount(a, minlength=len(poly))
        s = torch.zeros_like(poly).index_add_(0, a, x)
        full = (n > 0).cpu().numpy()
        if full.sum() >= 2:                                 # empty stretches: a straight line between their neighbours
            mean = (s / n.clamp(min=1)[:, None]).cpu().numpy()
            idx = np.arange(len(full))
            mean = np.stack([np.interp(idx, idx[full], mean[full, k]) for k in range(3)], -1)
            poly = torch.from_numpy(mean).to(poly)
        poly = torch.cat([poly[:1], (poly[:-2] + 2 * poly[1:-1] + poly[2:]) / 4, poly[-1:]])
        poly = _resample(poly, CURVE_POINTS)
    poly = _resample(_extend_ends(poly, x), CURVE_POINTS)
    if poly[-1, 0] < poly[0, 0]:
        poly = poly.flip(0)
    return poly


def relief_colour(rgb, sigma_r, source, device="cpu", tick=None):
    """rgb float32 [H, W, 3] as loaded (PNG: sRGB values, EXR: linear) -> (rgb float32 [H, W, 3] in the same
    encoding, info). The relief follows the picture's colour ramp; far off one ramp -> relief on its brightness."""
    if source == RL.SOURCES[2]:
        raise ValueError("'Raw depth map' takes a grey depth map; for a colour export set 'input' to the export's "
                         "format (PNG or EXR)")
    if source not in RL.SOURCES:
        raise ValueError(f"unknown input type {source!r}; choose one of {RL.SOURCES}")
    H, W, _ = rgb.shape
    x = torch.from_numpy(np.ascontiguousarray(rgb)).to(device=device, dtype=torch.float64).reshape(-1, 3)
    lin = x if source == RL.SOURCES[0] else torch.from_numpy(RL.eotf_srgb_ext(rgb).reshape(-1, 3)).to(x)
    with torch.no_grad():
        lab = to_oklab(lin)
        poly = fit_curve(lab.float()).double()
        t, dist = _project(lab, poly)
    off_share = float((dist > OFF_DIST).double().mean())
    info = dict(path="curve", off_share=off_share, curve_len=float((poly[1:] - poly[:-1]).norm(dim=1).sum()))
    if off_share > OFF_SHARE_MAX:
        return _relief_brightness(lin, H, W, sigma_r, source, device, tick, info)
    t_np = t.reshape(H, W).float().cpu().numpy()
    r, rinfo = RL.reshape(t_np, sigma_r, device=device, tick=tick)
    r, rinfo["levels_kept"] = RL.keep_levels(r, t_np)
    info.update(rinfo)
    with torch.no_grad():
        tn = torch.from_numpy(r.reshape(-1)).to(lab)
        out_lin = from_oklab(lab + _at(poly, tn) - _at(poly, t))
    return _encode(out_lin, H, W, source, info)


def _relief_brightness(lin, H, W, sigma_r, source, device, tick, info):
    """Fallback: the relief on the picture's brightness (sRGB of linear luminance), colours scaled with it."""
    info["path"] = "brightness"
    Y = (lin @ _LUMA.to(lin)).clamp(min=0)
    disp = RL.oetf_srgb_ext(Y.reshape(H, W).cpu().numpy())
    r, rinfo = RL.reshape(disp, sigma_r, device=device, tick=tick)
    r, rinfo["levels_kept"] = RL.keep_levels(r, disp)
    info.update(rinfo)
    Yn = torch.from_numpy(RL.eotf_srgb_ext(r).reshape(-1)).to(lin)
    gain = Yn / Y.clamp(min=1e-4)
    return _encode(lin * gain[:, None], H, W, source, info)


def _encode(out_lin, H, W, source, info):
    out_lin = out_lin.reshape(H, W, 3).float().cpu().numpy()
    if source == RL.SOURCES[0]:
        info["out_of_range"] = float(((out_lin < 0) | (out_lin > 1)).any(-1).mean())
        return out_lin, info
    out = RL.oetf_srgb_ext(out_lin)
    info["out_of_range"] = float(((out < 0) | (out > 1)).any(-1).mean())
    return np.clip(out, 0.0, 1.0).astype(np.float32), info
