"""Task 2 - classical (no learning) egg segmentation.

One function, `segment(rgb, cfg)`, whose behaviour is fully described by the `cfg` dict.
Each experiment in experiments/ is just a different cfg, so the log is reproducible.

Pipeline (each stage can be switched on/off by cfg):
  1. feature map   : gray | v | whiteness (V - k*S) | lab_l          -> 'bright / white' is egg-like
  2. smoothing     : Gaussian blur, optional CLAHE (local contrast)
  3. threshold     : otsu | fixed | adaptive (local mean)
  4. morphology    : open (remove specks) then close (fill holes / glare)
  5. components    : connected components -> shape filter (area, solidity, ellipse fit)
  6. optional      : GrabCut refinement, watershed split of touching eggs
Candidate generators (cfg['cand']): 'thresh' (steps 1-5 above), 'sweep' (many thresholds, pick egg-shaped
components), 'edges' (Canny contours that close into egg-shaped loops).
Returns an instance-id image (0 = background), same convention as the SAM masks.
"""
import cv2, numpy as np
from .sam_seg import egg_shape_stats   # same shape statistics as used to pick SAM eggs

BASE = dict(
    feat="gray", blur=5, clahe=False,
    thr="otsu", thr_val=170, adapt_block=51, adapt_c=-8,
    open_k=5, close_k=9,
    min_area=0.004, max_area=0.75, min_solidity=0.0, min_ellipse_iou=0.0, max_axis_ratio=99,
    border_penalty=False, grabcut=False, watershed=False, keep_best=0,
    cand="thresh",
    bg_k=3, bg_border=24,                            # feat='bgdist': colour clusters of the border ring
    sweep_lo=40, sweep_hi=240, sweep_step=6,         # cand='sweep'
    canny_lo=30, canny_hi=90, edge_close=7,          # cand='edges'
    nms_iou=0.3, contain=0.8,
    gc_max_growth=0.0, gc_min_ellipse=0.0,           # sanity limits on the GrabCut result (0 = off)
    fallback=False, fb_solidity=0.80, fb_ellipse=0.70,   # if nothing passes, relax the filter and keep the best
    center_sigma=0.0,                                # >0: score *= gaussian of distance from image centre
    gc_grow=1.0, gc_shrink=0.35, gc_iters=5,         # grabcut=True: seeds from the candidate mask
    sweep_feats=None,                                # list of feature names to pool; default [feat]
    score="shape", contrast_w=1.0, ring=9,           # 'shape' or 'shape_contrast'
    area_w=0.0,                                      # score *= area_fraction ** area_w  (prefer whole egg over its highlight)
    max_sat=0,                                       # >0: reject candidates whose mean HSV-S exceeds this
)


def feature_map(rgb, cfg):
    if cfg["feat"] == "gray":
        f = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    elif cfg["feat"] == "v":
        f = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2]
    elif cfg["feat"] == "whiteness":            # bright AND unsaturated
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        f = np.clip(hsv[..., 2] - cfg.get("sat_w", 0.8) * hsv[..., 1], 0, 255).astype(np.uint8)
    elif cfg["feat"] == "bgdist":               # how different is each pixel from the colours seen on the border?
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        b = cfg["bg_border"]
        ring = np.ones(lab.shape[:2], bool); ring[b:-b, b:-b] = False
        px = lab[ring]
        _, _, cen = cv2.kmeans(px, cfg["bg_k"], None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                               3, cv2.KMEANS_PP_CENTERS)
        d = np.min([np.linalg.norm(lab - c, axis=2) for c in cen], axis=0)
        f = np.clip(d * 3, 0, 255).astype(np.uint8)
    elif cfg["feat"] == "lab_l":
        f = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
    else:
        raise ValueError(cfg["feat"])
    if cfg["clahe"]:
        f = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(f)
    if cfg["blur"]:
        k = cfg["blur"] | 1
        f = cv2.GaussianBlur(f, (k, k), 0)
    return f


def threshold(f, cfg):
    if cfg["thr"] == "otsu":
        _, m = cv2.threshold(f, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif cfg["thr"] == "fixed":
        _, m = cv2.threshold(f, cfg["thr_val"], 255, cv2.THRESH_BINARY)
    elif cfg["thr"] == "adaptive":
        m = cv2.adaptiveThreshold(f, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                  cfg["adapt_block"] | 1, cfg["adapt_c"])
    else:
        raise ValueError(cfg["thr"])
    return m > 0


def morph(m, cfg):
    m = m.astype(np.uint8)
    if cfg["open_k"]:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["open_k"],) * 2))
    if cfg["close_k"]:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["close_k"],) * 2))
    return m > 0


def components(m, cfg):
    """Connected components -> list of filled binary masks that pass the shape filter."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    H, W = m.shape
    out = []
    for k in range(1, n):
        a = st[k, cv2.CC_STAT_AREA] / (H * W)
        if not (cfg["min_area"] <= a <= cfg["max_area"]):
            continue
        c = (lab == k)
        cnts, _ = cv2.findContours(c.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros(m.shape, np.uint8); cv2.drawContours(filled, cnts, -1, 1, cv2.FILLED)
        c = filled > 0
        sol, eiou, ax = egg_shape_stats(c)
        if sol < cfg["min_solidity"] or eiou < cfg["min_ellipse_iou"] or ax > cfg["max_axis_ratio"]:
            continue
        out.append((eiou * sol, c))
    out.sort(key=lambda t: -t[0])
    if cfg["keep_best"]:
        out = out[: cfg["keep_best"]]
    return [c for _, c in out]


def _fill(mask_u8):
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask_u8); cv2.drawContours(out, cnts, -1, 1, cv2.FILLED)
    return out > 0


def _score(c, cfg, ctx=None):
    """egg-likeness of a filled binary mask, or None if it fails the shape/size filter.
    ctx = dict(lab=..., hsv=...) of the image, needed by the contrast term / colour gate."""
    a = c.mean()
    if not (cfg["min_area"] <= a <= cfg["max_area"]):
        return None
    sol, eiou, ax = egg_shape_stats(c)
    if sol < cfg["min_solidity"] or eiou < cfg["min_ellipse_iou"] or ax > cfg["max_axis_ratio"]:
        return None
    sc = eiou * sol * (a ** cfg["area_w"])
    if cfg["center_sigma"]:
        ys, xs = np.nonzero(c)
        d2 = ((xs.mean() / c.shape[1] - 0.5) ** 2 + (ys.mean() / c.shape[0] - 0.5) ** 2)
        sc *= np.exp(-d2 / (2 * cfg["center_sigma"] ** 2))
    if ctx is not None:
        if cfg["max_sat"] and ctx["hsv"][..., 1][c].mean() > cfg["max_sat"]:
            return None                                     # not white-ish
        if cfg["score"] == "shape_contrast":
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg["ring"] + 1,) * 2)
            ring = (cv2.dilate(c.astype(np.uint8), k) > 0) & ~c
            if ring.sum() > 0:
                de = np.linalg.norm(ctx["lab"][c].mean(0) - ctx["lab"][ring].mean(0))
                sc *= (min(de, 40) / 40) ** cfg["contrast_w"]   # 0 = invisible boundary, 1 = strong
    return sc


def nms(cands, cfg):
    """cands = [(score, mask)]. Greedy: best first, drop anything overlapping an already kept egg."""
    keep = []
    for sc, m in sorted(cands, key=lambda t: -t[0]):
        if all((m & k).sum() / max((m | k).sum(), 1) <= cfg["nms_iou"] and
               (m & k).sum() / max(min(m.sum(), k.sum()), 1) <= cfg["contain"] for k in keep):
            keep.append(m)
    return keep


def sweep_candidates(f, cfg, ctx=None):
    """Threshold at many levels; every connected component at every level is a candidate."""
    out = []
    for t in range(cfg["sweep_lo"], cfg["sweep_hi"], cfg["sweep_step"]):
        m = morph(f >= t, cfg).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for k in range(1, n):
            if not (cfg["min_area"] <= st[k, cv2.CC_STAT_AREA] / m.size <= cfg["max_area"]):
                continue
            c = _fill((lab == k).astype(np.uint8))
            sc = _score(c, cfg, ctx)
            if sc is not None:
                out.append((sc, c))
    return out


def edge_candidates(gray, cfg):
    """Canny edges -> close small gaps -> every closed loop (incl. inner ones) filled = a candidate."""
    e = cv2.Canny(gray, cfg["canny_lo"], cfg["canny_hi"])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["edge_close"],) * 2)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        if len(c) < 30 or not (cfg["min_area"] <= cv2.contourArea(c) / e.size <= cfg["max_area"]):
            continue
        m = np.zeros(e.shape, np.uint8); cv2.drawContours(m, [c], -1, 1, cv2.FILLED)
        m = m > 0
        sc = _score(m, cfg)
        if sc is not None:
            out.append((sc, m))
    return out


def grabcut_refine(rgb, m, cfg):
    """Let GrabCut (colour GMMs + graph cut) grow/trim a candidate to the real egg boundary.
    sure-FG = candidate eroded, sure-BG = everything farther than `gc_grow` radii away."""
    r = max(int(np.sqrt(m.sum() / np.pi)), 3)                     # equivalent radius of the candidate
    mu = m.astype(np.uint8)
    k = lambda d: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(int(d), 1) + 1,) * 2)
    gc = np.full(m.shape, cv2.GC_PR_BGD, np.uint8)
    gc[cv2.dilate(mu, k(cfg["gc_grow"] * r)) == 0] = cv2.GC_BGD
    gc[mu > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(mu, k(cfg["gc_shrink"] * r)) > 0] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, cfg["gc_iters"], cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return m
    out = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    if n < 2:
        return m
    out = _fill((lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8))
    if out.sum() < 0.3 * m.sum():                                    # collapsed result
        return m
    if cfg["gc_max_growth"] and out.sum() > cfg["gc_max_growth"] * m.sum():   # leaked into the background
        return m
    if cfg["gc_min_ellipse"] and egg_shape_stats(out)[1] < cfg["gc_min_ellipse"]:  # no longer egg-shaped
        return m
    return out


def _candidates(rgb, cfg):
    """Stage 1-5: generate egg candidates with the generator chosen by cfg['cand']."""
    f = feature_map(rgb, cfg)
    if cfg["cand"] == "thresh":
        return components(morph(threshold(f, cfg), cfg), cfg)
    if cfg["cand"] == "sweep":
        ctx = dict(lab=cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32), hsv=cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
        cands = []
        for ft in (cfg["sweep_feats"] or [cfg["feat"]]):    # pool candidates from several feature maps
            cands += sweep_candidates(feature_map(rgb, {**cfg, "feat": ft}), cfg, ctx)
        comps = nms(cands, cfg)
    elif cfg["cand"] == "edges":
        comps = nms(edge_candidates(f, cfg), cfg)
    else:
        raise ValueError(cfg["cand"])
    return comps[: cfg["keep_best"]] if cfg["keep_best"] else comps


def segment(rgb, cfg=None):
    cfg = {**BASE, **(cfg or {})}
    comps = _candidates(rgb, cfg)
    if not comps and cfg["fallback"]:        # dataset prior: every image has >= 1 egg -> relax and keep the best
        comps = _candidates(rgb, {**cfg, "min_solidity": cfg["fb_solidity"], "min_ellipse_iou": cfg["fb_ellipse"],
                                  "max_sat": 0, "keep_best": 1})
    if cfg["grabcut"]:
        comps = nms([(1.0, grabcut_refine(rgb, c, cfg)) for c in comps], cfg)
    comps.sort(key=lambda c: -c.sum())
    inst = np.zeros(rgb.shape[:2], np.uint8)
    for i, c in enumerate(comps, 1):
        inst[c & (inst == 0)] = i
    return inst
