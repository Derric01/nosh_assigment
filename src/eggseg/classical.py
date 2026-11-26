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
)


def feature_map(rgb, cfg):
    if cfg["feat"] == "gray":
        f = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    elif cfg["feat"] == "v":
        f = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2]
    elif cfg["feat"] == "whiteness":            # bright AND unsaturated
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        f = np.clip(hsv[..., 2] - cfg.get("sat_w", 0.8) * hsv[..., 1], 0, 255).astype(np.uint8)
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


def segment(rgb, cfg=None):
    cfg = {**BASE, **(cfg or {})}
    f = feature_map(rgb, cfg)
    m = morph(threshold(f, cfg), cfg)
    comps = components(m, cfg)
    comps.sort(key=lambda c: -c.sum())
    inst = np.zeros(rgb.shape[:2], np.uint8)
    for i, c in enumerate(comps, 1):
        inst[c & (inst == 0)] = i
    return inst
