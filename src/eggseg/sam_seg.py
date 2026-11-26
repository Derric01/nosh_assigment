"""Task 1 - pseudo ground truth from SAM (official segment-anything package).

The dataset has no boxes/masks (classification folders), so SAM runs in
'segment everything' mode: a grid of point prompts -> many candidate masks.
Stage A (slow, CPU): generate candidates once and cache them  -> outputs/sam_raw/*.npz
Stage B (fast)      : keep only the egg-like candidates        -> outputs/sam_masks/*.png
Splitting the stages lets us tune the egg filter without re-running SAM.
"""
import numpy as np, cv2

# ---------------- Stage A: candidate generation ----------------
def load_generator(ckpt, model_type, points_per_side=32):
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    torch.set_num_threads(4)
    sam = sam_model_registry[model_type](checkpoint=ckpt).eval()
    return SamAutomaticMaskGenerator(sam, points_per_side=points_per_side,
                                     pred_iou_thresh=0.86, stability_score_thresh=0.92)


def save_raw(path, cands):
    segs = np.stack([c["segmentation"] for c in cands]) if cands else np.zeros((0, 1, 1), bool)
    np.savez_compressed(path, masks=np.packbits(segs, axis=-1), w=segs.shape[-1],
                        iou=[c["predicted_iou"] for c in cands],
                        stab=[c["stability_score"] for c in cands])


def load_raw(path):
    z = np.load(path)
    m = np.unpackbits(z["masks"], axis=-1)[..., : int(z["w"])].astype(bool)
    return m, z["iou"], z["stab"]


# ---------------- Stage B: egg-likeness filter ----------------
P = dict(min_area=0.004, max_area=0.75,   # fraction of image area
         min_solidity=0.93,               # area / convex-hull area  (eggs are convex)
         min_ellipse_iou=0.88,            # how well a fitted ellipse explains the mask
         max_axis_ratio=2.0,              # eggs are not long thin things
         max_sat=60, min_val=100,         # eggs here are white/cream: low mean HSV-S, high mean V (0-255)
         nms_iou=0.3, contain=0.8)


def _clean(m):
    """Keep largest connected component, fill holes."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n < 2:
        return None
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    m = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 1, cv2.FILLED)
    return out.astype(bool)


def egg_shape_stats(m):
    """solidity, ellipse IoU and axis ratio for a clean binary mask."""
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return 0, 0, 99
    hull = cv2.contourArea(cv2.convexHull(c))
    sol = cv2.contourArea(c) / max(hull, 1)
    el = cv2.fitEllipse(c)
    e = np.zeros(m.shape, np.uint8)
    cv2.ellipse(e, el, 1, -1)
    e = e.astype(bool)
    eiou = (e & m).sum() / max((e | m).sum(), 1)
    ax = max(el[1]) / max(min(el[1]), 1)
    return sol, eiou, ax


def select_eggs(masks, iou_pred, stab, p=P, rgb=None):
    """Return list of clean egg masks (largest first). `rgb` enables the colour prior."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV) if rgb is not None else None
    cands = []
    for m, q, s in zip(masks, iou_pred, stab):
        m = _clean(m)
        if m is None:
            continue
        a = m.mean()
        if not (p["min_area"] <= a <= p["max_area"]):
            continue
        sol, eiou, ax = egg_shape_stats(m)
        if sol < p["min_solidity"] or eiou < p["min_ellipse_iou"] or ax > p["max_axis_ratio"]:
            continue
        if hsv is not None and (hsv[..., 1][m].mean() > p["max_sat"] or hsv[..., 2][m].mean() < p["min_val"]):
            continue                          # skin / leaves / dark objects
        cands.append((eiou * sol * q, m))
    cands.sort(key=lambda t: -t[0])
    keep = []
    for sc, m in cands:                       # NMS: best-scoring first; drop overlaps / part-of / contains
        ok = True
        for k in keep:
            inter = (m & k).sum()
            if inter / max((m | k).sum(), 1) > p["nms_iou"] or inter / min(m.sum(), k.sum()) > p["contain"]:
                ok = False; break
        if ok:
            keep.append(m)
    keep.sort(key=lambda m: -m.sum())
    return keep


def to_instance_png(keep, shape):
    out = np.zeros(shape, np.uint8)
    for i, m in enumerate(keep, 1):
        out[m & (out == 0)] = i
    return out
