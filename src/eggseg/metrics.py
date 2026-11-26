"""Segmentation metrics. GT = SAM masks (Task 1), pred = classical masks (Task 2).
Masks are instance-id images (0 = background). Three views of quality:
  semantic : all eggs merged into one foreground mask (IoU, Dice, precision, recall)
  boundary : Boundary-F1 within a pixel tolerance + HD95 (how well the *edges* line up)
  instance : eggs matched one-to-one (Hungarian, IoU >= 0.5) -> instance P/R/F1, count error
"""
import cv2, numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt


def semantic(gt, pr):
    g, p = gt > 0, pr > 0
    tp, fp, fn = (g & p).sum(), (~g & p).sum(), (g & ~p).sum()
    if g.sum() == 0 and p.sum() == 0:           # nothing to find, nothing found
        return dict(iou=1.0, dice=1.0, prec=1.0, rec=1.0, tp=0, fp=0, fn=0)
    iou = tp / max(tp + fp + fn, 1)
    return dict(iou=iou, dice=2 * tp / max(2 * tp + fp + fn, 1),
                prec=tp / max(tp + fp, 1), rec=tp / max(tp + fn, 1), tp=int(tp), fp=int(fp), fn=int(fn))


def _edge(m):
    m = m.astype(np.uint8)
    return m - cv2.erode(m, np.ones((3, 3), np.uint8)) > 0


def boundary(gt, pr, tol=3):
    """Boundary F1: a predicted edge pixel is correct if a GT edge pixel is within `tol` px
    (and vice-versa). HD95 = 95th percentile of edge-to-edge distances (both directions)."""
    g, p = _edge(gt > 0), _edge(pr > 0)
    if not g.any() and not p.any():
        return dict(bf=1.0, hd95=0.0)
    if not g.any() or not p.any():
        return dict(bf=0.0, hd95=np.nan)
    dg, dp = distance_transform_edt(~g), distance_transform_edt(~p)   # dist to nearest edge px
    prec, rec = (dg[p] <= tol).mean(), (dp[g] <= tol).mean()
    bf = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return dict(bf=bf, hd95=float(np.percentile(np.r_[dg[p], dp[g]], 95)))


def instance(gt, pr, thr=0.5):
    gi, pi = [i for i in np.unique(gt) if i], [i for i in np.unique(pr) if i]
    iou = np.zeros((len(gi), len(pi)))
    for a, i in enumerate(gi):
        for b, j in enumerate(pi):
            x, y = gt == i, pr == j
            iou[a, b] = (x & y).sum() / max((x | y).sum(), 1)
    tp, matched = 0, []
    if iou.size:
        r, c = linear_sum_assignment(-iou)
        for a, b in zip(r, c):
            if iou[a, b] >= thr:
                tp += 1; matched.append(iou[a, b])
    fp, fn = len(pi) - tp, len(gi) - tp
    return dict(n_gt=len(gi), n_pred=len(pi), inst_tp=tp, inst_fp=fp, inst_fn=fn,
                inst_prec=tp / max(len(pi), 1) if pi else (1.0 if not gi else 0.0),
                inst_rec=tp / max(len(gi), 1) if gi else (1.0 if not pi else 0.0),
                matched_iou=float(np.mean(matched)) if matched else np.nan)


def evaluate(gt, pr):
    out = {}
    out.update(semantic(gt, pr)); out.update(boundary(gt, pr)); out.update(instance(gt, pr))
    p, r = out["inst_prec"], out["inst_rec"]
    out["inst_f1"] = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return out
