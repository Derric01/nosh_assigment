"""Leakage-aware 4:1 split.

Why: the 110 photos come from a handful of setups (same pan of water, same keyboard,
same rug...). If one photo of a setup is in dev and another in eval, the eval score
is optimistic because the tuned thresholds already 'know' that background.
So we cluster images by *scene similarity* and put whole clusters on one side.

Similarity = any of
  - perceptual hash (256 bit) Hamming distance <= PHASH_T   (near-duplicates)
  - correlation of 16x16 colour thumbnails        >= THUMB_T (same layout/colours)
  - HSV histogram intersection                     >= HIST_T  (same palette/background)
Pairs passing any test are merged with union-find -> groups.
"""
import numpy as np, imagehash, cv2
from PIL import Image

PHASH_T, THUMB_T, HIST_T = 40, 0.80, 0.80


def features(paths):
    ph, th, hs = [], [], []
    for p in paths:
        im = Image.open(p).convert("RGB")
        ph.append(imagehash.phash(im, hash_size=16))
        t = np.asarray(im.resize((16, 16))).astype(float).ravel()
        th.append((t - t.mean()) / (t.std() + 1e-9))
        hsv = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
        hs.append(h / h.sum())
    return ph, np.array(th), np.array(hs)


def pair_scores(paths):
    ph, th, hs = features(paths)
    n = len(paths)
    D = np.array([[a - b for b in ph] for a in ph], float)
    C = th @ th.T / th.shape[1]
    H = np.array([[np.minimum(a, b).sum() for b in hs] for a in hs])
    return D, C, H


def make_groups(paths):
    D, C, H = pair_scores(paths)
    n = len(paths)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= PHASH_T or C[i, j] >= THUMB_T or H[i, j] >= HIST_T:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    ids = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([ids[r] for r in roots]), (D, C, H)


def group_split(groups, eval_frac=0.2, seed=0):
    """Shuffle groups, then fill eval until it has ~20% of the images.
    Whole groups only, so no group straddles the boundary."""
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    rng.shuffle(ug)
    n, target = len(groups), eval_frac * len(groups)
    ev, cnt = [], 0
    for g in ug:
        size = (groups == g).sum()
        if abs(cnt + size - target) < abs(cnt - target):  # does adding it get closer?
            ev.append(g)
            cnt += size
    return np.isin(groups, ev)  # True = eval
