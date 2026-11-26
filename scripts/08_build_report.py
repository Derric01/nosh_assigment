"""Build report/report.md -> report.html -> report.pdf from the result files (no hand-typed numbers).
usage: python scripts/08_build_report.py"""
import sys, os, json, subprocess, re, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np, pandas as pd, cv2, markdown
from eggseg import data
from trials import TRIALS

ROOT = data.ROOT; os.chdir(ROOT); os.makedirs("report/figures", exist_ok=True)
S = json.load(open("outputs/eval/summary.json")); E, D = S["eval"], S["dev"]
ev = pd.read_csv("outputs/eval/eval_per_image.csv"); dv = pd.read_csv("outputs/eval/dev_per_image.csv"); al = pd.concat([ev, dv], ignore_index=True)
R = pd.read_csv("experiments/results.csv").set_index("id"); DEC = json.load(open("experiments/decisions.json"))
G = pd.read_csv("splits/groups.csv"); FIN = json.load(open("experiments/final_config.json"))
FIX = {k: v for k, v in json.load(open("scripts/sam_manual_fixes.json")).items() if not k.startswith("_")}
CNT = json.load(open("outputs/sam_counts.json"))
DRIVE = open("report/drive_link.txt").read().strip() if os.path.exists("report/drive_link.txt") else None
REPO = "https://github.com/derric01/nosh_assigment"
f3 = lambda x: f"{x:.3f}"


def md_table(df):
    return "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.values)


# ---------- figures made here ----------
def groups_figure():
    cap = G.groupby("group").size(); multi = cap[cap > 1].index[:12]
    by = {data.stem(p): p for p in data.list_images()}
    cells = []
    for g in multi:
        names = G[G.group == g].name.tolist()[:4]; sp = G[G.group == g].split.iloc[0]
        tiles = []
        for n in names:
            im = cv2.resize(cv2.imread(by[n]), (80, 80)); col = (60, 60, 210) if sp == "eval" else (150, 150, 150)
            tiles.append(cv2.copyMakeBorder(im, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=col))
        while len(tiles) < 4: tiles.append(np.full_like(tiles[0], 255))
        cells.append(np.hstack(tiles))
    rows = [np.hstack(cells[k:k + 3] + [np.full_like(cells[0], 255)] * (3 - len(cells[k:k + 3]))) for k in range(0, len(cells), 3)]
    cv2.imwrite("report/figures/fig_groups.jpg", np.vstack(rows))
groups_figure()
_g = cv2.imread("outputs/eval/gallery_worst.jpg"); h = _g.shape[0] // 2
cv2.imwrite("report/figures/gallery_worst_a.jpg", _g[:h]); cv2.imwrite("report/figures/gallery_worst_b.jpg", _g[h:])
for src, dst in [("outputs/sam_review_0.jpg", "sam_review_0.jpg"), ("outputs/sam_review_3.jpg", "sam_review_3.jpg"), ("outputs/eval/fig_trials.png", "fig_trials.png"),
                 ("outputs/eval/fig_overview.png", "fig_overview.png"), ("outputs/eval/fig_iou_vs_factors.png", "fig_iou_vs_factors.png"),
                 ("outputs/eval/fig_errors.png", "fig_errors.png"), ("outputs/eval/gallery_worst.jpg", "gallery_worst.jpg"), ("outputs/eval/gallery_best.jpg", "gallery_best.jpg"),
                 ("outputs/tracking/video_vs_sam.jpg", "video_vs_sam.jpg")]:
    if os.path.exists(src): shutil.copy(src, "report/figures/" + dst)

# ---------- split / leakage numbers ----------
ng = G.group.nunique(); sizes = G.groupby("group").size()
rng = np.random.RandomState(0); leaky = []
for _ in range(300):                                  # how leaky is a plain random 22-image eval split?
    idx = rng.permutation(len(G)); evi = set(idx[:22]); gg = G.group.values
    dev_groups = {gg[i] for i in range(len(G)) if i not in evi}
    leaky.append(sum(gg[i] in dev_groups and sizes[gg[i]] > 1 for i in evi))
ev_g = G[G.split == "eval"]
# ---------- SAM numbers ----------
cnt = pd.Series(CNT); bench = open("experiments/sam_benchmark.txt").read().strip().replace("\n", "<br>")
# ---------- failure statistics (pooled 110 images, descriptive) ----------
def grp(mask): return f"{al.iou[mask].mean():.2f} (n={int(mask.sum())}) vs {al.iou[~mask].mean():.2f} (n={int((~mask).sum())})"
lowc, hisat, small = al.contrast_dE < 60, al.egg_sat >= 40, al.egg_area < 0.05
one, multi = al.n_gt == 1, al.n_gt >= 2
extra = int(((al.n_gt == 1) & (al.n_pred > 1)).sum()); n1 = int(one.sum())
catg = ev.category.value_counts()
low = al[al.iou < 0.2]
# ---------- tracking numbers ----------
T = pd.read_csv("outputs/tracking/tracks.csv"); TS = json.load(open("outputs/tracking/summary.json"))
scene = range(177, 1151); cov = {}
for i, g in T.groupby("id"):
    cov[i] = (len(set(g.frame) & set(scene)) / len(scene), len(set(g[g.state == "detected"].frame) & set(scene)) / len(scene))
both = T.groupby("frame").id.nunique().reindex(list(scene)).fillna(0)
V = pd.read_csv("outputs/tracking/video_vs_sam.csv") if os.path.exists("outputs/tracking/video_vs_sam.csv") else None

# ---------- experiment table ----------
rows = []
for t in TRIALS:
    r = R.loc[t["id"]]
    rows.append([t["id"], t["name"], f3(r.iou), f3(r.dice), f3(r.bf), f3(r.inst_f1), f"{r.count_acc:.2f}", DEC[t["id"]]["decision"]])
trial_tbl = md_table(pd.DataFrame(rows, columns=["id", "trial", "IoU", "Dice", "BF", "inst-F1", "count-acc", "decision"]))
trial_notes = "\n".join(f"- **{t['id']}** - {t['change']}. *Why:* {t['why']}. *Result:* {DEC[t['id']]['reason']}" for t in TRIALS)
res_tbl = pd.DataFrame({"metric": ["IoU (mean per image)", "IoU (median)", "Dice", "pixel precision", "pixel recall", "boundary F1 (3 px)", "HD95 (px, mean)",
                                   "per-egg precision (IoU>=0.5)", "per-egg recall", "per-egg F1", "mean IoU of matched eggs", "micro IoU (pooled pixels)", "images with IoU >= 0.5", "images with IoU >= 0.8", "right egg count"],
                        "eval (22 images, held out)": [f3(E["iou"]), f3(E["iou_median"]), f3(E["dice"]), f3(E["prec"]), f3(E["rec"]), f3(E["bf"]), f"{E['hd95']:.1f}", f3(E["inst_prec"]), f3(E["inst_rec"]), f3(E["inst_f1"]), f3(E["matched_iou"]), f3(E["micro_iou"]), f"{E['frac_iou_ge_0_5']:.0%}", f"{E['frac_iou_ge_0_8']:.0%}", f"{E['count_acc']:.0%}"],
                        "dev (88 images, used for tuning)": [f3(D["iou"]), f3(D["iou_median"]), f3(D["dice"]), f3(D["prec"]), f3(D["rec"]), f3(D["bf"]), f"{D['hd95']:.1f}", f3(D["inst_prec"]), f3(D["inst_rec"]), f3(D["inst_f1"]), f3(D["matched_iou"]), f3(D["micro_iou"]), f"{D['frac_iou_ge_0_5']:.0%}", f"{D['frac_iou_ge_0_8']:.0%}", f"{D['count_acc']:.0%}"]})
fix_tbl = md_table(pd.DataFrame([[k, ("replace" if "replace" in v else "") + ("add" if "add" in v else "") + ("remove" if "remove" in v else ""), v["why"]] for k, v in FIX.items()], columns=["image", "action", "why"]))
cfg_json = json.dumps(FIN["params"], indent=1)

md = f"""
# Egg segmentation and tracking: CV intern task report

Code and instructions: **[{REPO}]({REPO})**  |  Tracked video (Google Drive): **{('[' + DRIVE + '](' + DRIVE + ')') if DRIVE else 'LINK TO BE ADDED'}**

## 0. Summary

| | result |
|---|---|
| Dataset | 110 images (640x640), 138 eggs by SAM; 88 dev / 22 eval images, split by scene group to avoid leakage |
| Task 1 | SAM ViT-B (automatic mode) + egg filter; {len(FIX)} of 110 images corrected with hand-drawn box prompts |
| Task 2 | classical algorithm (threshold sweep scored by egg shape, GrabCut refinement); dev IoU {f3(R.loc['T0'].iou)} (baseline) -> {f3(R.loc['F1'].iou)} over {len(TRIALS)} logged trials |
| Task 3 (eval split, SAM as ground truth) | mean IoU **{f3(E['iou'])}**, Dice {f3(E['dice'])}, boundary F1 {f3(E['bf'])}, per-egg F1 {f3(E['inst_f1'])}; {E['frac_iou_ge_0_8']:.0%} of images reach IoU >= 0.8 but {1-E['frac_iou_ge_0_5']:.0%} fail (IoU < 0.5) |
| Task 4 | 2 eggs tracked through the egg scene of the video with exactly 2 ids; egg 1 shown in {cov[1][0]:.0%} and egg 2 in {cov[2][0]:.0%} of the scene frames |

The classical method is a long way from SAM: it is good on contrasting backgrounds and breaks down when the egg has weak contrast with its surroundings.
Section 4 quantifies this.

## 1. Data and the dev/eval split

**Data.** The Roboflow export has 110 images, all 640x640 (Roboflow stretched them to square), in a classification-folder layout (train/valid/test, one class): **there are no boxes or masks**.
Roboflow's own train/valid/test partition is ignored. Eggs are white or cream; 1-3 per image on very different backgrounds (water in a pan, keyboard, rug, ash, leaves, egg tray, hands).
Two images (135, 145) are in Roboflow's `Unlabeled` folder; they show an egg in boiling water, so they are treated as egg images.

**Leakage control.** Photos of the same set-up (same rug, bowl, pan, ash, tray ...) look alike, so a random split would let tuning on dev leak into eval.
Images are therefore grouped by scene similarity and whole groups go to one side: two images are linked if their 256-bit perceptual hashes differ in <= 40 bits,
*or* their 16x16 colour thumbnails correlate >= 0.80, *or* their HSV histograms intersect >= 0.80; linked images are merged (union-find) into **{ng} groups**
(sizes: {', '.join(map(str, sorted(sizes, reverse=True)[:8]))} ... ; {int((sizes == 1).sum())} singletons). Groups were checked by eye (Figure 1). The split takes whole groups into eval until it holds 22 of 110 images
(first seed that gives exactly 88/22). For comparison, a plain random 22-image split would on average put **{np.mean(leaky):.1f} of the 22 eval images into a group that also has members in dev**
(mean over 300 random splits); here that number is 0. The most similar eval-to-dev pair has thumbnail correlation {ev_g.x_thumb.max():.2f}, histogram intersection {ev_g.x_hist.max():.2f}
and perceptual-hash distance {int(ev_g.x_phash.min())} bits, all on the safe side of the merge thresholds (0.80, 0.80, 40).

![Figure 1: example scene groups (each cluster of thumbnails is one group, at most 4 shown; grey border = dev, red border = eval)](figures/fig_groups.jpg)

The eval split is evaluated once, in Section 4, after the algorithm was frozen on dev results (`experiments/final_config.json`, frozen at commit `{FIN['frozen_at_parent_commit'][:8]}`).

## 2. Task 1 - segmentation with a pretrained model (SAM)

**Method.** The dataset has no prompts, so SAM ViT-B (official `segment-anything` package, CPU) runs in *segment-everything* mode (16x16 grid of point prompts, predicted-IoU >= 0.86, stability >= 0.92).
The candidate masks are cached and then filtered to eggs: area 1.2-50% of the image, solidity >= 0.93, ellipse-fit IoU >= 0.88, axis ratio <= 2, mean HSV saturation <= 60 and brightness >= 100,
border-touching masks must be >= 3% of the image with ellipse-IoU >= 0.93, and greedy NMS (IoU 0.3 / containment 0.8). The colour thresholds were chosen from the measured gap between the first false positives
(skin and leaf: saturation 77 and 124) and true eggs (<= 33). Output: one instance-id PNG per image (`outputs/sam_masks/`).
Benchmark on one image (4 CPU cores): {bench}.

**Manual corrections (disclosure).** The automatic selection found no egg in 10 of the 110 images and gave ragged or wrong masks in a few others, so {len(FIX)} images were corrected by giving SAM hand-drawn **box prompts**
(box coordinates read by eye from the overlays, stored in `scripts/sam_manual_fixes.json`). For eggs half hidden by leaves or a bar, the convex hull of the prompted mask was used (eggs are convex).
These {len(FIX)} images are the hardest ones, so the SAM ground truth there is partly a human judgement.

{fix_tbl}

**Result.** {int(cnt.sum())} eggs in 110 images (1 egg: {int((cnt == 1).sum())} images, 2 eggs: {int((cnt == 2).sum())}, 3 eggs: {int((cnt == 3).sum())}). Figure 2 shows 40 of the 110 results (the full set is in `outputs/sam_review_*.jpg`).
Limits: the masks are used as ground truth in Task 3 but are not perfect (e.g. eggs on leaves, backlit eggs, white-on-white boundaries), so every metric below measures *agreement with SAM*, not with a human annotation.

![Figure 2: SAM masks on images 1-20 (top) and 60-79 (bottom); number = eggs found](figures/sam_review_0.jpg)
![](figures/sam_review_3.jpg)

## 3. Task 2 - segmentation with classical image processing

**Final algorithm** (`src/eggseg/classical.py`, function `segment`, config below). Eggs are bright, convex, elliptical and white, but no single threshold isolates them in every image
(a white egg on a white bag merges with it), so instead of picking one threshold:

1. Build three feature maps: gray; *whiteness* = V - 0.8 S (HSV); and the Lab colour distance to the 3 dominant colours of the image border (background colours).
2. Threshold each map at ~34 levels (40..240 step 6); at every level clean with open/close and take every connected component as an **egg candidate**.
3. Score each candidate: ellipse-fit IoU x solidity (does it look like an egg?) x boundary contrast (Lab distance between its inside and a 9 px ring around it) x area^0.5 (prefer the whole egg over its highlight);
   reject candidates larger than 50% of the image, with solidity < 0.9, ellipse-IoU < 0.85, axis ratio > 2.2 or mean saturation > 70.
4. Greedy non-maximum suppression keeps the best non-overlapping candidates (one per egg).
5. **GrabCut** (10 iterations) seeded from each candidate (sure foreground = candidate eroded by 0.5 radius, sure background = farther than 1 radius) moves the boundary to the real egg edge.
6. If no candidate passed, repeat with relaxed thresholds (solidity >= 0.8, ellipse-IoU >= 0.7, no colour gate) and keep the single best one (every dataset image has an egg).

```
{cfg_json}
```
(all other parameters are the defaults in `classical.BASE`).

**Experiment log.** {len(TRIALS)} trials, all measured on the 88 dev images only, against the SAM masks. Every trial is a config in `experiments/trials.py`; full parameters, reasoning and the verdict
of each trial are in [`experiments/LOG.md`](../experiments/LOG.md) (generated from `results.csv`). Metrics: IoU/Dice/BF are means over images (BF = boundary F1 with 3 px tolerance); inst-F1 = per-egg F1 at IoU >= 0.5; count-acc = fraction of images with the right number of eggs.

{trial_tbl}

![Figure 3: dev IoU of every trial (filled = kept or reused, hollow = discarded)](figures/fig_trials.png)

**How the experiments led to the algorithm.**

{trial_notes}

*In short:* global thresholds (T0-T4) cannot separate a white egg from a bright background, they either merge them or pick the dim table reflection/background instead. Moving from "one threshold" to "many thresholds, choose the most egg-shaped component" (T7) was the first real gain;
then every addition fixed a specific failure seen in the dev error overlays: wrong egg-shaped blobs (contrast, T9), eggs invisible in one feature map (pooling, T10), skin/rugs (colour gate, T12), highlights instead of whole eggs (area prior, T14) and rough boundaries/partial eggs (GrabCut, T15, the biggest gain);
the fallback (T19) removed empty results. Unsuccessful attempts that were kept in the log on purpose: HSV-V and L* thresholds, adaptive threshold, border-colour distance as the only generator, Canny loops, relaxed shape thresholds, more smoothing, limiting GrabCut, a centre prior.

## 4. Task 3 - evaluation and analysis

Ground truth = SAM masks (Task 1), prediction = final classical algorithm, **eval split (22 images, evaluated once)**; dev numbers are shown for reference.
Definitions: *semantic* metrics merge all eggs into one foreground mask (IoU = TP/(TP+FP+FN), Dice, pixel precision/recall); *boundary F1* counts an edge pixel as correct if a GT edge pixel is within 3 px (and vice versa), HD95 is the 95th-percentile
edge-to-edge distance; *per-egg* metrics match predicted and GT eggs one-to-one (Hungarian) and count a match when IoU >= 0.5. Images with no egg in both masks would score 1; none occur here.

{md_table(res_tbl)}

Eval and dev agree within sampling noise (22 images is a small sample; the eval per-image IoU standard deviation is {E['iou_std']:.2f}), so tuning on dev did not obviously overfit.
The distribution is **bimodal** (Figure 4): {E['frac_iou_ge_0_8']:.0%} of eval images are good (IoU >= 0.8, matched-egg IoU {E['matched_iou']:.2f}) while {1-E['frac_iou_ge_0_5']:.0%} fail badly, so the mean ({f3(E['iou'])}) hides two very different regimes.

![Figure 4: IoU distribution, per-image eval IoU and failure taxonomy](figures/fig_overview.png)

### Qualitative evaluation: the eight worst eval images
Columns: image | SAM (ground truth) | classical prediction | error map (green = correct, red = false positive, blue = missed egg).

![Figure 5a: worst 4 eval images (images 30, 24, 20, 160)](figures/gallery_worst_a.jpg)
![Figure 5b: next 4 worst eval images (images 143, 3, 31, 140)](figures/gallery_worst_b.jpg)

What went wrong (matching the rows of Figure 5):

- **Low contrast egg on a bright background (images 30, 143, 3, 31):** a white egg on a white wall/bag/cloth/bowl. The sweep does not find the whole egg: either the egg merges with the background (image 30: the prediction is the white paper), or only its brightest part is found (143, 3, 31: the shaded half of the egg is missed, 'under-segmented').
  A likely reason why GrabCut does not recover the shaded half is that its colour is closer to the background than to the lit half (not verified separately).
- **Dark egg on a bright background (images 24, 20, backlit hand-held eggs):** the method assumes *bright egg*; here the egg is darker than the wall, so the prediction is a patch of bright wall (0 IoU).
- **Skin-coloured surroundings (image 160):** the predicted region covers the palm instead of the egg ('leak into background'); a pale palm has low saturation, so the white-egg colour gate does not reject it.
- **Large flat objects (image 140):** a large grey region (the metal plate) is predicted as an egg, probably because a smooth, bright, convex region passes the shape test after GrabCut rounds it off; the lowest egg also merges with the table.

Best cases for contrast (egg on a contrasting, smooth surface, IoU 0.97+):

![Figure 6: best 3 eval images](figures/gallery_best.jpg)

### Quantitative error analysis (all 110 images pooled, descriptive only)

![Figure 7: IoU against egg size and contrast, background texture and egg saturation (red = eval, grey = dev)](figures/fig_iou_vs_factors.png)
![Figure 8: error pixels, IoU by egg count, egg-count confusion matrix](figures/fig_errors.png)

- **Over-segmentation dominates.** False-positive pixels outnumber missed pixels {ev.fp.sum() / ev.fn.sum():.1f}:1 on eval ({al.fp.sum() / al.fn.sum():.1f}:1 pooled); pixel precision {E['prec']:.2f} vs recall {E['rec']:.2f}. GrabCut and the relaxed fallback grow into the background more often than they miss egg pixels.
- **Contrast is the strongest factor.** Mean IoU is {grp(lowc)} for images where the egg differs from its surroundings by Lab dE < 60 vs >= 60 (Figure 7, second panel).
- **Small eggs are harder:** egg area < 5% of the image: {grp(small)}.
- **Saturated (non-white) eggs fail:** egg mean saturation >= 40 (brown/pinkish, backlit): {grp(hisat)} - these are rejected by the white-egg colour gate, which is a deliberate dataset prior and the reason this method would not transfer to brown eggs.
- **The hard images are hard for SAM too:** the {len(FIX)} images that needed manual SAM prompts: {grp(al.manual_gt.astype(bool))}.
- **Wrong object beats no object.** {len(low)} of 110 images have IoU < 0.2, but only {int((low.n_pred == 0).sum())} of them have an empty prediction: the relaxed fallback and the candidate scoring usually pick a *wrong* region (Figure 5), which is worse than a miss for any downstream use.
- **Counting:** the right egg count in {E['count_acc']:.0%} of eval images. In single-egg images ({n1} of 110) the method predicts more than one egg in {extra} cases ({extra / n1:.0%}), i.e. spurious extra eggs (Figure 8, right); in the 17 multi-egg images the count is right for {int(((al.n_gt >= 2) & (al.n_gt == al.n_pred)).sum())}.
  Multi-egg images score higher (IoU {al.iou[multi].mean():.2f} vs {al.iou[one].mean():.2f} for single-egg); in the galleries they are mostly on contrasting surfaces such as a marble counter or water in a pan (an observation, not tested).
- **Boundaries:** where an egg is found, the boundary is decent (matched-egg IoU {E['matched_iou']:.2f}, boundary F1 {E['bf']:.2f}), but HD95 is {E['hd95']:.0f} px on average because the failures are far away from the egg.

### Where the algorithm should be improved
1. **Remove the bright-egg assumption**: also sweep inverted feature maps (dark egg on a bright wall) and let the contrast/shape score decide (images 24, 20).
2. **Shading-aware egg model**: a lit egg has a smooth, monotone brightness gradient and a shadow side. Fitting an ellipse to *both* the lit part and the adjacent shaded part (e.g. an ellipse-constrained GrabCut, or growing the candidate along smooth gradients) would fix the half-egg failures (143, 3, 31).
3. **Constrain GrabCut**: after refinement fit an ellipse and fall back (or intersect with the dilated ellipse) if the mask leaves it; T16 showed that a crude size limit hurts, so the constraint should be shape-based, not size-based.
4. **Explicit skin and large-object rejection**: a skin detector in YCrCb and a limit on candidate size relative to the *median* candidate would stop the hand/plate leaks (160, 140).
5. **Learn the candidate score**: the current score is a hand-weighted product; a small classifier (logistic regression / gradient-boosted trees on the shape, contrast, colour and position features already computed here) trained on the dev candidates, labelled by IoU with SAM, would weight them properly while staying classical.
6. **Count prior**: penalise a 2nd/3rd egg unless it is also well supported (the 24/93 spurious second eggs).
7. **Better ground truth**: some SAM masks are themselves uncertain (leaves, backlit); a second pretrained model or manual review of the eval images would give a more trustworthy upper bound.

## 5. Task 4 - video tracking

**Video.** Screen recording of a 59 s clip (1912x1074, 30 fps, 1779 frames) of two white eggs on a dark table which a hand touches and spins. Only frames ~177-1150 show the eggs; the rest are title/text cards and the YouTube interface with a recommended-videos grid
(whose egg thumbnails would otherwise create false tracks).

**Pipeline** (`scripts/05_track_video.py`, `src/eggseg/tracking.py`).

1. *Scene gate:* a frame is processed only if > 25% of its lower half is dark brown (the table); this selects exactly frames 177-1150 as one run.
2. *Segmentation (adapted from Task 2):* frame resized to 640x360; the *whiteness* feature thresholded at a **fixed** value of 200, open/close, then the same shape filter (solidity >= 0.9, ellipse-IoU >= 0.85, axis ratio <= 2.2). The two Task-2 features that were switched off: Otsu (it picked the dim table
   reflections under the eggs and missed the eggs: observed on frames 323-484, so it was replaced by a fixed threshold) and the sweep/GrabCut stages (a 16x slower sweep variant was tried and gave almost the same number of two-egg frames, 224 vs 220 of 323 sampled, so it was dropped).
3. *Tracking (SORT-style):* every egg is a track with a constant-velocity **Kalman filter** over its box (centre, width, height). Per frame: predict all tracks; cost = 0.5 (1 - IoU) + 0.5 x distance/gate between predicted boxes and detections;
   **Hungarian** assignment; matched tracks are corrected, unmatched detections start tentative tracks (confirmed after 8 detections), unmatched tracks coast on the prediction for 20 frames (shown as 'predicted').
4. *Re-identification:* the first version lost identities whenever the hand covered an egg for more than 20 frames (6 ids for 2 eggs); the presenter also moves eggs while they are hidden (one egg reappears ~0.5 frame widths away).
   Lost tracks are therefore remembered for 900 frames and a new track that reaches confirmation *inherits the id of the nearest lost track* (position + size) within 0.7 frame widths. This brought the result to exactly 2 ids.

**Result.** Output: `outputs/tracking/tracked.mp4` (masks, ids, 45-frame trails) uploaded to Google Drive: **{('[' + DRIVE + '](' + DRIVE + ')') if DRIVE else 'LINK TO BE ADDED'}**; per-frame boxes in `outputs/tracking/tracks.csv`.
In the egg scene (frames 177-1150, {len(scene)} frames) egg 1 is shown in {cov[1][0]:.0%} of frames (detected in {cov[1][1]:.0%}), egg 2 in {cov[2][0]:.0%} (detected in {cov[2][1]:.0%}); both are shown in {(both == 2).mean():.0%} of the frames. Total ids created: {int(T.id.nunique())} for 2 eggs.

![Figure 9: tracked frames (250, 480, 520, 606 / 700, 830, 920, 1100)](figures/tracking_frames.jpg)
![Figure 10: when each id is detected (dark) or predicted (light)](figures/fig_track_timeline.png)

**Limitations.**

- *Egg 2 is the spinning egg:* it is blurred and often under the hand, so it is missed for ~{1 - cov[2][1]:.0%} of the scene frames (frame 920 is an example of a visible but undetected egg). After 20 frames without a detection a track stops being drawn until the egg is re-found.
- *Identity after occlusion is a best guess:* the two eggs look the same, so when both are hidden or an egg is moved while hidden the id can only be assigned from position and size. It happens to stay consistent in this clip, but there is no appearance cue to prove it (frames 500-850 are the critical part).
- *No tracking ground truth exists,* so no MOTA/IDF1 is reported; the numbers above are coverage and id counts, checked on sampled frames (Figures 9 and 11).
- The detector is tuned for white eggs on a dark table (fixed threshold, scene gate); it is not the general Task-2 algorithm.
"""
if V is not None:
    md += f"""
**Segmentation check against SAM on {len(V)} sampled egg-scene frames.** With SAM masks (same egg filter as Task 1) as proxy ground truth, the video detector reaches mean IoU {V.iou.mean():.2f}, Dice {V.dice.mean():.2f}, boundary F1 {V.bf.mean():.2f}, per-egg F1 {V.inst_f1.mean():.2f}.
SAM itself is not reliable on this video: it also segments the **table reflections under the eggs** as eggs (blue regions in Figure 11), finding more than 2 eggs in {int((V.n_gt > 2).sum())} of {len(V)} frames. These reflections are SAM errors, but they lower the scores above (recall {V.rec.mean():.2f}, precision {V.prec.mean():.2f}: the detector never predicts anything SAM does not).
In the {int((V.n_gt == V.n_pred).sum())} of {len(V)} frames where both find the same number of eggs the mean IoU is {V.iou[V.n_gt == V.n_pred].mean():.2f}. This is a sanity check on mask quality, not a benchmark; the frames where an egg is under the hand (e.g. 790-890) are missed by both SAM-as-filtered and the detector.

![Figure 11: video detector vs SAM on sampled frames (green = agree, red = detector only, blue = SAM only)](figures/video_vs_sam.jpg)
"""
md += f"""
## 6. Reproducing the results
See the README ([{REPO}]({REPO})): environment, data download and the numbered scripts (`scripts/01_run_sam.py` ... `scripts/08_build_report.py`), with expected CPU run times. Everything ran on 4 CPU cores without a GPU.

## 7. Use of AI tools
Claude Code (an AI coding assistant) was used to help write the code, run the experiments and draft this report, under the author's direction. Report tables and figures are generated by the scripts in the repository.
The manual choices (hand-drawn box prompts for {len(FIX)} SAM masks; the white-egg and at-least-one-egg assumptions) are described in Sections 2 and 3.
"""
md += """
## 8. Author and credit

**Author:** Derric01 (GitHub). This report and repository are the author's submission for the CV intern task. The author directed the work, supplied the tracking video, and is responsible for the submission.

**Background (as stated by the author):** prior hands-on experience with image annotation, with training YOLO models and with SAM (Segment Anything Model).
Of these, only SAM is used in this project (Task 1). Annotation and YOLO training were not: the dataset comes without annotations and no neural network was trained here; they are mentioned for context only.
A YOLO model trained on annotated eggs is the natural learned counterpart to the classical method and would be the first thing to try next, in the spirit of improvement idea 5 in Section 4 (learning the decision instead of hand-weighting it).
"""
open("report/report.md", "w").write(md)
css = """body{font-family:'DejaVu Sans',Arial,sans-serif;font-size:9.6pt;line-height:1.42;color:#1a1a1a}h1{font-size:19pt;margin:0 0 6px}h2{font-size:14pt;border-bottom:2px solid #d1495b;padding-bottom:2px;margin-top:20px;page-break-after:avoid}h3{font-size:11pt;margin-top:12px;page-break-after:avoid}
table{border-collapse:collapse;margin:8px 0;font-size:7.8pt;width:100%}th,td{border:1px solid #ccc;padding:2px 5px;text-align:left;vertical-align:top}th{background:#f0f0f0}
img{max-width:100%;display:block;margin:6px auto;page-break-inside:avoid}p>img+img{margin-top:2px}code,pre{font-family:'DejaVu Sans Mono',monospace;font-size:7.6pt;background:#f6f6f6}pre{padding:6px;white-space:pre-wrap}a{color:#1b5fa8}figure img[src*='gallery_worst']{max-width:72%}figure img[src*='fig_groups']{max-width:85%}li{margin-bottom:2px}"""
html = markdown.markdown(md, extensions=["tables", "fenced_code"])
# figure captions: turn "alt text" of images starting with 'Figure' into captions
html = re.sub(r'<img alt="(Figure[^"]*)" src="([^"]+)" />', r'<figure style="margin:6px 0;page-break-inside:avoid"><img src="\2"><figcaption style="font-size:8pt;color:#555;text-align:center">\1</figcaption></figure>', html)
open("report/report.html", "w").write(f"<!doctype html><html><head><meta charset='utf-8'><title>Egg segmentation and tracking</title><style>{css}</style></head><body>{html}</body></html>")
subprocess.run(["node", "scripts/html2pdf.js", "report/report.html", "report/report.pdf"], check=True)
print("report/report.pdf written")
# keep the README results table in sync with the real numbers
rd = open("README.md").read()
tbl = (f"| split | images | mean IoU | Dice | boundary F1 | per-egg F1 | right egg count |\n|---|---|---|---|---|---|---|\n"
       f"| dev (used for tuning) | 88 | {f3(D['iou'])} | {f3(D['dice'])} | {f3(D['bf'])} | {f3(D['inst_f1'])} | {D['count_acc']:.0%} |\n"
       f"| **eval (held out, evaluated once)** | 22 | **{f3(E['iou'])}** | {f3(E['dice'])} | {f3(E['bf'])} | {f3(E['inst_f1'])} | {E['count_acc']:.0%} |\n")
rd = re.sub(r"\| split \| images.*?\n\n\(The table", tbl + "\n(The table", rd, flags=re.S)
open("README.md", "w").write(rd)
