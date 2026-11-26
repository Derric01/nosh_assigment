# Execution Plan — Nosh CV Intern Task (egg segmentation + tracking)

Source brief: `CV_Intern_Task_Wellfound.pdf` (4 tasks + PDF report + public repo).

## 0. Environment facts (verified during planning)
- CPU only (no GPU), 4 cores, 15 GB RAM, Python 3.13, ~60 GB disk.
- Network: PyPI and GitHub (incl. release assets) reachable. **Blocked** at planning time:
  Google Drive, YouTube, HuggingFace, dl.fbaipublicfiles.com, download.pytorch.org, Roboflow.
  - Dataset zip: https://drive.google.com/file/d/1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv/view (needs Drive hosts allowed)
  - Roboflow page (reference only): https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1
  - Video: https://www.youtube.com/watch?v=Y4JyV94y2_A — YouTube streams come from unpredictable
    `*.googlevideo.com` hosts, so it cannot be fetched here; **the user supplies the mp4**
    (Drive share link or upload into `data/video/`).
- SAM weights: get via Ultralytics GitHub release assets (`sam_b.pt`, `sam_l.pt`, `mobile_sam.pt`),
  since fbaipublicfiles/HF are blocked. Install torch CPU from PyPI (if download.pytorch.org stays blocked).

## 1. Repo layout
```
README.md               setup + how to run every step, results summary, AI-tool disclosure
requirements.txt        opencv-python-headless, numpy, scipy, scikit-image, scikit-learn, pandas,
                        matplotlib, ultralytics, torch (cpu), imagehash, tqdm, gdown, markdown
src/eggseg/
  data.py               dataset discovery/loading (handles Roboflow COCO/YOLO export layouts)
  split.py              leakage-aware dev/eval split
  sam_seg.py            Task 1 SAM inference
  classical.py          Task 2 algorithm (configurable params dict; every trial = a config)
  metrics.py            semantic + instance + boundary metrics
  viz.py                overlays, error maps, plots
  tracking.py           Task 4 tracker (Kalman + Hungarian, SORT-style)
scripts/
  00_download.sh        gdown dataset zip -> data/raw ; unzip
  01_run_sam.py         -> outputs/sam_masks/{stem}.png (instance-id PNG) + overlays
  02_make_split.py      -> splits/dev.txt, splits/eval.txt, splits/groups.csv
  03_run_experiments.py runs each trial config on DEV only -> experiments/results.csv
  04_evaluate.py        final algorithm on EVAL (and dev) -> metrics csv + figures
  05_track_video.py     video -> outputs/tracking/tracked.mp4 + tracks.csv
  06_build_report.py    report/report.md(+figures) -> HTML -> report/report.pdf (Playwright Chromium)
experiments/            trial configs (yaml/json) + log
report/                 report.md, figures/, report.pdf (committed)
data/, outputs/large    gitignored (keep SAM masks small PNGs committed if total < ~50 MB)
```

## 2. Task 1 — SAM pseudo-ground-truth
0. **GATE — inspect before building anything.** Write `data/NOTES.md`: image count, resolutions,
   annotation format, eggs/image, egg colours, backgrounds, whether images look like video frames or
   Roboflow augmentations. Also inspect the tracking video as soon as it is available (scene, #eggs,
   touching eggs, motion, lighting) — Task 2 design must keep it in mind, not discover it at the end.
   Adjust the rest of this plan to what the data shows.
1. Inspect the export: count images, resolution, whether annotations exist (COCO json / YOLO txt boxes /
   polygons), number of eggs per image, egg colours (white/brown), backgrounds.
2. Prompting strategy (SAM needs prompts):
   - **Primary:** if the dataset has egg boxes, use each box as a SAM box prompt
     (Ultralytics `SAM('sam_l.pt')`, `bboxes=`), one instance mask per egg. Highest quality, document it.
   - **Fallback / also for "identify":** SAM automatic ("segment everything") mode + filter masks by
     egg-likeness: ellipse-fit residual, solidity > 0.9, area range, aspect ratio 1–1.8, mean brightness/
     low saturation; NMS on overlaps. Use this for images without boxes and report agreement with the
     box-prompted variant on a subset.
   - Because the brief says the model should *identify* eggs, report how well the box-free automatic
     variant recovers the egg instances (recall vs dataset boxes). Open-vocabulary detectors
     (YOLO-World/GroundingDINO) need CLIP/HF downloads that are blocked here — mention, don't rely on.
3. **Benchmark first:** time 5 images each with `mobile_sam`, `sam_b`, `sam_l`, then pick
   model size by measured CPU speed (target < ~5 s/img; `sam_b` if `sam_l` too slow). Resize long
   side to 1024 as SAM does internally anyway.
4. Post-process: keep largest component per instance, fill holes. Save instance-ID PNGs + overlay grid
   (like the brief's red-egg example). Commit the masks so later steps never require re-running SAM.
   Manually spot-check ~30 overlays; note any SAM failures in report.

## 3. Task 2 — split + classical algorithm
### Split (4:1, leakage-aware)
- Roboflow exports often contain augmented copies of one source (`<orig>_jpg.rf.<hash>.jpg`) and
  near-duplicate frames. Group by (a) original stem before `.rf.` and (b) perceptual-hash (pHash, Hamming
  ≤ ~8) clustering with union-find. Split **by group** (GroupShuffleSplit-style, fixed seed) to 80/20 of
  images; also stratify roughly by egg count / egg colour if feasible. Report group stats and nearest
  cross-split pHash distance as leakage evidence.
- Eval split is touched **once**, at the end. All tuning on dev.

### Algorithm development (each trial = config, logged with dev mIoU/Dice/instance-F1)
Planned trial sequence (keep honest results incl. failures):
- T0 baseline: grayscale + Otsu + largest component.
- T1 colour: HSV (low S, high V) and Lab thresholds; handle brown eggs (Lab b*, hue range) — compare.
- T2 preprocessing: Gaussian/bilateral blur, CLAHE, illumination normalisation (background division).
- T3 morphology: open/close kernel sizes, hole fill.
- T4 shape filtering: contour ellipse fit, solidity, area, aspect ratio, convexity to reject hands,
  shadows, white background objects.
- T5 refinement: GrabCut seeded from T4 mask (sure-FG = eroded, sure-BG = dilated complement).
- T6 instance split: distance transform + watershed for touching eggs.
- T7 alternatives expected to be weaker (log anyway): Canny + Hough/ellipse fitting, k-means in Lab,
  adaptive threshold, skin-colour removal.
- The trial list above is a hypothesis: order the real trials by the failures actually observed on dev
  (look at dev error overlays after each trial).
- Work at a fixed working resolution (e.g. long side 640–800) for speed; upsample masks to native
  resolution for metrics.
- Small grid/random search on key params (thresholds, kernel sizes, min area) on dev only.
- `experiments/results.csv` + `experiments/LOG.md`: for every trial — what changed, why, params,
  dev metrics, kept/discarded + reason. Include a plot of dev mIoU across trials (progression chart).

## 4. Task 3 — evaluation & analysis (eval split; also report dev for reference)
- Semantic (foreground egg vs background): IoU, Dice, pixel precision/recall, per-image and mean,
  plus micro (pooled-pixel) values.
- Boundary: Boundary F1 (tolerance ~2–3 px at native res) and HD95.
- Instance: Hungarian matching at IoU ≥ 0.5 → instance precision/recall/F1, count error, mean matched IoU.
- Define edge cases: images with no egg in GT and/or prediction (IoU = 1 if both empty, 0 if one
  empty; report count of such images separately). Note that SAM "GT" has its own errors.
- Breakdowns: by #eggs, egg colour, egg area fraction, image brightness/contrast, background type.
- Plots: IoU histogram/CDF, IoU vs. egg size / brightness scatter, FP vs FN pixel bars, per-trial chart.
- Qualitative: worst-k (e.g. 8) eval images with error overlays (TP green, FP red, FN blue), a few
  best cases for contrast, categorised failure modes (shadows, specular highlights, skin/hand, white
  background, touching eggs, brown eggs). Short writeup + concrete improvement ideas.

## 5. Task 4 — video tracking
- Input: user-provided mp4 in `data/video/`. Process at reduced resolution if slow; keep original fps.
- Per frame: classical segmentation (may adapt params for the video; document changes) → instances.
- Tracker: SORT-style — Kalman (cx, cy, area, aspect + velocities), Hungarian on (1 − IoU) with centroid
  gating, `max_age`, `min_hits`; optionally mask IoU instead of box IoU. Stable colour per ID.
- Output: `outputs/tracking/tracked.mp4` (mask overlay + ID labels + trails) + `tracks.csv`. Report
  ID-switch observations and track-count timeline. User uploads mp4 to Google Drive; put link in report.
- No tracking GT exists: evaluate with proxies — run SAM on ~20 sampled frames and score the per-frame
  segmentation against it; report #tracks vs visible eggs over time, track-length distribution, and
  manually counted ID switches on sampled clips.
- Expect domain shift vs. the image dataset; adapt parameters if needed and document what changed and why.
- Optional comparison: SAM on a few keyframes to sanity-check the classical masks in the video.

## 6. Report (report/report.pdf, committed)
Sections: Overview · Data & split (leakage handling) · Task 1 method + examples · Task 2 algorithm +
experiment log table + progression plot · Task 3 metrics tables, plots, qualitative failures,
improvements · Task 4 method + screenshots + **Google Drive link placeholder `<<DRIVE_LINK>>`** ·
Reproduction instructions · AI-tool disclosure. Build via markdown → HTML → PDF with Playwright Chromium.

## 7. Execution order & checkpoints (commit + push to `claude/practical-babbage-a510jm` after each)
1. Env setup + dataset download + data inspection notes.
2. Task 1 SAM masks + overlay spot-check.
3. Split.
4. Experiments (dev) → final config.
5. Eval + analysis figures.
6. Video tracking (once mp4 available).
7. Report PDF + README. Final review that every brief bullet is covered.

## 8. Guardrails (from plan review)
- **Never invent numbers.** Every metric in the log/report comes from a script run; `experiments/LOG.md`
  tables are generated from `experiments/results.csv`. Failed trials are recorded as they happened.
- **Eval split is used once**, after the final config is frozen (record the commit hash in the report).
- **Keep code simple and explainable** — the candidate must walk through it and modify it live in the
  interview. Plain functions, one params dict, short modules, comments on *why*; no needless abstraction.
- Fixed seeds everywhere; README lists exact commands in order with expected runtimes on CPU.
- Keep the repo small: no dataset, no weights, no large videos in git (tracked video goes to Drive;
  commit only a short GIF/screenshots).
- If CPU time becomes a bottleneck (large dataset / long video), subsample and say so explicitly.
