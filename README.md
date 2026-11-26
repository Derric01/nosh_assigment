# Egg segmentation & tracking (CV intern task)

Segment eggs with a pretrained model (SAM), build a classical image-processing segmenter that tries to match it,
evaluate the gap, and track individual eggs in a video.

**Report (PDF): [`report/report.pdf`](report/report.pdf)** - all four tasks, the experiment log, the analysis and the link to the tracked video.
Tracked video (Google Drive, public): https://drive.google.com/file/d/1majh-sPgpO1AUofLZXiiyqViPrLVvGd-/view?usp=drivesdk
Full trial-by-trial log: [`experiments/LOG.md`](experiments/LOG.md).

## Results at a glance
Numbers below are generated from `outputs/eval/summary.json` (final classical algorithm, trial F1, scored against the SAM masks used as ground truth).

| split | images | mean IoU | Dice | boundary F1 | per-egg F1 | right egg count |
|---|---|---|---|---|---|---|
| dev (used for tuning) | 88 | 0.619 | 0.704 | 0.516 | 0.667 | 68% |
| **eval (held out, evaluated once)** | 22 | **0.594** | 0.667 | 0.580 | 0.618 | 73% |

(The table in the report is authoritative; this README only describes how to reproduce it.)

## Repository layout
```
src/eggseg/        data.py split.py sam_seg.py classical.py metrics.py viz.py tracking.py   (the library)
scripts/           numbered scripts, run in order (see below)
experiments/       trials.py (trial definitions) results.csv decisions.json LOG.md final_config.json
splits/            dev.txt eval.txt groups.csv   (leakage-aware 88/22 split)
outputs/           sam_masks/ (SAM ground truth PNGs, committed) eval/ tracking/ review sheets
report/            report.pdf + figures
PLAN.md            the plan this work followed
```

## Setup
Python 3.10+, CPU is enough (everything here was run on 4 CPU cores, no GPU).
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
For the PDF report only (step 11): `npm install playwright && npx playwright install chromium` (the report is rendered to PDF with headless Chromium).

Data and weights (not committed):
```bash
# dataset (Roboflow export, 110 images) from the Drive link in the task PDF
mkdir -p data/raw && gdown 1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv -O data/raw/egg.zip && unzip -q data/raw/egg.zip -d data/raw/egg
# SAM ViT-B weights (official segment-anything checkpoint format)
mkdir -p weights && curl -L -o weights/sam_b.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/sam_b.pt
#   (this is the official SAM ViT-B state dict; it loads directly with segment-anything. The original Meta URL also works:
#    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
# tracking video: download the YouTube video from the task PDF and save it as data/video/egg_video.mp4
```

## Run everything (in order)
| step | command | what it does | CPU time |
|---|---|---|---|
| 1 | `python scripts/02_make_split.py` | leakage-aware 4:1 dev/eval split -> `splits/` | seconds |
| 2 | `python scripts/01_run_sam.py --model vit_b --pps 16` | Task 1: SAM candidate masks for all images (cached, resumable) | ~1 h |
| 3 | `python scripts/01b_select_eggs.py` | pick egg masks from the candidates (shape + colour filter) | 1 min |
| 4 | `python scripts/01c_manual_fixes.py` | 15 images corrected with hand-drawn SAM box prompts (`scripts/sam_manual_fixes.json`) | 5 min |
| 5 | `python scripts/01d_review_sheets.py` | contact sheets for visual review of the SAM masks | seconds |
| 6 | `python scripts/03_run_experiments.py` | Task 2: run every trial in `experiments/trials.py` on the **dev** split | ~1 h |
| 7 | `python scripts/06_make_log.py` | build `experiments/LOG.md` + progression figure | seconds |
| 8 | `python scripts/04_evaluate.py` | Task 3: frozen final config on eval (+dev): metrics, plots, galleries | 3 min |
| 9 | `python scripts/05_track_video.py` | Task 4: segment + track -> `outputs/tracking/tracked.mp4`, `tracks.csv` | 1 min |
| 10 | `python scripts/05b_video_sam_check.py` | Task 4: video detector vs SAM on 20 sampled frames | ~15 min |
| 11 | `python scripts/07_tracking_figures.py`, `python scripts/08_build_report.py` | report figures + `report/report.pdf` | 1 min |

`python scripts/inspect_trial.py T15` shows the 20 worst dev images of a trial (green = correct, red = false positive, blue = missed).
The eval split is only read by `scripts/04_evaluate.py`; `experiments/final_config.json` records the configuration that was frozen on dev results before that.

## Algorithm in one paragraph (final classical method, trial F1)
For three feature maps (gray, "whiteness" = V - 0.8 S, and Lab distance to the border colours) threshold the image at ~30 levels;
every connected component at every level is an egg candidate, scored by how well a fitted ellipse explains it (ellipse-IoU x solidity),
how strongly its colour differs from a ring around it, and its size, and rejected if it is not white-ish (mean saturation > 70) or too
large/thin. Greedy non-maximum suppression keeps the best non-overlapping candidates; GrabCut (10 iterations, seeded from each
candidate) then moves the boundary onto the real egg edge; if nothing passed, a relaxed filter keeps the single best candidate.
Code: `src/eggseg/classical.py` (`segment`), config in `experiments/final_config.json`.

## Where AI tools were used
This project was developed with Claude Code:  wrote the code, ran the
experiments and drafted the report text. All numbers in the report are produced by the scripts in this repository (tables and logs are generated
from `results.csv` / `summary.json`, not typed by hand). Judgement calls made during the work are stated in the report:
the manual box-prompt corrections of 15 SAM masks, and the dataset-specific priors (white eggs, at least one egg per image).
