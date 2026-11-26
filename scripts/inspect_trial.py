"""Look at a trial's dev-set errors: python scripts/inspect_trial.py T0 [n_worst]  -> outputs/inspect_<id>.jpg
(only the DEV split is read here, so inspecting never touches eval)."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2, numpy as np, pandas as pd
from eggseg import data, classical, metrics, viz

tid, k = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20
cfg = json.loads(pd.read_csv("experiments/results.csv").set_index("id").loc[tid, "params"])
by = {data.stem(p): p for p in data.list_images()}
rows = []
for n in data.read_split("dev"):
    rgb = data.load_rgb(by[n]); gt = data.load_instance_mask(n); pr = classical.segment(rgb, cfg)
    rows.append((metrics.semantic(gt, pr)["iou"], n, rgb, gt, pr))
rows.sort(key=lambda r: r[0])
tiles = [viz.label(viz.error_overlay(r[2], r[3], r[4]), f"{r[1][-3:].strip('_')} IoU {r[0]:.2f}") for r in rows[:k]]
cv2.imwrite(f"outputs/inspect_{tid}.jpg", cv2.cvtColor(viz.sheet(tiles, 5, 256), cv2.COLOR_RGB2BGR))
print("worst:", [(r[1][-4:], round(r[0], 2)) for r in rows[:k]])
print("median IoU %.2f, best:" % np.median([r[0] for r in rows]), [(r[1][-4:], round(r[0], 2)) for r in rows[-5:]])
