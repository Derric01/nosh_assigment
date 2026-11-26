"""Task 1 stage B: filter cached SAM candidates to eggs -> instance PNGs, overlays, review sheets."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2, numpy as np
from eggseg import data, sam_seg, viz

os.makedirs(data.SAM_DIR, exist_ok=True); os.makedirs("outputs/sam_overlays", exist_ok=True)
paths, tiles, counts = data.list_images(), [], {}
for p in paths:
    n = data.stem(p)
    if not os.path.exists(f"outputs/sam_raw/{n}.npz"):
        continue          # SAM stage A has not reached this image yet
    rgb = data.load_rgb(p)
    masks, iou, stab = sam_seg.load_raw(f"outputs/sam_raw/{n}.npz")
    keep = sam_seg.select_eggs(masks, iou, stab, rgb=rgb)
    inst = sam_seg.to_instance_png(keep, rgb.shape[:2])
    cv2.imwrite(os.path.join(data.SAM_DIR, n + ".png"), inst)
    ov = viz.overlay(rgb, inst)
    cv2.imwrite(f"outputs/sam_overlays/{n}.jpg", cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    tiles.append(viz.label(ov, f"{n.replace('not_damaged_','')}: {len(keep)}"))
    counts[n] = len(keep)
json.dump(counts, open("outputs/sam_counts.json", "w"), indent=1)
for k in range(0, len(tiles), 20):
    s = viz.sheet(tiles[k:k + 20], 5, 256)
    cv2.imwrite(f"outputs/sam_review_{k // 20}.jpg", cv2.cvtColor(s, cv2.COLOR_RGB2BGR))
vals = np.array(list(counts.values()))
print("eggs/image histogram:", {int(v): int((vals == v).sum()) for v in np.unique(vals)})
