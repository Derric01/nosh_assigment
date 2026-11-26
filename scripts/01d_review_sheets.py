"""Task 1 stage D: build review contact sheets + egg counts from the FINAL masks (after manual fixes)."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2, numpy as np
from eggseg import data, viz

paths, tiles, counts = data.list_images(), [], {}
for p in paths:
    n = data.stem(p); inst = data.load_instance_mask(n)
    counts[n] = int(len([i for i in np.unique(inst) if i]))
    tiles.append(viz.label(viz.overlay(data.load_rgb(p), inst), f"{n.replace('not_damaged_','')}: {counts[n]}"))
json.dump(counts, open("outputs/sam_counts.json", "w"), indent=1)
for k in range(0, len(tiles), 20):
    cv2.imwrite(f"outputs/sam_review_{k // 20}.jpg", cv2.cvtColor(viz.sheet(tiles[k:k + 20], 5, 256), cv2.COLOR_RGB2BGR))
v = np.array(list(counts.values()))
print("eggs/image:", {int(k): int((v == k).sum()) for k in np.unique(v)}, " total eggs:", int(v.sum()))
