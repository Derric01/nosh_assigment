"""Task 4 sanity check: there is no tracking ground truth, so compare the video detector with SAM on
20 evenly spaced frames of the egg scene (SAM masks = proxy GT, same egg filter as Task 1).
Outputs outputs/tracking/video_vs_sam.csv + video_vs_sam.jpg"""
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import cv2, numpy as np, pandas as pd
from eggseg import classical, metrics, sam_seg, viz, data

vt = importlib.import_module("05_track_video")
gen = sam_seg.load_generator(os.path.join(data.ROOT, "weights", "sam_b.pt"), "vit_b", 16)
P = {**sam_seg.P, "min_area": 0.006, "max_area": 0.3}
cap = cv2.VideoCapture("data/video/egg_video.mp4")
rows, tiles = [], []
for fi in np.linspace(190, 1140, 20).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, f = cap.read()
    rgb = cv2.cvtColor(cv2.resize(f, (vt.PROC_W, vt.PROC_H)), cv2.COLOR_BGR2RGB)
    cands = gen.generate(rgb)
    keep = sam_seg.select_eggs(np.stack([c["segmentation"] for c in cands]), [c["predicted_iou"] for c in cands],
                               [c["stability_score"] for c in cands], p=P, rgb=rgb)
    gt = sam_seg.to_instance_png(keep, rgb.shape[:2])
    pr = classical.segment(rgb, vt.VIDEO_CFG)
    r = metrics.evaluate(gt, pr); r["frame"] = int(fi); rows.append(r)
    tiles.append(viz.label(viz.error_overlay(rgb, gt, pr), f"f{fi} IoU {r['iou']:.2f} gt{r['n_gt']} pr{r['n_pred']}"))
    print(fi, "IoU %.2f  gt eggs %d  pred eggs %d" % (r["iou"], r["n_gt"], r["n_pred"]), flush=True)
df = pd.DataFrame(rows); df.to_csv("outputs/tracking/video_vs_sam.csv", index=False)
cv2.imwrite("outputs/tracking/video_vs_sam.jpg", cv2.cvtColor(viz.sheet(tiles, 5, 320), cv2.COLOR_RGB2BGR))
print(df[["iou", "dice", "prec", "rec", "bf", "inst_f1"]].mean().round(3).to_dict())
