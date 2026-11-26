"""Task 1 stage C: apply scripts/sam_manual_fixes.json (box-prompted SAM) on top of stage B output.
Run order: 01_run_sam.py -> 01b_select_eggs.py -> 01c_manual_fixes.py (01b rewrites the PNGs, 01c patches them)."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2, numpy as np, torch
from segment_anything import sam_model_registry, SamPredictor
from eggseg import data, sam_seg, viz

torch.set_num_threads(4)
fixes = {k: v for k, v in json.load(open("scripts/sam_manual_fixes.json")).items() if not k.startswith("_")}
sam = sam_model_registry["vit_b"](checkpoint=os.path.join(data.ROOT, "weights", "sam_b.pt")).eval()
pred = SamPredictor(sam)
by = {data.stem(p): p for p in data.list_images()}

for num, fx in fixes.items():
    n = "not_damaged_" + num
    rgb = data.load_rgb(by[n])
    inst = data.load_instance_mask(n)
    masks = [inst == i for i in np.unique(inst) if i]           # automatic eggs from stage B
    if "replace" in fx:
        masks = []
    for x, y in fx.get("remove", []):
        masks = [m for m in masks if not m[y, x]]
    boxes = fx.get("replace", []) + fx.get("add", [])
    if boxes:
        pred.set_image(rgb)
        new = []
        for b in boxes:
            pts = np.array(fx.get("points", [])).reshape(-1, 2)
            m, sc, _ = pred.predict(point_coords=pts if len(pts) else None,
                                    point_labels=np.ones(len(pts), int) if len(pts) else None,
                                    box=np.array(b), multimask_output=False)
            c = sam_seg._clean(m[0])
            if c is not None and fx.get("hull"):      # eggs are convex: fill in what stems/bars hide
                h = cv2.convexHull(np.argwhere(c)[:, ::-1].astype(np.int32))
                c = cv2.fillConvexPoly(np.zeros(c.shape, np.uint8), h, 1).astype(bool)
            if c is not None:
                new.append(c)
        if fx.get("union") and new:                       # parts of one egg -> one mask (then hull)
            u = np.any(new, axis=0)
            if fx.get("hull"):
                h = cv2.convexHull(np.argwhere(u)[:, ::-1].astype(np.int32))
                u = cv2.fillConvexPoly(np.zeros(u.shape, np.uint8), h, 1).astype(bool)
            new = [u]
        masks += new
    masks.sort(key=lambda m: -m.sum())
    out = sam_seg.to_instance_png(masks, rgb.shape[:2])
    cv2.imwrite(os.path.join(data.SAM_DIR, n + ".png"), out)
    cv2.imwrite(f"outputs/sam_overlays/{n}.jpg", cv2.cvtColor(viz.overlay(rgb, out), cv2.COLOR_RGB2BGR))
    print(n, "->", len(masks), "eggs", flush=True)
