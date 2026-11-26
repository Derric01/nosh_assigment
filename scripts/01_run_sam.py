"""Task 1 stage A: run SAM on every image (resumable) and cache raw candidates.
usage: python scripts/01_run_sam.py [--model vit_b|vit_l] [--pps 32]"""
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eggseg import data, sam_seg

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="vit_b"); ap.add_argument("--pps", type=int, default=32)
a = ap.parse_args()
ck = {"vit_b": "sam_b.pt", "vit_l": "sam_l.pt"}[a.model]
os.makedirs("outputs/sam_raw", exist_ok=True)
gen = sam_seg.load_generator(os.path.join(data.ROOT, "weights", ck), a.model, a.pps)
paths = data.list_images()
for i, p in enumerate(paths):
    out = f"outputs/sam_raw/{data.stem(p)}.npz"
    if os.path.exists(out):
        continue
    t = time.time()
    c = gen.generate(data.load_rgb(p))
    sam_seg.save_raw(out, c)
    print(f"[{i+1}/{len(paths)}] {data.stem(p)} {len(c)} cands {time.time()-t:.0f}s", flush=True)
