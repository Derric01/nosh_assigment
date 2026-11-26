"""Run every trial in experiments/trials.py on the DEV split only -> experiments/results.csv.
Eval split is never loaded here.   usage: python scripts/03_run_experiments.py [--rerun]"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np, pandas as pd
from eggseg import data, classical, metrics
from trials import TRIALS

ap = argparse.ArgumentParser(); ap.add_argument("--rerun", action="store_true"); a = ap.parse_args()
RES = "experiments/results.csv"
done = pd.read_csv(RES) if os.path.exists(RES) and not a.rerun else pd.DataFrame()
by_stem = {data.stem(p): p for p in data.list_images()}
dev = data.read_split("dev")
imgs = {n: data.load_rgb(by_stem[n]) for n in dev}
gts = {n: data.load_instance_mask(n) for n in dev}
cfgs = {}
rows = [] if a.rerun else done.to_dict("records")
for t in TRIALS:
    cfgs[t["id"]] = {**cfgs.get(t["parent"], {}), **t["delta"]}
    if t["id"] in set(done.get("id", [])):
        continue
    t0 = time.time()
    per = [metrics.evaluate(gts[n], classical.segment(imgs[n], cfgs[t["id"]])) for n in dev]
    df = pd.DataFrame(per)
    rows.append(dict(id=t["id"], name=t["name"], params=json.dumps(cfgs[t["id"]]),
                     iou=df.iou.mean(), dice=df.dice.mean(), prec=df.prec.mean(), rec=df.rec.mean(),
                     bf=df.bf.mean(), inst_f1=df.inst_f1.mean(), count_acc=(df.n_gt == df.n_pred).mean(),
                     iou_median=df.iou.median(), iou_lt50=(df.iou < 0.5).mean(), secs=time.time() - t0))
    r = rows[-1]
    print(f"{r['id']:>4} {r['name'][:40]:40} IoU {r['iou']:.3f} Dice {r['dice']:.3f} BF {r['bf']:.3f} "
          f"instF1 {r['inst_f1']:.3f} count_acc {r['count_acc']:.2f}", flush=True)
pd.DataFrame(rows).to_csv(RES, index=False)
