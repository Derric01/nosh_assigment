"""Build experiments/LOG.md + progression figure from results.csv, trials.py and decisions.json (no hand-typed numbers)."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trials import TRIALS

R = pd.read_csv("experiments/results.csv").set_index("id")
D = json.load(open("experiments/decisions.json"))
KEEP = lambda d: d.startswith(("kept", "FINAL"))
cfgs = {}
lines = ["# Task 2 experiment log (dev split only: 88 images; GT = SAM masks)\n",
         "Metrics are means over dev images. IoU/Dice/BF = semantic (all eggs merged) overlap and boundary-F1 (3 px); inst-F1 = per-egg F1 at IoU>=0.5; count-acc = fraction of images with the right number of eggs.\n",
         "| id | change | dev IoU | Dice | BF | inst-F1 | count-acc | decision |", "|---|---|---|---|---|---|---|---|"]
for t in TRIALS:
    r = R.loc[t["id"]]
    lines.append(f"| {t['id']} | {t['name']} | {r.iou:.3f} | {r.dice:.3f} | {r.bf:.3f} | {r.inst_f1:.3f} | {r.count_acc:.2f} | {D[t['id']]['decision']} |")
lines.append("\n## Details\n")
for t in TRIALS:
    r = R.loc[t["id"]]; cfgs[t["id"]] = {**cfgs.get(t["parent"], {}), **t["delta"]}
    lines += [f"### {t['id']} - {t['name']}  (parent: {t['parent'] or '-'})",
              f"- **What changed:** {t['change']}", f"- **Why:** {t['why']}",
              f"- **Parameters (delta vs parent):** `{json.dumps(t['delta'])}`",
              f"- **Dev result:** IoU {r.iou:.3f}, Dice {r.dice:.3f}, BF {r.bf:.3f}, inst-F1 {r.inst_f1:.3f}, count-acc {r.count_acc:.2f}, median IoU {r.iou_median:.2f}, images with IoU<0.5: {r.iou_lt50:.0%}",
              f"- **Decision:** {D[t['id']]['decision']} - {D[t['id']]['reason']}\n"]
open("experiments/LOG.md", "w").write("\n".join(lines))

# progression chart: the best-so-far lineage of kept trials, with discarded ones as hollow markers
fig, ax = plt.subplots(figsize=(13, 4.2))
ids = [t["id"] for t in TRIALS]; y = [R.loc[i].iou for i in ids]
ax.plot(range(len(ids)), y, color="#cccccc", zorder=1)
for k, i in enumerate(ids):
    kept = KEEP(D[i]["decision"])
    ax.scatter(k, y[k], s=70, zorder=3, facecolor="#3b8b5a" if kept else "white", edgecolor="#3b8b5a" if kept else "#d1495b", linewidth=2)
ax.set_xticks(range(len(ids))); ax.set_xticklabels(ids, rotation=60); ax.set(ylabel="dev IoU", title="Dev IoU per trial (filled = kept/used, hollow = discarded)")
ax.grid(alpha=.3); plt.tight_layout(); os.makedirs("outputs/eval", exist_ok=True); plt.savefig("outputs/eval/fig_trials.png", dpi=130)
print("wrote experiments/LOG.md and outputs/eval/fig_trials.png")
