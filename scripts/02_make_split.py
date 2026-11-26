"""Leakage-aware 4:1 dev/eval split -> splits/{dev,eval}.txt + splits/groups.csv"""
import sys, os, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from eggseg import data, split

paths = data.list_images()
names = [data.stem(p) for p in paths]
groups, (D, C, H) = split.make_groups(paths)
# group sizes make an exact 22 awkward, so take the first seed that hits 22/88 exactly
SEED = next(sd for sd in range(1000) if split.group_split(groups, 0.2, sd).sum() == round(0.2 * len(paths)))
is_eval = split.group_split(groups, 0.2, SEED)
print("seed", SEED)

# leakage evidence: for each eval image, similarity to its closest DEV image vs closest eval image
dev, ev = np.where(~is_eval)[0], np.where(is_eval)[0]
def best(i, pool):
    pool = [j for j in pool if j != i]
    return dict(thumb=C[i, pool].max(), hist=H[i, pool].max(), phash=D[i, pool].min())
rows = [dict(name=names[i], group=int(groups[i]), split="eval" if is_eval[i] else "dev",
             **{"x_" + k: v for k, v in best(i, dev).items()}) for i in range(len(paths))]
df = pd.DataFrame(rows)
os.makedirs("splits", exist_ok=True)
df.to_csv("splits/groups.csv", index=False)
open("splits/dev.txt", "w").write("\n".join(df[df.split == "dev"].name) + "\n")
open("splits/eval.txt", "w").write("\n".join(df[df.split == "eval"].name) + "\n")

print("dev", (~is_eval).sum(), "eval", is_eval.sum(), "groups", len(set(groups)))
print("max cross-split similarity (eval -> nearest dev): thumb %.2f  hist %.2f  min phash %d"
      % (D.shape and C[np.ix_(ev, dev)].max(), H[np.ix_(ev, dev)].max(), D[np.ix_(ev, dev)].min()))
print("(merge thresholds: thumb>=%.2f, hist>=%.2f, phash<=%d)" % (split.THUMB_T, split.HIST_T, split.PHASH_T))
