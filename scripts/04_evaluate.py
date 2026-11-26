"""Task 3: evaluate the FROZEN final classical algorithm against the SAM masks.
usage: python scripts/04_evaluate.py            (reads experiments/final_config.json)
Eval split is read here for the first time; the config must already be frozen (see final_config.json)."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eggseg import data, classical, metrics, viz

fin = json.load(open("experiments/final_config.json")); cfg = fin["params"]
OUT = "outputs/eval"; os.makedirs(OUT, exist_ok=True)
by = {data.stem(p): p for p in data.list_images()}
fixed = {"not_damaged_" + k for k in json.load(open("scripts/sam_manual_fixes.json")) if not k.startswith("_")}


def describe(rgb, gt):
    """image/egg descriptors used to explain failures"""
    g = gt > 0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32); hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ring = (cv2.dilate(g.astype(np.uint8), np.ones((19, 19), np.uint8)) > 0) & ~g
    de = float(np.linalg.norm(lab[g].mean(0) - lab[ring].mean(0))) if g.any() and ring.any() else np.nan
    bg = ~(cv2.dilate(g.astype(np.uint8), np.ones((19, 19), np.uint8)) > 0)
    return dict(egg_area=float(g.mean()), egg_sat=float(hsv[..., 1][g].mean()), egg_val=float(hsv[..., 2][g].mean()),
                contrast_dE=de, bg_val=float(hsv[..., 2][bg].mean()) if bg.any() else np.nan,
                bg_texture=float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_32F)[bg].std()) if bg.any() else np.nan)


def category(r):
    if r["n_pred"] == 0: return "missed (nothing found)"
    if r["iou"] >= 0.8: return "good (IoU>=0.8)"
    if r["prec"] < 0.5 and r["rec"] >= 0.5: return "leak into background"
    if r["rec"] < 0.5 and r["prec"] >= 0.5: return "under-segmented egg"
    if r["iou"] < 0.2: return "wrong object / missed egg"
    return "partially right"


def run(split):
    rows, cache = [], {}
    for n in data.read_split(split):
        rgb = data.load_rgb(by[n]); gt = data.load_instance_mask(n); pr = classical.segment(rgb, cfg)
        r = metrics.evaluate(gt, pr); r.update(describe(rgb, gt)); r.update(name=n, split=split, manual_gt=n in fixed)
        r["category"] = category(r); rows.append(r); cache[n] = (rgb, gt, pr)
    return pd.DataFrame(rows), cache


dfs, caches = {}, {}
for sp in ("eval", "dev"):
    dfs[sp], caches[sp] = run(sp)
    dfs[sp].to_csv(f"{OUT}/{sp}_per_image.csv", index=False)

cols = ["iou", "dice", "prec", "rec", "bf", "hd95", "inst_prec", "inst_rec", "inst_f1", "matched_iou"]
summ = {}
for sp, d in dfs.items():
    tp, fp, fn = d.tp.sum(), d.fp.sum(), d.fn.sum()
    s = {c: float(d[c].mean()) for c in cols}
    s.update(iou_median=float(d.iou.median()), iou_std=float(d.iou.std()), micro_iou=float(tp / (tp + fp + fn)),
             count_acc=float((d.n_gt == d.n_pred).mean()), mean_count_err=float((d.n_pred - d.n_gt).abs().mean()),
             frac_iou_ge_0_5=float((d.iou >= 0.5).mean()), frac_iou_ge_0_8=float((d.iou >= 0.8).mean()), n_images=len(d))
    summ[sp] = s
json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
print(pd.DataFrame(summ).round(3).to_string())

# ---------- plots ----------
ev, dv = dfs["eval"], dfs["dev"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
bins = np.linspace(0, 1, 11)
ax[0].hist([dv.iou, ev.iou], bins=bins, label=["dev", "eval"], color=["#9aa5b1", "#d1495b"]); ax[0].set(xlabel="per-image IoU", ylabel="images", title="IoU distribution"); ax[0].legend()
e = ev.sort_values("iou"); ax[1].bar(range(len(e)), e.iou, color=["#d1495b" if v < 0.5 else "#edae49" if v < 0.8 else "#3b8b5a" for v in e.iou])
ax[1].set(xlabel="eval images (sorted)", ylabel="IoU", title="Eval IoU per image (red<0.5, amber<0.8)")
cat = pd.concat([ev, dv], ignore_index=True).groupby(["split", "category"]).size().unstack(0).fillna(0)
cat.plot.barh(ax=ax[2], color=["#9aa5b1", "#d1495b"]); ax[2].set(title="Failure taxonomy (images)", xlabel="images")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_overview.png", dpi=130); plt.close()

al = pd.concat([ev, dv], ignore_index=True)
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, (x, xl, lg) in zip(ax, [("egg_area", "egg area fraction", True), ("contrast_dE", "egg-vs-surround colour contrast (Lab dE)", False),
                               ("bg_texture", "background texture (Laplacian std)", False), ("egg_sat", "egg mean saturation", False)]):
    a.scatter(al[x], al.iou, c=np.where(al.split == "eval", "#d1495b", "#9aa5b1"), s=22, alpha=.85)
    a.set(xlabel=xl, ylabel="IoU", ylim=(-.02, 1.02), title=f"IoU vs {x}");
    if lg: a.set_xscale("log")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_iou_vs_factors.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].bar(["FP px", "FN px"], [ev.fp.sum() / 1e3, ev.fn.sum() / 1e3], color=["#d1495b", "#2e6fbb"]); ax[0].set(ylabel="kilo-pixels (eval)", title="Error pixels: false positive vs missed")
ev.assign(n=ev.n_gt).boxplot(column="iou", by="n", ax=ax[1]); ax[1].set(title="IoU by number of GT eggs", xlabel="eggs in image"); plt.suptitle("")
ct = pd.crosstab(al.n_gt, al.n_pred); im = ax[2].imshow(ct.values, cmap="Blues"); ax[2].set(xticks=range(len(ct.columns)), xticklabels=ct.columns, yticks=range(len(ct.index)), yticklabels=ct.index, xlabel="predicted eggs", ylabel="SAM eggs", title="Egg-count confusion (dev+eval)")
for (i, j), v in np.ndenumerate(ct.values): ax[2].text(j, i, int(v), ha="center", va="center")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_errors.png", dpi=130); plt.close()

# ---------- qualitative: worst / best eval images, 4 panels each ----------
def panel(n, sp="eval"):
    rgb, gt, pr = caches[sp][n]
    r = dfs[sp].set_index("name").loc[n]
    return np.hstack([viz.label(cv2.resize(x, (320, 320)), t) for x, t in
                      [(rgb, n.replace("not_damaged_", "img ")), (viz.overlay(rgb, gt), "SAM (GT)"), (viz.overlay(rgb, pr), "classical"),
                       (viz.error_overlay(rgb, gt, pr), f"IoU {r.iou:.2f} {r.category[:14]}")]])
order = ev.sort_values("iou")
for tag, names in (("worst", order.name[:8]), ("best", order.name[-3:])):
    cv2.imwrite(f"{OUT}/gallery_{tag}.jpg", cv2.cvtColor(np.vstack([panel(n) for n in names]), cv2.COLOR_RGB2BGR))
print(ev.category.value_counts().to_string())
