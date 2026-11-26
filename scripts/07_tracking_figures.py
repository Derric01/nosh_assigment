"""Figures for the report's Task 4 section: track timeline + sample frames of the tracked video."""
import cv2, numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv("outputs/tracking/tracks.csv"); N = json.load(open("outputs/tracking/summary.json"))["frames"]
fig, ax = plt.subplots(figsize=(13, 2.6))
col = {1: "#d1495b", 2: "#2e6fbb", 3: "#3b8b5a"}
for i, g in d.groupby("id"):
    for st, mk, a in (("detected", "|", 1.0), ("predicted", "|", 0.35)):
        s = g[g.state == st]; ax.scatter(s.frame, [i] * len(s), marker=mk, s=60, color=col.get(i, "k"), alpha=a, label=f"egg {i} {st}" if i == 1 else None)
ax.set(yticks=sorted(d.id.unique()), yticklabels=[f"egg {i}" for i in sorted(d.id.unique())], xlim=(0, N), xlabel="frame",
       title="Track timeline (dark = detected in the frame, light = coasting on the Kalman prediction; blank = lost / no egg scene)")
ax.axvspan(0, 177, color="#eee"); ax.axvspan(1151, N, color="#eee"); ax.grid(alpha=.3, axis="x")
plt.tight_layout(); plt.savefig("report/figures/fig_track_timeline.png", dpi=130)

cap = cv2.VideoCapture("outputs/tracking/tracked.mp4"); tiles = []
for fi in [250, 480, 520, 606, 700, 830, 920, 1100]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, f = cap.read(); tiles.append(cv2.resize(f, (480, 270)))
cv2.imwrite("report/figures/tracking_frames.jpg", np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))
print("ok")
