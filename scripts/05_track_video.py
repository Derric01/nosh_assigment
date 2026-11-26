"""Task 4: segment + track the eggs in the video.
usage: python scripts/05_track_video.py [--video data/video/egg_video.mp4] [--out outputs/tracking]
Outputs: tracked.mp4 (overlay: masks, ids, trails), tracks.csv, summary.json"""
import sys, os, json, argparse, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2, numpy as np, pandas as pd
from eggseg import classical, tracking

# The image-dataset pipeline adapted to this video (see report, Task 4): a FIXED high threshold on
# 'whiteness' (V - 0.8 S) because the eggs are the only saturated-white objects on a dark table. Otsu
# picked the dim table reflections instead (observed on frames 323-484). Same shape filter as Task 2.
VIDEO_CFG = dict(feat="whiteness", thr="fixed", thr_val=200, blur=5, open_k=7, close_k=7,
                 min_area=0.004, max_area=0.1, min_solidity=0.9, min_ellipse_iou=0.85, max_axis_ratio=2.2)
PROC_W, PROC_H = 640, 360           # detection resolution
OUT_W, OUT_H = 1280, 720            # output resolution
COLORS = [(60, 60, 255), (255, 160, 40), (60, 220, 60), (0, 200, 255), (220, 60, 220), (200, 200, 0)]  # BGR


def table_fraction(frame):
    """Scene gate: fraction of the lower half that is dark brown (the table). Title cards, text cards and the
    recommended-videos screen at the end of this screen recording have none of it."""
    h = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)[45:]
    return ((h[..., 0] >= 3) & (h[..., 0] <= 25) & (h[..., 1] >= 60) & (h[..., 2] > 25) & (h[..., 2] < 150)).mean()


def draw(frame, tracks, detected_ids, frame_idx, on):
    img = cv2.resize(frame, (OUT_W, OUT_H))
    s = OUT_W / PROC_W
    ov = img.copy()
    for t in tracks:
        col = COLORS[(t.id - 1) % len(COLORS)]
        det = t.since_update == 0
        if det:
            cs, _ = cv2.findContours(t.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cs = [(c * s).astype(np.int32) for c in cs]
            cv2.drawContours(ov, cs, -1, col, cv2.FILLED)
    img = cv2.addWeighted(ov, 0.4, img, 0.6, 0)
    for t in tracks:
        col = COLORS[(t.id - 1) % len(COLORS)]
        det = t.since_update == 0
        if len(t.trail) > 1:
            pts = (np.array(t.trail[-45:]) * s).astype(np.int32)
            cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
        b = (t.box * s).astype(int)
        if det:
            cs, _ = cv2.findContours(t.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, [(c * s).astype(np.int32) for c in cs], -1, col, 2, cv2.LINE_AA)
        else:                                          # coasting on the Kalman prediction (e.g. hand over the egg)
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), col, 1, cv2.LINE_AA)
        label = f"egg {t.id}" + ("" if det else " (predicted)")
        cv2.putText(img, label, (b[0], max(b[1] - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (b[0], max(b[1] - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
    cv2.putText(img, f"frame {frame_idx}  tracking {'ON' if on else 'OFF (no egg scene)'}", (14, OUT_H - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/video/egg_video.mp4"); ap.add_argument("--out", default="outputs/tracking")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "27", "-preset", "medium",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", os.path.join(a.out, "tracked.mp4")],
                          stdin=subprocess.PIPE)
    tr, rows, off_run = tracking.Tracker(PROC_W), [], 0
    for fi in range(N):
        ok, frame = cap.read()
        if not ok:
            break
        on = table_fraction(frame) > 0.25
        small = cv2.resize(frame, (PROC_W, PROC_H))
        if on:
            off_run = 0
            inst = classical.segment(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), VIDEO_CFG)
            tracks = tr.step([inst == i for i in range(1, inst.max() + 1)])
        else:
            off_run += 1
            if off_run > 15 and tr.tracks:             # scene really ended: forget old tracks (ids restart)
                tr = tracking.Tracker(PROC_W)
            tracks = []
        for t in tracks:
            b = t.box
            rows.append(dict(frame=fi, id=t.id, cx=(b[0] + b[2]) / 2, cy=(b[1] + b[3]) / 2, w=b[2] - b[0], h=b[3] - b[1],
                             state="detected" if t.since_update == 0 else "predicted"))
        ff.stdin.write(draw(frame, tracks, None, fi, on).tobytes())
        if fi % 200 == 0:
            print(f"frame {fi}/{N}  tracks {[t.id for t in tracks]}", flush=True)
    ff.stdin.close(); ff.wait()
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "tracks.csv"), index=False)
    life = df.groupby("id").agg(first=("frame", "min"), last=("frame", "max"), frames=("frame", "count"),
                                detected=("state", lambda s: int((s == "detected").sum())))
    print(life)
    json.dump(dict(frames=N, fps=fps, tracks=int(df.id.nunique()), lifetimes=life.reset_index().to_dict("records")),
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
