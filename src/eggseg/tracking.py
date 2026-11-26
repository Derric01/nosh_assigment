"""Task 4 - multi-egg tracker (SORT-style: Kalman filter + Hungarian matching).

Each egg is a Track with a constant-velocity Kalman filter over its box (cx, cy, w, h).
Per frame:
  1. predict every track one step ahead                       (where should the egg be now?)
  2. cost[track, detection] = 0.5*(1 - IoU) + 0.5*dist/gate    (overlap + centre distance, gated)
  3. Hungarian assignment minimises total cost                 (one detection <-> one track)
  4. matched tracks are corrected with the detection; unmatched detections start new (tentative) tracks;
     unmatched tracks keep coasting on the prediction for up to `max_age` frames (hand occlusion,
     motion blur); after that they become 'lost' but are remembered for `max_lost` frames.
  5. A tentative track is confirmed after `min_hits` detections. At that moment it is compared with the
     lost tracks: if one is close enough (position + size) the egg is considered to be that same egg
     coming back and INHERITS its id (re-identification after a long occlusion); otherwise it gets a new id.
"""
import numpy as np, cv2
from scipy.optimize import linear_sum_assignment


def box_of(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


class Track:
    _next_id = 1

    def __init__(self, box, mask):
        cx, cy, w, h = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]
        kf = cv2.KalmanFilter(8, 4)                       # state: cx cy w h vx vy vw vh
        kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            kf.transitionMatrix[i, i + 4] = 1.0           # position += velocity
        kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        kf.processNoiseCov = np.diag([1, 1, 1, 1, 4, 4, 1, 1]).astype(np.float32) * 1e-1
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 4.0
        kf.errorCovPost = np.eye(8, dtype=np.float32) * 10.0
        kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], np.float32)
        self.kf, self.mask = kf, mask
        self.id = None                                    # assigned when confirmed (keeps ids 1,2,3 contiguous)
        self.hits, self.since_update, self.age = 1, 0, 0
        self.trail = [(cx, cy)]

    def predict(self):
        self.kf.statePost[4:] *= 0.8                      # damp velocity while coasting so predictions do not run away
        s = self.kf.predict().ravel(); self.age += 1; self.since_update += 1
        return np.array([s[0] - s[2] / 2, s[1] - s[3] / 2, s[0] + s[2] / 2, s[1] + s[3] / 2])

    def update(self, box, mask):
        cx, cy, w, h = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]
        self.kf.correct(np.array([[cx], [cy], [w], [h]], np.float32))
        self.hits += 1; self.since_update = 0; self.mask = mask
        self.trail.append((cx, cy))

    @property
    def box(self):
        s = self.kf.statePost.ravel()
        return np.array([s[0] - s[2] / 2, s[1] - s[3] / 2, s[0] + s[2] / 2, s[1] + s[3] / 2])


class Tracker:
    def __init__(self, frame_w, max_age=20, min_hits=8, gate=1.5, max_lost=900, reid_dist=0.7):
        self.W, self.max_age, self.min_hits, self.gate = frame_w, max_age, min_hits, gate
        self.max_lost, self.reid_dist = max_lost, reid_dist
        self.tracks, self.next_id = [], 1

    def _reid_cost(self, lost, new):
        """position + size mismatch between a lost track's last box and a new track's box (0 = identical)."""
        a, b = lost.box, new.box
        d = np.hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2, (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2) / self.W
        sz = abs(np.log(max(b[2] - b[0], 1) / max(a[2] - a[0], 1))) + abs(np.log(max(b[3] - b[1], 1) / max(a[3] - a[1], 1)))
        return d + 0.5 * sz, d

    def step(self, masks):
        """masks: list of boolean egg masks in this frame. Returns confirmed tracks (detected or coasting)."""
        dets = [(box_of(m), m) for m in masks]
        active = [t for t in self.tracks if t.id is None or t.since_update <= self.max_age]
        lost = [t for t in self.tracks if t.id is not None and t.since_update > self.max_age]
        preds = [t.predict() for t in active]
        for t in lost:
            t.since_update += 1                                # lost tracks only age; they are not predicted/matched
        cost = np.full((len(active), len(dets)), 1e6)
        for i, (t, p) in enumerate(zip(active, preds)):
            size = max(p[2] - p[0], p[3] - p[1], 1)
            pc = ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2)
            for j, (b, _) in enumerate(dets):
                d = np.hypot((b[0] + b[2]) / 2 - pc[0], (b[1] + b[3]) / 2 - pc[1])
                if d < self.gate * size:                      # gate: far-away detections are never matched
                    cost[i, j] = 0.5 * (1 - iou(p, b)) + 0.5 * d / (self.gate * size)
        matched_d = set()
        if len(active) and len(dets):
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e5:
                    active[i].update(*dets[j]); matched_d.add(j)
        new = [Track(b, m) for j, (b, m) in enumerate(dets) if j not in matched_d]
        self.tracks = active + lost + new
        # drop dead tracks: tentative ones after 1 miss, confirmed ones after max_lost frames
        self.tracks = [t for t in self.tracks if (t.since_update <= 1 if t.id is None else t.since_update <= self.max_lost)]
        # confirm tentative tracks (with re-identification against lost tracks)
        for t in [t for t in self.tracks if t.id is None and t.hits >= self.min_hits]:
            cands = [(self._reid_cost(l, t), l) for l in self.tracks
                     if l.id is not None and l.since_update > self.max_age and l is not t]
            cands = [(c, l) for (c, d), l in cands if d < self.reid_dist]
            if cands:
                c, l = min(cands, key=lambda x: x[0])
                t.id = l.id; t.trail = [t.trail[-1]]           # same egg again: inherit id, restart the trail segment
                self.tracks.remove(l)
            else:
                t.id = self.next_id; self.next_id += 1
        return [t for t in self.tracks if t.id is not None and t.since_update <= self.max_age]
