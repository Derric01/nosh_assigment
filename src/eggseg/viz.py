"""Overlay / contact-sheet helpers."""
import cv2, numpy as np

PALETTE = [(255, 0, 60), (0, 170, 255), (255, 200, 0), (0, 220, 120), (200, 0, 255), (255, 120, 0)]


def overlay(rgb, inst, alpha=0.55):
    """Colour each instance + draw its outline (like the red-egg example in the brief)."""
    out = rgb.copy().astype(np.float32)
    for i in np.unique(inst):
        if i == 0:
            continue
        m = inst == i
        col = np.array(PALETTE[(i - 1) % len(PALETTE)], np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * col
    out = out.astype(np.uint8)
    for i in np.unique(inst):
        if i == 0:
            continue
        cs, _ = cv2.findContours((inst == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, (255, 255, 255), 1)
    return out


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def sheet(tiles, ncols, size=256):
    """Grid of equally sized RGB tiles."""
    tiles = [cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA) for t in tiles]
    while len(tiles) % ncols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + ncols]) for i in range(0, len(tiles), ncols)]
    return np.vstack(rows)


def error_overlay(rgb, gt, pr, alpha=0.6):
    """TP green, FP red, FN blue (foreground = any egg)."""
    g, p = gt > 0, pr > 0
    out = rgb.astype(np.float32).copy()
    for m, col in (((g & p), (0, 200, 0)), ((~g & p), (255, 0, 0)), ((g & ~p), (0, 80, 255))):
        out[m] = (1 - alpha) * out[m] + alpha * np.array(col, np.float32)
    return out.astype(np.uint8)
