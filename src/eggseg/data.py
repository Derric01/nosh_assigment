"""Dataset helpers. The Roboflow export is a classification-folder layout
(train/valid/test, all one 'egg' class) with NO boxes or masks - so every mask
in this project comes from SAM (Task 1) and is stored as a PNG per image."""
import glob, os
import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "data", "raw", "egg")
SAM_DIR = os.path.join(ROOT, "outputs", "sam_masks")


def list_images():
    """All 110 images, sorted. Roboflow's own train/valid/test split is ignored:
    we make our own leakage-aware 4:1 split in split.py."""
    fs = glob.glob(os.path.join(RAW, "*", "*", "*.jpg"))
    return sorted(fs, key=stem)


def stem(path):
    """'not_damaged_100_jpg.rf.<hash>.jpg' -> 'not_damaged_100'."""
    return os.path.basename(path).split("_jpg")[0]


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def load_instance_mask(name):
    """Instance-id PNG written by 01_run_sam.py: 0 = background, 1..k = eggs."""
    return cv2.imread(os.path.join(SAM_DIR, name + ".png"), cv2.IMREAD_UNCHANGED)


def read_split(which):
    with open(os.path.join(ROOT, "splits", which + ".txt")) as f:
        return [l.strip() for l in f if l.strip()]
