"""
Step 1 of the zone rebuild: data + sync check.

For a few dataset frames per scene, show the DVS frame and the RGB frame of the
same index side by side:
  - DVS frame f with the DVS annotation of frame f (cyan) and of frame f-1 (magenta)
  - RGB frame f with its YOLO labels (green)

Timing: the two cameras see the scene from different poses, so DVS and RGB
pixels cannot be compared directly here. Step 1 only shows how far insects move
between f-1 and f (in DVS px); the actual f / f-1 test is done in step 3, where
the rectified row offset does not depend on depth.

Run: PYTHONPATH=src ./venv/bin/python src/dti/io/data_sync.py
"""

import csv
import glob
import os
from collections import defaultdict

import cv2
import numpy as np

from dti.paths import TRAINING_ROOT

TRAIN = TRAINING_ROOT
DATA_DIR = f"{TRAIN}/data"
DATASET_DIR = f"{TRAIN}/dataset"
OUT_DIR = "outputs/step1"
SCENES = ["1_l-l-l", "2_l-h-l", "3_m-h-h", "4_m-m-h", "5_h-l-h"]
RGB_W, RGB_H = 1920, 1200
SHIFTS = [-2, -1, 0, 1]

CYAN, MAGENTA, GREEN, WHITE = (255, 255, 0), (255, 0, 255), (0, 255, 0), (255, 255, 255)


def load_dvs(scene):
    """frame -> list of (left, top, w, h) boxes in DVS pixels, 'certain' only."""
    boxes = defaultdict(list)
    with open(f"{DATA_DIR}/{scene}_annotation.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["confidence"] == "certain":
                boxes[int(r["frame_index"])].append(
                    tuple(float(r[k]) for k in ("left", "top", "width", "height")))
    return boxes


def load_rgb(scene):
    """frame -> list of (left, top, w, h) boxes in RGB pixels, from YOLO labels."""
    labels = {}
    for img in glob.glob(f"{DATASET_DIR}/images/*/{scene}_frame_*.jpg"):
        split = img.split("/")[-2]
        stem = os.path.basename(img)[:-4]
        boxes = []
        with open(f"{DATASET_DIR}/labels/{split}/{stem}.txt") as f:
            for line in f:
                _, cx, cy, w, h = map(float, line.split()[:5])
                boxes.append(((cx - w / 2) * RGB_W, (cy - h / 2) * RGB_H, w * RGB_W, h * RGB_H))
        labels[int(stem[-6:])] = boxes
    return labels


def center(b):
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def read_frame(scene, cam, index):
    cap = cv2.VideoCapture(f"{DATA_DIR}/{scene}_{cam}.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    assert ok, (scene, cam, index)
    return frame


def motion(dvs, f):
    """Displacement (DVS px) of the single insect from f-1 to f, or None."""
    if len(dvs.get(f, [])) == 1 and len(dvs.get(f - 1, [])) == 1:
        return float(np.hypot(*np.subtract(center(dvs[f][0]), center(dvs[f - 1][0]))))
    return None


def pick_frames(rgb, dvs):
    """Frame with most insects, fastest single insect, and the median dataset frame."""
    frames = sorted(f for f in rgb if rgb[f])
    busiest = max(frames, key=lambda f: (len(rgb[f]) + len(dvs.get(f, [])), -f))
    fastest = max(frames, key=lambda f: motion(dvs, f) or -1)
    picks = [busiest, fastest]
    middle = [f for f in frames if f not in picks]
    return picks + [middle[len(middle) // 2]]


def draw_box(img, b, color, scale, thick):
    x, y, w, h = (v * scale for v in b)
    pad = max(0, (12 - min(w, h)) / 2)  # keep tiny insects visible after downscaling
    cv2.rectangle(img, (int(x - pad), int(y - pad)), (int(x + w + pad), int(y + h + pad)), color, thick)


def render(scene, f, dvs, rgb):
    out_h = 600
    s_dvs, s_rgb = out_h / 720, out_h / RGB_H
    dvs_img = cv2.resize(read_frame(scene, "dvs", f), None, fx=s_dvs, fy=s_dvs, interpolation=cv2.INTER_AREA)
    rgb_img = cv2.resize(read_frame(scene, "rgb", f), None, fx=s_rgb, fy=s_rgb, interpolation=cv2.INTER_AREA)

    for b in dvs.get(f - 1, []):
        draw_box(dvs_img, b, MAGENTA, s_dvs, 2)
    for b in dvs.get(f, []):
        draw_box(dvs_img, b, CYAN, s_dvs, 1)
    for b in rgb.get(f, []):
        draw_box(rgb_img, b, GREEN, s_rgb, 2)
    canvas = np.hstack([dvs_img, np.full((out_h, 8, 3), 40, np.uint8), rgb_img])
    lines = [
        f"{scene}  frame {f}   left: DVS frame {f} (1280x720)   right: RGB frame {f} (1920x1200), same index",
        f"DVS: cyan box = DVS annotation frame {f} ({len(dvs.get(f, []))}), magenta box = DVS annotation frame {f-1} "
        f"({len(dvs.get(f-1, []))}), certain only; tiny boxes padded to 12 px",
        f"RGB: green box = YOLO label ({len(rgb.get(f, []))});  single-insect motion f-1 -> f: "
        + (f"{motion(dvs, f):.1f} DVS px" if motion(dvs, f) is not None else "n/a (not exactly one insect)"),
    ]
    cap = np.zeros((22 * len(lines) + 10, canvas.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(cap, t, (8, 22 * i + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.imwrite(f"{OUT_DIR}/{scene}_frame_{f:06d}.jpg", np.vstack([canvas, cap]), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Per scene: dataset frames, RGB labels, DVS boxes (certain) at those frames,")
    print("and motion of single insects f-1 -> f over ALL annotated frames (DVS px)")
    print(f"{'scene':10s} {'frames':>6s} {'labels':>6s} {'dvs':>5s} {'n':>6s} {'median':>7s} {'p90':>6s} {'>3px':>6s}")
    for scene in SCENES:
        dvs, rgb = load_dvs(scene), load_rgb(scene)
        m = np.array([v for v in (motion(dvs, f) for f in dvs) if v is not None])
        n_lab = sum(len(v) for v in rgb.values())
        n_dvs = sum(len(dvs.get(f, [])) for f in rgb)
        print(f"{scene:10s} {len(rgb):6d} {n_lab:6d} {n_dvs:5d} {len(m):6d} {np.median(m):7.1f} "
              f"{np.percentile(m, 90):6.1f} {np.mean(m > 3):6.0%}")
        picks = pick_frames(rgb, dvs)
        for f in picks:
            render(scene, f, dvs, rgb)
        print(f"{'':10s} images: frames {picks}  RGB labels {[len(rgb[f]) for f in picks]}  "
              f"DVS f {[len(dvs.get(f, [])) for f in picks]}  DVS f-1 {[len(dvs.get(f - 1, [])) for f in picks]}  "
              f"motion {[None if motion(dvs, f) is None else round(motion(dvs, f), 1) for f in picks]}")


if __name__ == "__main__":
    main()
