"""
Step 8b of the zone rebuild: scene 6 (6_h-h-h) through the checks of steps 1 and 3.

Scene 6 was labelled after steps 1-8. This script brings it to the same state
as scenes 1-5 without touching their outputs (own folder, nothing overwritten,
nothing fitted):

  [1] data: labelled frames, RGB labels, DVS boxes at those frames, insect motion
  [2] rectified row offset RGB - DVS and disparity (calibration from step 2 only)
      - single = frames with exactly one DVS insect and one RGB label
      - best case = every RGB label paired with the DVS insect that gives the
        smallest total |row offset| (cannot be explained by wrong pairing)
  [3] timing: DVS frame f+k vs RGB frame f (step 10 predicts k = 0 for scene 6,
      because its exposure of 7.8 ms covers most of DVS frame f)
  [4] row offset by image position and inside / outside the calibration board area
  [5] privacy masks: labels or DVS boxes inside a masked rectangle

RGB labels are read from training/annotations/6_h-h-h_annotations (CVAT export),
not from training/dataset.

Run: PYTHONPATH=src ./venv/bin/python src/dti/io/scene6.py
"""

import glob
import json
import os

import cv2
import numpy as np

from dti.io.data_sync import TRAIN, RGB_W, RGB_H, center, pick_frames
from dti.geometry import rectify as s3
from dti.geometry.scene_correction import match
from dti.paths import RAW_DATA_DIR

SCENE = "6_h-h-h"
LABEL_DIR = f"{TRAIN}/annotations/{SCENE}_annotations/labels/train"
MASK_DIR = RAW_DATA_DIR
OUT_DIR = "outputs/step8b"
SHIFTS = [-2, -1, 0, 1]


def load_rgb():
    """frame -> list of (left, top, w, h) boxes in RGB pixels, from the CVAT export."""
    labels = {}
    for path in glob.glob(f"{LABEL_DIR}/frame_*.txt"):
        boxes = []
        with open(path) as f:
            for line in f:
                _, cx, cy, w, h = map(float, line.split()[:5])
                boxes.append(((cx - w / 2) * RGB_W, (cy - h / 2) * RGB_H, w * RGB_W, h * RGB_H))
        labels[int(os.path.basename(path)[6:-4])] = boxes
    return labels


def best_case(rect, dvs, rgb, k):
    """Row offset (RGB px) and disparity (rect px) with the pairing that minimises the total |row offset|."""
    dy, disp, xy = [], [], []
    for f in sorted(rgb):
        recs = dvs.get(f + k, [])
        qd = s3.rect_dvs(rect, [r["c"] for r in recs]) if recs else np.zeros((0, 2))
        for q_rgb, row_dvs, _ in match(rect, recs, rgb[f], None):
            j = int(np.argmin(np.abs(qd[:, 1] - row_dvs)))
            dy.append((q_rgb[1] - row_dvs) * rect["to_rgb"])
            disp.append(qd[j, 0] - q_rgb[0])
            xy.append(q_rgb)
    return np.array(dy), np.array(disp), np.array(xy).reshape(-1, 2)


def in_masks(masks, f, x, y):
    return any(f in m["frames"] and m["left"] <= x <= m["right"] and m["top"] <= y <= m["bottom"]
               for m in masks.values())


def load_masks(cam):
    masks = json.load(open(f"{MASK_DIR}/{SCENE}_{cam}_masks.json"))
    for m in masks.values():
        m["frames"] = set(m["frames"])
    return masks


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    s3.OUT_DIR = OUT_DIR
    rect = s3.setup()
    dvs, rgb = s3.load_dvs(SCENE), load_rgb()
    frames = sorted(rgb)

    print(f"[1] {SCENE}: {len(frames)} labelled frames ({frames[0]}..{frames[-1]}), "
          f"{sum(len(v) for v in rgb.values())} RGB labels")
    for k in (0, -1):
        recs = [r for f in frames for r in dvs.get(f + k, [])]
        print(f"    DVS boxes (certain) at frame f{k:+d}: {len(recs)}, hand-placed keyframes "
              f"{np.mean([r['key'] for r in recs]):.0%}")
    n_dvs = [len(dvs.get(f, [])) for f in frames]
    print(f"    frames with exactly one DVS insect and one RGB label: "
          f"{sum(len(rgb[f]) == 1 and n == 1 for f, n in zip(frames, n_dvs))};  "
          f"frames with more DVS insects than RGB labels: {sum(n > len(rgb[f]) for f, n in zip(frames, n_dvs))}")
    mot = np.array([m for m in (s3.motion(dvs, f) for f in dvs) if m > 0])
    print(f"    motion of single insects f-1 -> f over all annotated frames (DVS px): n={len(mot)}, "
          f"median {np.median(mot):.1f}, p90 {np.percentile(mot, 90):.1f}, >3 px {np.mean(mot > 3):.0%}")
    sizes = np.array([b[2:] for v in rgb.values() for b in v])
    print(f"    RGB label size: median {np.median(sizes[:, 0]):.0f} x {np.median(sizes[:, 1]):.0f} px")

    print(f"\n[2] Row offset RGB - DVS (RGB px), disparity and depth; 1 rect px = {rect['to_rgb']:.2f} RGB px")
    print(f"    {'pairs':22s} {'row offset':40s} {'|off|<5px':>9s} {'disp<=0':>8s} {'depth median, IQR (m)':>24s}")
    for k in (0, -1):
        single = s3.measure(rect, s3.single_pairs(dvs, rgb, k))
        best = best_case(rect, dvs, rgb, k)[:2]
        for name, (dy, disp) in ((f"single, DVS f{k:+d}", single), (f"best case, DVS f{k:+d}", best)):
            z = rect["fB_mm"] / disp[disp > 0] / 1000
            print(f"    {name:22s} {s3.fmt(dy):40s} {np.mean(np.abs(dy) < 5):9.0%} {np.mean(disp <= 0):8.0%} "
                  f"{np.median(z):8.2f}  {np.percentile(z, 25):5.2f}..{np.percentile(z, 75):5.2f}")

    print("\n[3] Timing: median |row offset| (RGB px) for DVS frame f+k vs RGB frame f")
    print("    single = one DVS insect, one RGB label; key+mov = DVS box hand-placed and insect moves > 3 DVS px/frame")
    print(f"    {'k':>3s} {'single':>12s} {'key+mov':>12s} {'best case':>12s}")
    for k in SHIFTS:
        pairs = s3.single_pairs(dvs, rgb, k)
        sel = [p for p in pairs if p[1]["key"] and s3.motion(dvs, p[0] + k) > 3]
        a, s = np.abs(s3.measure(rect, pairs)[0]), np.abs(s3.measure(rect, sel)[0])
        b = np.abs(best_case(rect, dvs, rgb, k)[0])
        print(f"    {k:+3d} {np.median(a):6.1f} ({len(a):3d}) "
              f"{np.median(s) if len(s) else float('nan'):6.1f} ({len(s):3d}) {np.median(b):6.1f} ({len(b):3d})")

    print("\n[4] Row offset (RGB px) by image position, best-case pairs, DVS frame f")
    dy, _, xy = best_case(rect, dvs, rgb, 0)
    w, h = rect["size"]
    for name, v, size in (("left / middle / right third", xy[:, 0], w), ("top / middle / bottom third", xy[:, 1], h)):
        parts = [dy[(v >= size * i / 3) & (v < size * (i + 1) / 3)] for i in range(3)]
        print(f"    {name:28s} " + "   ".join(s3.fmt(p) for p in parts))
    hull = s3.board_hull()
    pairs = s3.single_pairs(dvs, rgb, 0)
    sdy, _ = s3.measure(rect, pairs)
    inside = np.array([cv2.pointPolygonTest(hull, center(p[2]), False) >= 0 for p in pairs], bool)
    print(f"    calibration board area (single pairs): inside {s3.fmt(sdy[inside])}   outside {s3.fmt(sdy[~inside])}")

    print("\n[5] Privacy masks")
    m_rgb, m_dvs = load_masks("rgb"), load_masks("dvs")
    masked_frames = [f for f in frames if any(f in m["frames"] for m in m_rgb.values())]
    n_lab = sum(in_masks(m_rgb, f, *center(b)) for f in frames for b in rgb[f])
    n_box = sum(in_masks(m_dvs, f, *r["c"]) for f in frames for r in dvs.get(f, []))
    print(f"    labelled frames with an RGB mask: {len(masked_frames)} of {len(frames)};  "
          f"RGB labels inside a mask: {n_lab};  DVS boxes inside a DVS mask: {n_box}")

    print("\n[6] Images (rectified, same frame index)")
    picks = pick_frames(rgb, {f: [r["box"] for r in v] for f, v in dvs.items()})
    for f in picks:
        s3.render(rect, SCENE, f, dvs, rgb)
    print(f"    frames {picks} -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
