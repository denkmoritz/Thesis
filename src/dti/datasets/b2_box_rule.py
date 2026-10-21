"""
Step 16 of the zone rebuild: larger boxes for the label-free training boxes (B).

The B boxes of step 15 sit on the insects but are smaller than the human boxes,
so YOLO trained on them draws boxes that are too small. Here several rules for
the box size are compared WITHOUT training, by how well the resulting boxes fit
the human labels. Positions come from the same motion peaks as in step 15.

Rules (blob = connected pixels around the peak that are at least t x the peak):
  blob t        bounding box of the blob, t = 0.50 (step 15), 0.35, 0.25, 0.15
  blob t x s    the same, enlarged by s = 1.5 (box drawn with some margin)
  dvs x r       DVS box size x r, r = 1.37 (focal ratio, no labels) and 0.68
                (median RGB/DVS size ratio, measured on labels)
All blob rules keep the cap of step 15 (at most DVS box x 1.37).

Printed per scene, for B boxes whose centre lies on a human label (circle rule):
  IoU >= 0.5   share that overlaps the human box well (what the IoU score needs)
  median IoU   and the median size ratio B / human box

Choosing a rule with these numbers uses human labels for one setting. To keep
it fair, the rule is chosen leave-one-scene-out: for each scene, the rule that
fits best on the other five scenes.

Run: PYTHONPATH=src ./venv/bin/python src/dti/datasets/b2_box_rule.py
"""

import json
import os

import cv2
import numpy as np

from dti.io.data_sync import DATA_DIR, RGB_W, RGB_H, load_rgb
from dti.geometry.rectify import load_dvs, setup
from dti.geometry import scene_correction as s4
from dti.geometry.zone import Zone, raw_to_rect
from dti.io import scene6 as s6
from dti.evaluation.metrics import score_matrix
from dti.motion.cue import read_frames, motion_maps, peaks
from dti.geometry.zone_v3 import (CORR, ALL_SCENES, Z_NEAR, NEG_SHARE_MAX, DISP_MARGIN, scene_stats,
                                         sigma_from)
from dti.datasets.setup1 import MIN_PEAK

OUT_DIR = "outputs/step16"
CACHE = f"{OUT_DIR}/cache.npz"
K = s4.K
HALF = 48  # px, motion crop around each peak
RULES = [("blob 0.50", "blob", 0.50, 1.0), ("blob 0.35", "blob", 0.35, 1.0), ("blob 0.25", "blob", 0.25, 1.0),
         ("blob 0.15", "blob", 0.15, 1.0), ("blob 0.50 x1.5", "blob", 0.50, 1.5), ("blob 0.35 x1.5", "blob", 0.35, 1.5),
         ("blob 0.25 x1.5", "blob", 0.25, 1.5), ("dvs x 1.37", "dvs", 1.37, 1.0), ("dvs x 0.68", "dvs", 0.68, 1.0)]


def collect():
    """Per B box candidate: scene, frame, peak position, DVS box size, motion crop; plus the human labels."""
    rect = setup()
    corr = json.load(open(CORR))
    optics = rect["KR"][1, 1] / np.load("data/output/stereo_calibration.npz")["KL"][1, 1]
    fy, fx = np.mgrid[0:RGB_H, 0:RGB_W]
    grid = raw_to_rect(rect, np.c_[fx.ravel(), fy.ravel()])
    data = {s: (load_dvs(s), s6.load_rgb() if s == s6.SCENE else load_rgb(s)) for s in ALL_SCENES}
    stats = {s: scene_stats(rect, *data[s], np.array(corr[s]["coef_rect_px"])) for s in ALL_SCENES}
    masks6 = s6.load_masks("rgb")
    rows, crops, labels = [], [], []
    for si, scene in enumerate(ALL_SCENES):
        dvs, rgb = data[scene]
        coef = np.array(corr[scene]["coef_rect_px"])
        others = [s for s in ALL_SCENES if s != scene]
        zone = Zone(rect, *sigma_from(np.concatenate([stats[s][0] for s in others]),
                                      np.concatenate([stats[s][1] for s in others])), optics)
        if corr[scene]["negative_disparity_share"] > NEG_SHARE_MAX:
            p5, _, p95 = corr[scene]["disparity_p5_p50_p95"]
            zone.d_lo, zone.d_hi = p5 - DISP_MARGIN, p95 + DISP_MARGIN
        else:
            zone.d_lo, zone.d_hi = 0.0, rect["fB_mm"] / Z_NEAR
        cap = cv2.VideoCapture(f"{DATA_DIR}/{scene}_rgb.mp4")
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for f in sorted(f for f in rgb if rgb[f]):
            for b in rgb[f]:
                labels.append((si, f, b[0], b[1], b[0] + b[2], b[1] + b[3]))
            recs = dvs.get(f + K, [])
            if not recs:
                continue
            m = motion_maps(read_frames(cap, f, n_video))["median"]
            if scene == s6.SCENE:
                for mk in masks6.values():
                    if any(g in mk["frames"] for g in range(f - 10, f + 11)):
                        m[max(0, mk["top"] - 10):mk["bottom"] + 10, max(0, mk["left"] - 10):mk["right"] + 10] = 0
            mp = np.pad(m, HALF)
            for r in recs:
                pk = peaks(m, zone.weights(grid, [r], coef, True).reshape(RGB_H, RGB_W) > 0.5, 1)
                if pk and m[pk[0][1], pk[0][0]] / 255 >= MIN_PEAK:
                    x, y = pk[0]
                    rows.append((si, f, x, y, r["box"][2], r["box"][3], optics))
                    crops.append(mp[y:y + 2 * HALF + 1, x:x + 2 * HALF + 1])
        cap.release()
        print(f"{scene}: {sum(1 for q in rows if q[0] == si)} B candidates", flush=True)
    np.savez_compressed(CACHE, rows=np.array(rows), crops=np.array(crops), labels=np.array(labels))


def box(rule, row, crop):
    _, kind, t, s = rule
    _, _, x, y, dw, dh, optics = row
    if kind == "dvs":
        return x, y, dw * t, dh * t
    win = (crop >= t * crop[HALF, HALF]).astype(np.uint8)
    _, lab = cv2.connectedComponents(win)
    ys, xs = np.nonzero(lab == lab[HALF, HALF])
    w = min((xs.max() - xs.min() + 1) * s, dw * optics)
    h = min((ys.max() - ys.min() + 1) * s, dh * optics)
    return x - HALF + (xs.min() + xs.max()) / 2, y - HALF + (ys.min() + ys.max()) / 2, max(w, 4.0), max(h, 4.0)


def fit_stats(rule, rows, crops, labels, scene_idx):
    """For B boxes on a human label: IoU >= 0.5 share, median IoU, median size ratio B / human."""
    ious, ratios = [], []
    by_frame = {}
    for lb in labels[labels[:, 0] == scene_idx]:
        by_frame.setdefault(int(lb[1]), []).append(lb[2:6])
    for row, crop in zip(rows, crops):
        if int(row[0]) != scene_idx:
            continue
        cx, cy, w, h = box(rule, row, crop)
        b = np.array([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 1.0]])
        g = np.c_[np.array(by_frame[int(row[1])]), np.ones(len(by_frame[int(row[1])]))]
        on = score_matrix(b, g, "circle")[0] > 0
        if not on.any():
            continue
        x1, y1 = np.maximum(b[0, 0], g[:, 0]), np.maximum(b[0, 1], g[:, 1])
        x2, y2 = np.minimum(b[0, 2], g[:, 2]), np.minimum(b[0, 3], g[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
        iou = inter / (w * h + area_g - inter)
        j = int(np.argmax(np.where(on, iou, -1)))
        ious.append(iou[j])
        ratios.append(np.sqrt(w * h / area_g[j]))
    return np.array(ious), np.array(ratios)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(CACHE):
        collect()
    c = np.load(CACHE)
    rows, crops, labels = c["rows"], c["crops"], c["labels"]
    table = {}
    for rule in RULES:
        for si in range(len(ALL_SCENES)):
            table[rule[0], si] = fit_stats(rule, rows, crops, labels, si)

    print(f"\nB boxes on a human label: share with IoU >= 0.5  (median size B / human)")
    print(f"{'rule':16s} " + " ".join(f"{s:>14s}" for s in ALL_SCENES))
    for rule in RULES:
        cells = []
        for si in range(len(ALL_SCENES)):
            iou, ratio = table[rule[0], si]
            cells.append(f"{np.mean(iou >= 0.5):5.0%} ({np.median(ratio):4.2f}x)")
        print(f"{rule[0]:16s} " + " ".join(f"{c:>14s}" for c in cells))

    print("\nRule chosen leave-one-scene-out (best IoU >= 0.5 share on the other five scenes, pooled):")
    for si, scene in enumerate(ALL_SCENES):
        def pooled(name):
            return np.mean(np.concatenate([table[name, o][0] for o in range(len(ALL_SCENES)) if o != si]) >= 0.5)
        best = max(RULES, key=lambda r: pooled(r[0]))[0]
        iou, _ = table[best, si]
        base, _ = table["blob 0.50", si]
        print(f"  {scene}: {best:16s} -> on this scene IoU >= 0.5: {np.mean(iou >= 0.5):4.0%} "
              f"(step-15 rule: {np.mean(base >= 0.5):4.0%})")


if __name__ == "__main__":
    main()
