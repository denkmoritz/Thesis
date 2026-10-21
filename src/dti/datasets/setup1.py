"""
Step 15 of the zone rebuild: dataset for YOLO training with motion as input
("setup 1"), with two kinds of training boxes (A and B), leave-one-scene-out.

Image for every labelled frame of the six scenes (PNG, lossless, 1920x1200):
  red channel    grey RGB frame
  green channel  motion picture: |frame - median of 11 frames around it|, x 2
  blue channel   zone (0 .. 255 = weight 0 .. 1), from the DVS annotation of frame
                 f-1 (certain) with the zone v3 of step 14 (label-free correction)

Training boxes:
  A  the human RGB labels
  B  no human label: for every DVS insect the strongest motion peak in its zone,
     kept if the peak is stronger than MIN_PEAK (above the motion seen in frames
     without insects, step 11f). Box = extent of the motion blob around the peak
     (pixels at least half as strong as the peak), at most DVS box x 1.37.

Folds: for each scene k, train = all frames of the other five scenes with boxes
A or B, test = all frames of scene k with the human labels (used for scoring
only). Layout written to training/dataset_m3/ (paths inside are PALMA paths):
  A/images/all/{stem}.png   A/labels/all/{stem}.txt
  B/images -> ../A/images   B/labels/all/{stem}.txt
  folds/{scene}_{A|B}_train.txt, {scene}_{A|B}_test.txt, {scene}_{A|B}.yaml

Printed: per scene the quality of the B boxes measured against the human labels
(information only, not used for anything), and the dataset size.

Run: PYTHONPATH=src ./venv/bin/python src/dti/datasets/setup1.py
"""

import json
import os

import cv2
import numpy as np

from dti.io.data_sync import DATA_DIR, SCENES, RGB_W, RGB_H, load_rgb
from dti.geometry.rectify import load_dvs, setup
from dti.geometry import scene_correction as s4
from dti.geometry.zone import Zone, raw_to_rect
from dti.io import scene6 as s6
from dti.evaluation.metrics import score_matrix
from dti.motion.cue import read_frames, motion_maps, peaks
from dti.geometry.zone_v3 import (CORR, ALL_SCENES, Z_NEAR, NEG_SHARE_MAX, DISP_MARGIN, scene_stats,
                                         sigma_from)
from dti.paths import TRAINING_ROOT

OUT = f"{TRAINING_ROOT}/dataset_m3"
PALMA_DIR = "/scratch/tmp/mdenk/data/insect_dataset_m3"
PREVIEW_DIR = "outputs/step15"
K = s4.K
MIN_PEAK = 0.10
MOTION_GAIN = 2.0
MIN_BOX = 4.0
GREEN, MAGENTA, WHITE = (0, 255, 0), (255, 0, 255), (255, 255, 255)


def blob_box(m, x, y, max_w, max_h):
    """Box around the motion blob at peak (x, y): pixels at least half as strong as the peak."""
    half = int(max(max_w, max_h, 8))
    x0, y0 = max(0, x - half), max(0, y - half)
    win = (m[y0:y + half + 1, x0:x + half + 1] >= 0.5 * m[y, x]).astype(np.uint8)
    _, lab = cv2.connectedComponents(win)
    ys, xs = np.nonzero(lab == lab[y - y0, x - x0])
    w = float(np.clip(xs.max() - xs.min() + 1, MIN_BOX, max_w))
    h = float(np.clip(ys.max() - ys.min() + 1, MIN_BOX, max_h))
    return x0 + (xs.min() + xs.max()) / 2, y0 + (ys.min() + ys.max()) / 2, w, h


def write_labels(path, boxes):
    with open(path, "w") as f:
        for cx, cy, w, h in boxes:
            f.write(f"0 {cx / RGB_W:.6f} {cy / RGB_H:.6f} {w / RGB_W:.6f} {h / RGB_H:.6f}\n")


def xyxy(boxes):
    b = np.array(boxes, float).reshape(-1, 4)
    return np.c_[b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2,
                 np.ones(len(b))]


def preview(stem, img, a, b):
    out = img.copy()
    for boxes, col in ((a, GREEN), (b, MAGENTA)):
        for cx, cy, w, h in boxes:
            pad = max(0, (14 - min(w, h)) / 2)
            cv2.rectangle(out, (int(cx - w / 2 - pad), int(cy - h / 2 - pad)),
                          (int(cx + w / 2 + pad), int(cy + h / 2 + pad)), col, 2)
    out = cv2.resize(out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    lines = [f"{stem}: training image of setup 1.  red = grey RGB frame, green = motion picture, blue = zone",
             f"green box = human RGB label, A ({len(a)});  magenta box = box from zone + motion, B ({len(b)});  "
             "tiny boxes padded"]
    cap = np.zeros((22 * len(lines) + 10, out.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(cap, t, (8, 22 * i + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.imwrite(f"{PREVIEW_DIR}/{stem}.jpg", np.vstack([out, cap]), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists -- remove it first to rebuild")
    for sub in ("A/images/all", "A/labels/all", "B/labels/all", "folds"):
        os.makedirs(f"{OUT}/{sub}")
    os.symlink("../A/images", f"{OUT}/B/images")
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    rect = setup()
    corr = json.load(open(CORR))
    optics = rect["KR"][1, 1] / np.load("data/output/stereo_calibration.npz")["KL"][1, 1]
    fy, fx = np.mgrid[0:RGB_H, 0:RGB_W]
    grid = raw_to_rect(rect, np.c_[fx.ravel(), fy.ravel()])
    data = {s: (load_dvs(s), s6.load_rgb() if s == s6.SCENE else load_rgb(s)) for s in ALL_SCENES}
    stats = {s: scene_stats(rect, *data[s], np.array(corr[s]["coef_rect_px"])) for s in ALL_SCENES}
    masks6 = s6.load_masks("rgb")
    stems = {s: [] for s in ALL_SCENES}

    print(f"B boxes against the human labels (information only)\n{'scene':8s} {'frames':>6s} {'labels A':>8s} "
          f"{'boxes B':>7s} {'DVS':>5s} | {'B on a label (circle)':>21s} {'labels with a B box':>19s} "
          f"{'B with IoU >= 0.5':>17s} | {'median size A':>13s} {'B':>9s}")
    for scene in ALL_SCENES:
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
        n_a = n_b = n_dvs = on_label = iou_ok = lab_hit = 0
        size_a, size_b, shown = [], [], []
        for f in sorted(f for f in rgb if rgb[f]):
            stem = f"{scene}_frame_{f:06d}"
            frames = read_frames(cap, f, n_video)
            m = motion_maps(frames)["median"]
            if scene == s6.SCENE:
                for mk in masks6.values():
                    if any(g in mk["frames"] for g in range(f - 10, f + 11)):
                        m[max(0, mk["top"] - 10):mk["bottom"] + 10, max(0, mk["left"] - 10):mk["right"] + 10] = 0
            recs = dvs.get(f + K, [])
            w_zone = np.zeros((RGB_H, RGB_W))
            b_boxes = []
            for r in recs:
                w = zone.weights(grid, [r], coef, True).reshape(RGB_H, RGB_W)
                w_zone = np.maximum(w_zone, w)
                pk = peaks(m, w > 0.5, 1)
                if pk and m[pk[0][1], pk[0][0]] / 255 >= MIN_PEAK:
                    box = blob_box(m, *pk[0], r["box"][2] * optics, r["box"][3] * optics)
                    if all(np.hypot(box[0] - q[0], box[1] - q[1]) > 12 for q in b_boxes):
                        b_boxes.append(box)
            a_boxes = [(b[0] + b[2] / 2, b[1] + b[3] / 2, b[2], b[3]) for b in rgb[f]]
            img = np.dstack([np.uint8(np.rint(w_zone * 255)),  # blue
                             np.uint8(np.clip(m.astype(np.float32) * MOTION_GAIN, 0, 255)),  # green
                             cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)])  # red
            cv2.imwrite(f"{OUT}/A/images/all/{stem}.png", img)
            write_labels(f"{OUT}/A/labels/all/{stem}.txt", a_boxes)
            write_labels(f"{OUT}/B/labels/all/{stem}.txt", b_boxes)
            stems[scene].append(stem)

            n_a, n_b, n_dvs = n_a + len(a_boxes), n_b + len(b_boxes), n_dvs + len(recs)
            if b_boxes:
                A, B = xyxy(a_boxes), xyxy(b_boxes)
                circ, iou = score_matrix(B, A, "circle") > 0, score_matrix(B, A, "iou") > 0
                on_label, iou_ok, lab_hit = on_label + circ.any(1).sum(), iou_ok + iou.any(1).sum(), lab_hit + circ.any(0).sum()
            size_a += [np.sqrt(b[2] * b[3]) for b in a_boxes]
            size_b += [np.sqrt(b[2] * b[3]) for b in b_boxes]
            shown.append((len(a_boxes) + len(b_boxes), stem, img, a_boxes, b_boxes))
        cap.release()
        _, stem, img, a_boxes, b_boxes = max(shown, key=lambda t: (t[0], t[1]))
        preview(stem, img, a_boxes, b_boxes)
        print(f"{scene:8s} {len(stems[scene]):6d} {n_a:8d} {n_b:7d} {n_dvs:5d} | {on_label / max(n_b, 1):21.0%} "
              f"{lab_hit / n_a:19.0%} {iou_ok / max(n_b, 1):17.0%} | {np.median(size_a):10.0f} px "
              f"{np.median(size_b) if size_b else float('nan'):6.0f} px", flush=True)

    print(f"\n{'left-out':8s} {'train images':>12s} {'test images':>11s}")
    for scene in ALL_SCENES:
        train = [s for other in ALL_SCENES if other != scene for s in stems[other]]
        for lab in "AB":
            with open(f"{OUT}/folds/{scene}_{lab}_train.txt", "w") as f:
                f.writelines(f"{PALMA_DIR}/{lab}/images/all/{s}.png\n" for s in train)
            with open(f"{OUT}/folds/{scene}_{lab}_test.txt", "w") as f:  # test always with the human labels (A)
                f.writelines(f"{PALMA_DIR}/A/images/all/{s}.png\n" for s in stems[scene])
            with open(f"{OUT}/folds/{scene}_{lab}.yaml", "w") as f:
                f.write(f"# setup 1 (grey + motion + zone), training boxes {lab}, left-out scene {scene}\n"
                        f"train: {PALMA_DIR}/folds/{scene}_{lab}_train.txt\n"
                        f"val: {PALMA_DIR}/folds/{scene}_{lab}_test.txt\n\nnc: 1\nnames: ['insect']\n")
        print(f"{scene:8s} {len(train):12d} {len(stems[scene]):11d}")
    size = sum(os.path.getsize(f"{OUT}/A/images/all/{f}") for f in os.listdir(f"{OUT}/A/images/all")) / 1e9
    print(f"\nwrote {OUT}: {sum(len(v) for v in stems.values())} images, {size:.1f} GB;  "
          f"one preview per scene in {PREVIEW_DIR}/")
    print("B on a label = B boxes whose centre lies in the circle of a human label; labels with a B box = human labels")
    print("hit by a B box; B with IoU >= 0.5 = B boxes that overlap a human label well; size = sqrt(width x height)")


if __name__ == "__main__":
    main()
