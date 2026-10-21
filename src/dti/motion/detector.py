"""
Step 11e of the zone rebuild: two detectors without any training.

  zone+motion   For every DVS insect (annotation of frame f-1, certain): its
                zone in the RGB frame (step 7, weight > 0.5); the three strongest
                motion peaks inside it are the predicted positions. Box size =
                DVS box size x the RGB/DVS size ratio measured on the first 60 %
                of the scenes. Confidence = strength of the peak.
  motion only   No DVS, no zone: the ten strongest motion peaks of the whole
                RGB frame. Box size = median label size in the first 60 % of
                the scene. Confidence = strength of the peak.

Motion maps as in step 11d:
  median  |I(f) - median of 11 frames from f-10 to f+10|  (uses later frames)
  prev    |I(f) - I(f-1)|                                 (no delay)

Test frames: the last 40 % of scenes 1-5 (same as the pilot). The predictions
are written as YOLO text files and scored with step11_eval.py, next to the
YOLOv7 runs on the same frames:
  net cross   pilot P0, zone multiply, scene left out of training
  net within  v2, zone multiply, first 60 % of the scene in training

What the detectors use from the test scene: the row correction fitted on its
first 60 % (as the networks' zones do); "motion only" its median label size.

Run: PYTHONPATH=src ./venv/bin/python src/dti/motion/detector.py
"""

import glob
import json
import os

import cv2
import numpy as np

from dti.io.data_sync import DATA_DIR, SCENES, load_rgb
from dti.geometry.rectify import load_dvs, setup
from dti.geometry.scene_correction import K
from dti.geometry.epiline import CORR, RGB_W, RGB_H
from dti.geometry.zone import Zone, NO_DEPTH_LIMIT, estimate_sigma, estimate_h_scale, raw_to_rect
from dti.evaluation.metrics import load_dir, evaluate, render
from dti.motion.cue import DATASET, read_frames, motion_maps, peaks, PEAK_SEP

OUT_DIR = "outputs/step11e"
MAPS = ["median", "prev"]
N_ZONE_PEAKS, N_FRAME_PEAKS = 3, 10
NETS = {"net cross": "outputs/pilot_p0/p0_{scene}_multiply/labels",
        "net within": "outputs/v2_within_scene/v2_multiply/labels"}


def refine(m, x, y, half):
    """Centre of the motion blob around a peak (weighted mean in a small window)."""
    x0, x1 = max(0, int(x - half)), min(RGB_W, int(x + half) + 1)
    y0, y1 = max(0, int(y - half)), min(RGB_H, int(y + half) + 1)
    w = m[y0:y1, x0:x1].astype(np.float64) ** 2
    if w.sum() == 0:
        return float(x), float(y)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((w * xx).sum() / w.sum()), float((w * yy).sum() / w.sum())


def dedupe(preds):
    """Keep the strongest of predictions closer than half the peak distance (overlapping zones)."""
    keep = []
    for p in sorted(preds, key=lambda p: -p[4]):
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > PEAK_SEP / 2 for q in keep):
            keep.append(p)
    return keep


def write(name, stem, preds):
    os.makedirs(f"{OUT_DIR}/{name}/labels", exist_ok=True)
    with open(f"{OUT_DIR}/{name}/labels/{stem}.txt", "w") as f:
        for cx, cy, w, h, conf in preds:
            f.write(f"0 {cx / RGB_W:.6f} {cy / RGB_H:.6f} {w / RGB_W:.6f} {h / RGB_H:.6f} {conf:.6f}\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rect = setup()
    corr = json.load(open(CORR))
    rows, vals = estimate_sigma(rect, corr)
    optics, ratio = estimate_h_scale(rect, corr)
    zone = Zone(rect, rows, vals, optics)
    fy, fx = np.mgrid[0:RGB_H, 0:RGB_W]
    grid = raw_to_rect(rect, np.c_[fx.ravel(), fy.ravel()])
    everything = np.ones((RGB_H, RGB_W), bool)
    print(f"box size for zone+motion: DVS box x {ratio:.2f} (measured on the first 60 %; optics alone would give "
          f"{optics:.2f})\n")

    for scene in SCENES:
        dvs, rgb = load_dvs(scene), load_rgb(scene)
        c = corr[scene]
        coef = np.array(c["coef_rect_px"])
        fit = [b for f, boxes in rgb.items() if f <= c["fit_frames"][1] for b in boxes]
        size = (float(np.median([b[2] for b in fit])), float(np.median([b[3] for b in fit])))
        stems = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{DATASET}/images/val/{scene}_frame_*.jpg"))
        cap = cv2.VideoCapture(f"{DATA_DIR}/{scene}_rgb.mp4")
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for stem in stems:
            f = int(stem[-6:])
            maps = motion_maps(read_frames(cap, f, n_video))
            recs = dvs.get(f + K, [])
            pieces = [zone.weights(grid, [r], coef, scene not in NO_DEPTH_LIMIT).reshape(RGB_H, RGB_W) > 0.5
                      for r in recs]
            for name in MAPS:
                m = maps.get(name)
                zp, fp = [], []
                if m is not None:
                    for r, piece in zip(recs, pieces):
                        w, h = r["box"][2] * ratio, r["box"][3] * ratio
                        for x, y in peaks(m, piece, N_ZONE_PEAKS):
                            zp.append((*refine(m, x, y, max(w, h, 8)), w, h, m[y, x] / 255))
                    for x, y in peaks(m, everything, N_FRAME_PEAKS):
                        fp.append((*refine(m, x, y, max(*size, 8)), *size, m[y, x] / 255))
                write(f"zone_motion_{name}", stem, dedupe(zp))
                write(f"motion_only_{name}", stem, fp)
        cap.release()
        print(f"{scene}: {len(stems)} frames done (motion-only box size {size[0]:.0f} x {size[1]:.0f} px)", flush=True)

    gts = load_dir(f"{DATASET}/labels/val")
    cols = [("zone+motion median", f"{OUT_DIR}/zone_motion_median/labels"),
            ("zone+motion prev", f"{OUT_DIR}/zone_motion_prev/labels"),
            ("motion only median", f"{OUT_DIR}/motion_only_median/labels"),
            ("motion only prev", f"{OUT_DIR}/motion_only_prev/labels")] + list(NETS.items())
    for title, key in (("AP, IoU rule (IoU >= 0.5)", "ap_iou"), ("AP, circle rule", "ap_circle"),
                       ("best F1, circle rule (threshold chosen on the test scene, optimistic)", "f1_best")):
        print(f"\n{title}")
        print(f"{'scene':8s} {'labels':>6s} | " + " ".join(f"{n:>18s}" for n, _ in cols))
        for scene in SCENES:
            gt = {s: b for s, b in gts.items() if s.startswith(scene + "_frame_")}
            cells = []
            for n, path in cols:
                path = path.format(scene=scene)
                if not os.path.isdir(path):
                    cells.append(f"{'n/a':>18s}")
                    continue
                pred = {s: b for s, b in load_dir(path).items() if s in gt}
                res, per_image = evaluate(pred, gt)
                cells.append(f"{res['all'][key]:18.3f}")
                if n == "zone+motion median" and key == "f1_best":
                    busiest = max(per_image, key=lambda s: (len(per_image[s]["gt"]), s))
                    render(busiest, per_image[busiest], f"{DATASET}/images/val", OUT_DIR, res["all"]["thr_best"])
            print(f"{scene:8s} {sum(len(b) for b in gt.values()):6d} | " + " ".join(cells))
    print(f"\nexample images (zone+motion median, busiest test frame per scene): {OUT_DIR}/")


if __name__ == "__main__":
    main()
