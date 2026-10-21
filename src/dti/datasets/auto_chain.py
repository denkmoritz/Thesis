"""
Step 17 of the zone rebuild: the fully automatic chain.

The human DVS annotation is replaced by our DVS detector (step 12) everywhere it
was an input. Human labels are used only for scoring (RGB labels; DVS annotations
only to pair labels with DVS insects when the correction is judged) and for the
two settings measured on the OTHER five scenes before (zone width, B2 box rule).

  1. Confidence cut-off: per scene, the cut-off with the best F2 score (recall
     counts twice as much as precision: a missed insect costs more than a false
     zone) of the detector on the OTHER five scenes, against their certain DVS
     annotations, all video frames, circle rule.
  2. Row correction: label-free as in step 13, but from detector boxes (frames
     with exactly one detection at f-1 above the cut-off).
  3. Zone: as zone v3 (step 14), from every detection above the cut-off at DVS
     frames f-1 and f (step 10: the RGB frame lies between the two).
  4. Per labelled RGB frame: the 3-channel image of setup 1 (grey, motion, zone),
     training boxes A (human) and B2 (label-free: strongest motion peak per
     detection zone if >= 0.10, box = blob at 25 % of the peak), and the
     detector without training (three strongest peaks per zone, B2 box rule).

Writes training/dataset_m3d/ (A/, B2/, folds/ as dataset_m3), predictions of the
detector without training to outputs/step17/zone_motion/labels, the
corrections to outputs/step17/auto_corrections.json.

Run: PYTHONPATH=src ./venv/bin/python src/dti/datasets/auto_chain.py
"""

import csv
import json
import os
from collections import defaultdict

import cv2
import numpy as np

from dti.io.data_sync import DATA_DIR, RGB_W, RGB_H, load_rgb, center
from dti.geometry.rectify import load_dvs, setup
from dti.geometry import scene_correction as s4
from dti.geometry.zone import Zone, raw_to_rect
from dti.io import scene6 as s6
from dti.evaluation.metrics import evaluate, load_dir
from dti.motion.cue import read_frames, motion_maps, peaks
from dti.dvs.score import DVS_W, DVS_H
from dti.geometry.correction_compare import labelfree_matches, robust_linear
from dti.geometry.zone_v3 import ALL_SCENES, Z_NEAR, NEG_SHARE_MAX, DISP_MARGIN, scene_stats, sigma_from
from dti.datasets.setup1 import MIN_PEAK, MOTION_GAIN, MIN_BOX, write_labels
from dti.paths import TRAINING_ROOT

OUT = f"{TRAINING_ROOT}/dataset_m3d"
PALMA_DIR = "/scratch/tmp/mdenk/data/insect_dataset_m3d"
OUT_DIR = "outputs/step17"
DET = "outputs/dvs/frames"
K = s4.K
BLOB_T = 0.25
N_PEAKS = 3


def detections(scene):
    d = np.load(f"{DET}/{scene}.npz")
    return d["det"], int(d["n_video"])


def detector_curve(scene, det, n_video):
    """Confidence and right/wrong (circle rule) of every detection, and the number of certain DVS boxes."""
    certain, uncertain = defaultdict(list), defaultdict(list)
    with open(f"{DATA_DIR}/{scene}_annotation.csv", newline="") as f:
        for r in csv.DictReader(f):
            x, y, w, h = (float(r[k]) for k in ("left", "top", "width", "height"))
            (certain if r["confidence"] == "certain" else uncertain)[int(r["frame_index"])].append(
                [x, y, x + w, y + h, 1.0])
    by_frame = defaultdict(list)
    for row in det:
        by_frame[int(row[0])].append(row[1:])
    gts, preds = {}, {}
    for fr in range(n_video):
        stem = f"{scene}_frame_{fr:06d}"
        gts[stem] = np.array(certain.get(fr, []), float).reshape(-1, 5)
        p = np.array(by_frame.get(fr, []), float).reshape(-1, 5)
        if fr in uncertain and len(p):
            c, u = np.c_[(p[:, 0] + p[:, 2]) / 2, (p[:, 1] + p[:, 3]) / 2], np.array(uncertain[fr])
            p = p[~((c[:, None, 0] >= u[None, :, 0]) & (c[:, None, 0] <= u[None, :, 2]) &
                    (c[:, None, 1] >= u[None, :, 1]) & (c[:, None, 1] <= u[None, :, 3])).any(1)]
        preds[stem] = p
    _, per = evaluate(preds, gts)
    conf = np.concatenate([per[s]["circle"]["pred"][:, 4] for s in per])
    hit = np.concatenate([per[s]["circle"]["tp"] for s in per]).astype(bool)
    return conf, hit, sum(len(g) for g in gts.values())


def best_f2(curves):
    conf = np.concatenate([c[0] for c in curves])
    hit = np.concatenate([c[1] for c in curves])
    n_gt = sum(c[2] for c in curves)
    order = np.argsort(-conf)
    tp = np.cumsum(hit[order])
    p, r = tp / np.arange(1, len(tp) + 1), tp / n_gt
    f2 = 5 * p * r / np.maximum(4 * p + r, 1e-9)
    i = int(f2.argmax())
    return float(conf[order][i]), float(p[i]), float(r[i])


def records(det, thr):
    """frame -> list of DVS records (box, centre, conf) above the cut-off, like load_dvs()."""
    out = defaultdict(list)
    for row in det[det[:, 5] >= thr]:
        x1, y1, x2, y2 = (float(np.clip(v, 0, m)) for v, m in zip(row[1:5], (DVS_W, DVS_H, DVS_W, DVS_H)))
        out[int(row[0])].append(dict(box=(x1, y1, x2 - x1, y2 - y1), c=((x1 + x2) / 2, (y1 + y2) / 2),
                                     conf=float(row[5]), key=False))
    return out


def blob_box(m, x, y, max_w, max_h):
    half = int(max(max_w, max_h, 8))
    x0, y0 = max(0, x - half), max(0, y - half)
    win = (m[y0:y + half + 1, x0:x + half + 1] >= BLOB_T * m[y, x]).astype(np.uint8)
    _, lab = cv2.connectedComponents(win)
    ys, xs = np.nonzero(lab == lab[y - y0, x - x0])
    w = float(np.clip(xs.max() - xs.min() + 1, MIN_BOX, max_w))
    h = float(np.clip(ys.max() - ys.min() + 1, MIN_BOX, max_h))
    return x0 + (xs.min() + xs.max()) / 2, y0 + (ys.min() + ys.max()) / 2, w, h


def dedupe(items):
    keep = []
    for it in sorted(items, key=lambda t: -t[-1]):
        if all(np.hypot(it[0] - q[0], it[1] - q[1]) > 12 for q in keep):
            keep.append(it)
    return keep


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists -- remove it first to rebuild")
    for sub in ("A/images/all", "A/labels/all", "B2/labels/all", "folds"):
        os.makedirs(f"{OUT}/{sub}")
    os.symlink("../A/images", f"{OUT}/B2/images")
    os.makedirs(f"{OUT_DIR}/zone_motion/labels", exist_ok=True)

    # 1. confidence cut-off per scene, chosen on the other five scenes
    det = {s: detections(s) for s in ALL_SCENES}
    curves = {s: detector_curve(s, *det[s]) for s in ALL_SCENES}
    thr = {}
    print("1. confidence cut-off (best F2 on the other five scenes; P / R there):")
    for s in ALL_SCENES:
        thr[s], p, r = best_f2([curves[o] for o in ALL_SCENES if o != s])
        print(f"   {s}: {thr[s]:.3f}  (P {p:.0%}, R {r:.0%})", flush=True)

    # 2. label-free correction from detector boxes
    rect = setup()
    to_rgb = rect["to_rgb"]
    optics = rect["KR"][1, 1] / np.load("data/output/stereo_calibration.npz")["KL"][1, 1]
    fy, fx = np.mgrid[0:RGB_H, 0:RGB_W]
    g = raw_to_rect(rect, np.c_[fx.ravel(), fy.ravel()])
    gx, gy = g[:, 0].reshape(RGB_H, RGB_W), g[:, 1].reshape(RGB_H, RGB_W)
    human = {s: load_dvs(s) for s in ALL_SCENES}
    rgb = {s: s6.load_rgb() if s == s6.SCENE else load_rgb(s) for s in ALL_SCENES}
    auto = {s: records(det[s][0], thr[s]) for s in ALL_SCENES}
    corr = {}
    print("\n2. label-free correction from detector boxes; leftover row offset on the last 40 % of the labels "
          "(RGB px, median |.|, < 10 px):")
    for s in ALL_SCENES:
        matches, n_picked = labelfree_matches(s, rect, gx, gy, auto[s])
        coef, inl = robust_linear(matches, to_rgb)
        disp = matches[inl, 3]
        frames = sorted(f for f in rgb[s] if rgb[s][f])
        test = frames[int(s4.FIT_SHARE * len(frames)):]
        cells = []
        for name, c in (("none", None), ("auto", coef)):
            xy, off = s4.collect(rect, human[s], rgb[s], test, c)
            a = np.abs((off - s4.predict(c, xy)) * to_rgb)
            cells.append(f"{name} {np.median(a):4.1f} px, {np.mean(a < 10):4.0%}")
        corr[s] = dict(coef_rect_px=[float(v) for v in coef], n_matches=int(inl.sum()),
                       disparity_p5_p50_p95=[float(np.percentile(disp, q)) for q in (5, 50, 95)],
                       negative_disparity_share=float(np.mean(disp < 0)), confidence_cutoff=thr[s])
        print(f"   {s}: {int(inl.sum())} matches of {n_picked} frames, negative disparity "
              f"{corr[s]['negative_disparity_share']:.0%};  " + ";  ".join(cells), flush=True)
    with open(f"{OUT_DIR}/auto_corrections.json", "w") as fh:
        json.dump(corr, fh, indent=2)

    # 3 + 4. zones from detections at f-1 and f, images, boxes A / B2, detector without training
    stats = {s: scene_stats(rect, human[s], rgb[s], np.array(corr[s]["coef_rect_px"])) for s in ALL_SCENES}
    masks6 = s6.load_masks("rgb")
    stems = {s: [] for s in ALL_SCENES}
    print(f"\n3./4. per scene: {'frames':>6s} {'labels':>6s} {'zones':>6s} {'label in zone':>13s} {'zone area':>9s} "
          f"{'boxes B2':>8s}")
    for scene in ALL_SCENES:
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
        n_lab = n_zones = n_b2 = 0
        covered, areas = [], []
        for f in sorted(f for f in rgb[scene] if rgb[scene][f]):
            stem = f"{scene}_frame_{f:06d}"
            frames = read_frames(cap, f, n_video)
            m = motion_maps(frames)["median"]
            if scene == s6.SCENE:
                for mk in masks6.values():
                    if any(q in mk["frames"] for q in range(f - 10, f + 11)):
                        m[max(0, mk["top"] - 10):mk["bottom"] + 10, max(0, mk["left"] - 10):mk["right"] + 10] = 0
            recs = auto[scene].get(f + K, []) + auto[scene].get(f, [])
            w_zone = np.zeros((RGB_H, RGB_W))
            b2, preds = [], []
            for r in recs:
                w = zone.weights(grid_flat(gx, gy), [r], coef, True).reshape(RGB_H, RGB_W)
                w_zone = np.maximum(w_zone, w)
                cap_w, cap_h = r["box"][2] * optics, r["box"][3] * optics
                for i, (x, y) in enumerate(peaks(m, w > 0.5, N_PEAKS)):
                    if m[y, x] == 0:
                        continue
                    bx = blob_box(m, x, y, cap_w, cap_h)
                    preds.append((*bx, m[y, x] / 255))
                    if i == 0 and m[y, x] / 255 >= MIN_PEAK:
                        b2.append((*bx, m[y, x] / 255))
            boxes = rgb[scene][f]
            a_boxes = [(b[0] + b[2] / 2, b[1] + b[3] / 2, b[2], b[3]) for b in boxes]
            b2 = [t[:4] for t in dedupe(b2)]
            img = np.dstack([np.uint8(np.rint(w_zone * 255)),
                             np.uint8(np.clip(m.astype(np.float32) * MOTION_GAIN, 0, 255)),
                             cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)])
            cv2.imwrite(f"{OUT}/A/images/all/{stem}.png", img)
            write_labels(f"{OUT}/A/labels/all/{stem}.txt", a_boxes)
            write_labels(f"{OUT}/B2/labels/all/{stem}.txt", b2)
            with open(f"{OUT_DIR}/zone_motion/labels/{stem}.txt", "w") as fh:
                for x, y, w, h, c in dedupe(preds):
                    fh.write(f"0 {x / RGB_W:.6f} {y / RGB_H:.6f} {w / RGB_W:.6f} {h / RGB_H:.6f} {c:.6f}\n")
            for b in boxes:
                cx, cy = (int(np.clip(v, 0, lim - 1)) for v, lim in zip(center(b), (RGB_W, RGB_H)))
                covered.append(w_zone[cy, cx] > 0.5)
            areas.append(np.mean(w_zone > 0.5))
            n_lab, n_zones, n_b2 = n_lab + len(boxes), n_zones + len(recs), n_b2 + len(b2)
            stems[scene].append(stem)
        cap.release()
        print(f"   {scene:8s} {len(stems[scene]):6d} {n_lab:6d} {n_zones:6d} {np.mean(covered):13.0%} "
              f"{np.mean(areas):9.1%} {n_b2:8d}", flush=True)

    for scene in ALL_SCENES:
        train = [s for o in ALL_SCENES if o != scene for s in stems[o]]
        for lab in ("A", "B2"):
            with open(f"{OUT}/folds/{scene}_{lab}_train.txt", "w") as f:
                f.writelines(f"{PALMA_DIR}/{lab}/images/all/{s}.png\n" for s in train)
            with open(f"{OUT}/folds/{scene}_{lab}_test.txt", "w") as f:
                f.writelines(f"{PALMA_DIR}/A/images/all/{s}.png\n" for s in stems[scene])
            with open(f"{OUT}/folds/{scene}_{lab}.yaml", "w") as f:
                f.write(f"# automatic chain (zones from our DVS detector), boxes {lab}, left-out scene {scene}\n"
                        f"train: {PALMA_DIR}/folds/{scene}_{lab}_train.txt\n"
                        f"val: {PALMA_DIR}/folds/{scene}_{lab}_test.txt\n\nnc: 1\nnames: ['insect']\n")

    gts = load_dir(f"{OUT}/A/labels/all")
    preds = load_dir(f"{OUT_DIR}/zone_motion/labels")
    print("\ndetector without training on the automatic zones (AP circle / with ignore regions / AP IoU):")
    for scene in ALL_SCENES:
        gt = {s: b for s, b in gts.items() if s.startswith(scene + "_frame_")}
        pr = {s: b for s, b in preds.items() if s in gt}
        a = evaluate(pr, gt)[0]["all"]
        b = evaluate(pr, gt, "outputs/step14/ignore")[0]["all"]
        print(f"   {scene}: {a['ap_circle']:.3f} / {b['ap_circle']:.3f} / {a['ap_iou']:.3f}")
    print(f"\nwrote {OUT} ({sum(len(v) for v in stems.values())} images) and {OUT_DIR}/")


_GRID = {}


def grid_flat(gx, gy):
    """Rectified coordinates of all RGB pixels as (n, 2), built once."""
    if "g" not in _GRID:
        _GRID["g"] = np.c_[gx.ravel(), gy.ravel()]
    return _GRID["g"]


if __name__ == "__main__":
    main()
