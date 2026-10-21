"""
Step 11f of the zone rebuild: false alarms on frames without an insect.

All test frames so far contain an insect. Here 20 frames per scene are added in
which no insect should be present: frames of the test part (last 40 %) with no
DVS annotation of any kind (certain or uncertain) within 50 frames before and
after. Their label file is empty, so every prediction on them is a false alarm.

Detectors (step 11e, background-model motion map and previous-frame map):
  zone+motion   has no DVS insect on these frames -> no zone -> no prediction.
                This holds by construction with human DVS annotations; a real DVS
                detector will give false zones (not tested here).
  motion only   the ten strongest motion peaks of the frame, as before.

Printed per scene:
  - AP and best F1 (circle rule) without and with the empty frames
  - at the confidence threshold that gives the best F1 on the insect frames:
    share of empty frames with at least one false alarm, false alarms per frame
  - strength of the strongest motion peak: empty frames vs insect frames

The chosen frames are written to outputs/step11f/frames/ for a visual
check (they are assumed empty, not verified).

Run: PYTHONPATH=src ./venv/bin/python src/dti/motion/empty_frames.py
"""

import csv
import json
import os

import cv2
import numpy as np

from dti.io.data_sync import DATA_DIR, SCENES, load_rgb, RGB_W, RGB_H
from dti.geometry.rectify import load_dvs
from dti.geometry.scene_correction import K
from dti.geometry.epiline import CORR
from dti.evaluation.metrics import load_dir, evaluate, render
from dti.motion.cue import DATASET, read_frames, motion_maps, peaks
from dti.motion.detector import refine, N_FRAME_PEAKS

OUT_DIR = "outputs/step11f"
PRED_11E = "outputs/step11e"
N_EMPTY = 20
GAP = 50  # frames without any DVS annotation before and after
MAPS = ["median", "prev"]


def empty_candidates(scene, n_video, lo, hi):
    """Frames in lo..hi with no DVS annotation (any confidence) within GAP frames."""
    has = np.zeros(n_video, bool)
    with open(f"{DATA_DIR}/{scene}_annotation.csv", newline="") as f:
        for r in csv.DictReader(f):
            if int(r["frame_index"]) < n_video:
                has[int(r["frame_index"])] = True
    near = np.convolve(has, np.ones(2 * GAP + 1), mode="same") > 0
    ok = np.flatnonzero(~near)
    return ok[(ok >= max(lo, 10)) & (ok <= min(hi, n_video - 11))]


def main():
    for sub in ("frames", "labels", "motion_only_median/labels", "motion_only_prev/labels", "zone_motion/labels"):
        os.makedirs(f"{OUT_DIR}/{sub}", exist_ok=True)
    corr = json.load(open(CORR))
    everything = np.ones((RGB_H, RGB_W), bool)
    for scene in SCENES:
        dvs, rgb = load_dvs(scene), load_rgb(scene)
        c = corr[scene]
        fit = [b for f, boxes in rgb.items() if f <= c["fit_frames"][1] for b in boxes]
        size = (float(np.median([b[2] for b in fit])), float(np.median([b[3] for b in fit])))
        cap = cv2.VideoCapture(f"{DATA_DIR}/{scene}_rgb.mp4")
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cand = empty_candidates(scene, n_video, *c["test_frames"])
        picks = [int(cand[i]) for i in np.linspace(0, len(cand) - 1, N_EMPTY).round().astype(int)]
        for f in picks:
            assert not dvs.get(f + K, []), (scene, f)  # no DVS insect -> zone+motion predicts nothing
            stem = f"{scene}_frame_{f:06d}"
            frames = read_frames(cap, f, n_video)
            cv2.imwrite(f"{OUT_DIR}/frames/{stem}.jpg", frames[0], [cv2.IMWRITE_JPEG_QUALITY, 95])
            open(f"{OUT_DIR}/labels/{stem}.txt", "w").close()
            open(f"{OUT_DIR}/zone_motion/labels/{stem}.txt", "w").close()
            maps = motion_maps(frames)
            for name in MAPS:
                m = maps[name]
                with open(f"{OUT_DIR}/motion_only_{name}/labels/{stem}.txt", "w") as fh:
                    for x, y in peaks(m, everything, N_FRAME_PEAKS):
                        cx, cy = refine(m, x, y, max(*size, 8))
                        fh.write(f"0 {cx / RGB_W:.6f} {cy / RGB_H:.6f} {size[0] / RGB_W:.6f} {size[1] / RGB_H:.6f} "
                                 f"{m[y, x] / 255:.6f}\n")
        cap.release()
        print(f"{scene}: {len(cand)} candidate frames in the test part, {len(picks)} chosen "
              f"({picks[0]}..{picks[-1]})", flush=True)

    gts, empty = load_dir(f"{DATASET}/labels/val"), load_dir(f"{OUT_DIR}/labels")
    detectors = [("zone+motion median", f"{PRED_11E}/zone_motion_median/labels", f"{OUT_DIR}/zone_motion/labels"),
                 ("motion only median", f"{PRED_11E}/motion_only_median/labels", f"{OUT_DIR}/motion_only_median/labels"),
                 ("motion only prev", f"{PRED_11E}/motion_only_prev/labels", f"{OUT_DIR}/motion_only_prev/labels")]
    print(f"\n{'scene':8s} {'detector':19s} | {'AP circle':>9s} {'+ empty':>8s} | {'best F1':>7s} {'+ empty':>8s} | "
          f"{'at conf':>7s} {'empty frames with alarm':>23s} {'alarms/frame':>12s} | "
          f"{'top peak: insect':>16s} {'empty':>6s}")
    for scene in SCENES:
        gt = {s: b for s, b in gts.items() if s.startswith(scene + "_frame_")}
        em = {s: b for s, b in empty.items() if s.startswith(scene + "_frame_")}
        for name, pos_dir, emp_dir in detectors:
            pos = {s: b for s, b in load_dir(pos_dir).items() if s in gt}
            neg = {s: b for s, b in load_dir(emp_dir).items() if s in em}
            a = evaluate(pos, gt)[0]["all"]
            res, per_image = evaluate({**pos, **neg}, {**gt, **em})
            b = res["all"]
            thr = a["thr_best"]
            alarms = [int(np.sum(neg.get(s, np.zeros((0, 5)))[:, 4] >= thr)) for s in em]
            top = lambda d, keys: np.median([d[s][:, 4].max() if s in d and len(d[s]) else 0.0 for s in keys])
            print(f"{scene:8s} {name:19s} | {a['ap_circle']:9.3f} {b['ap_circle']:8.3f} | {a['f1_best']:7.2f} "
                  f"{b['f1_best']:8.2f} | {thr:7.3f} {np.mean(np.array(alarms) > 0):23.0%} {np.mean(alarms):12.1f} | "
                  f"{top(pos, gt):16.3f} {top(neg, em):6.3f}")
            if name == "motion only median":
                worst = max(em, key=lambda s: (neg[s][:, 4].max() if len(neg.get(s, [])) else 0, s))
                render(worst, per_image[worst], f"{OUT_DIR}/frames", OUT_DIR, thr)
    print(f"\n+ empty = the same with the {N_EMPTY} empty frames per scene added;  at conf = threshold of the best F1 on "
          "the insect frames;")
    print("top peak = median confidence of the strongest prediction per frame (insect frames vs empty frames)")
    print(f"empty frames for a visual check: {OUT_DIR}/frames/;  example per scene (empty frame with the strongest "
          f"false alarm of 'motion only median'): {OUT_DIR}/")


if __name__ == "__main__":
    main()
