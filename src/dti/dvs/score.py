"""
Step 12b of the zone rebuild: score our DVS detector like Gebauer et al.

Input: detections of dvs.slurm on EVERY frame of each left-out scene's DVS
video (outputs/dvs/detect/dvs_<scene>/labels/<scene>_dvs_<n>.txt,
n = video frame + 1, checked: the boxes of frame f are in file n = f + 1).

Ground truth: the certain DVS annotations of every frame (hand-placed and
filled-in rows), as the paper ("confident user annotations only"). Detections
whose centre lies inside an uncertain box are removed (neither right nor wrong).

Printed per scene:
  F1 best      box F1 with the paper's rule (centre inside the smallest circle
               around the box), threshold chosen on the scene -- the same kind of
               number as the paper's table, next to the paper's YOLOv7 value
  P, R         precision and recall at that threshold
  AP circle    threshold-free
  keyframes    the same F1 on hand-placed frames only
Saves the detections as outputs/dvs/frames/<scene>.npz
(frame, x1, y1, x2, y2, conf in DVS px) for the later steps.

Run: PYTHONPATH=src ./venv/bin/python src/dti/dvs/score.py
"""

import csv
import glob
import os
from collections import defaultdict

import cv2
import numpy as np

from dti.io.data_sync import DATA_DIR
from dti.evaluation.metrics import evaluate, centres
from dti.geometry.zone_v3 import ALL_SCENES

DET = "outputs/dvs/detect"
OUT = "outputs/dvs/frames"
DVS_W, DVS_H = 1280, 720
PAPER_F1 = {"1_l-l-l": 64.6, "2_l-h-l": 58.5, "3_m-h-h": 46.0, "4_m-m-h": 37.1, "5_h-l-h": 46.1, "6_h-h-h": 24.3}


def load_detections(scene):
    """video frame -> array (n, 5): x1, y1, x2, y2, conf in DVS px."""
    out = {}
    for p in glob.glob(f"{DET}/dvs_{scene}/labels/{scene}_dvs_*.txt"):
        a = np.loadtxt(p, ndmin=2)
        cx, cy, w, h = a[:, 1] * DVS_W, a[:, 2] * DVS_H, a[:, 3] * DVS_W, a[:, 4] * DVS_H
        out[int(p.rsplit("_", 1)[1][:-4]) - 1] = np.c_[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, a[:, 5]]
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"{'scene':8s} {'frames':>6s} {'certain boxes':>13s} | {'F1 best':>7s} {'at conf':>7s} {'P':>5s} {'R':>5s} "
          f"{'paper F1':>8s} | {'AP circle':>9s} | {'keyframes F1':>12s}")
    for scene in ALL_SCENES:
        n_video = int(cv2.VideoCapture(f"{DATA_DIR}/{scene}_dvs.mp4").get(cv2.CAP_PROP_FRAME_COUNT))
        certain, uncertain, key = defaultdict(list), defaultdict(list), set()
        with open(f"{DATA_DIR}/{scene}_annotation.csv", newline="") as f:
            for r in csv.DictReader(f):
                x, y, w, h = (float(r[k]) for k in ("left", "top", "width", "height"))
                fr = int(r["frame_index"])
                (certain if r["confidence"] == "certain" else uncertain)[fr].append([x, y, x + w, y + h, 1.0])
                if r["is_keyframe"] == "True":
                    key.add(fr)
        dets = load_detections(scene)
        gts, preds = {}, {}
        for fr in range(n_video):
            stem = f"{scene}_frame_{fr:06d}"
            gts[stem] = np.array(certain.get(fr, []), float).reshape(-1, 5)
            p = dets.get(fr, np.zeros((0, 5)))
            if fr in uncertain and len(p):
                c, u = centres(p), np.array(uncertain[fr])
                inside = ((c[:, None, 0] >= u[None, :, 0]) & (c[:, None, 0] <= u[None, :, 2]) &
                          (c[:, None, 1] >= u[None, :, 1]) & (c[:, None, 1] <= u[None, :, 3])).any(1)
                p = p[~inside]
            preds[stem] = p
        res = evaluate(preds, gts)[0]["all"]
        thr = res["thr_best"]
        tp = fp = 0
        _, per = evaluate(preds, gts)
        conf = np.concatenate([per[s]["circle"]["pred"][:, 4] for s in per])
        hit = np.concatenate([per[s]["circle"]["tp"] for s in per]).astype(bool)
        tp, n_pred, n_gt = int(np.sum(hit & (conf >= thr))), int(np.sum(conf >= thr)), res["labels"]
        kstems = {f"{scene}_frame_{fr:06d}" for fr in key if fr < n_video}
        kres = evaluate({s: preds[s] for s in kstems}, {s: gts[s] for s in kstems})[0]["all"]
        print(f"{scene:8s} {n_video:6d} {n_gt:13d} | {100 * res['f1_best']:7.1f} {thr:7.3f} {100 * tp / max(n_pred, 1):5.1f} "
              f"{100 * tp / n_gt:5.1f} {PAPER_F1[scene]:8.1f} | {res['ap_circle']:9.3f} | {100 * kres['f1_best']:12.1f}",
              flush=True)
        rows = [np.r_[fr, b] for fr, a in dets.items() for b in a]
        np.savez_compressed(f"{OUT}/{scene}.npz", det=np.array(rows).reshape(-1, 6), n_video=n_video)
    print("\nF1 in %, circle rule, all video frames, certain annotations; paper F1 = Gebauer et al., best YOLOv7 per scene")
    print(f"detections per frame saved in {OUT}/<scene>.npz")


if __name__ == "__main__":
    main()
