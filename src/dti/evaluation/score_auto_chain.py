"""
Step 17b of the zone rebuild: score the automatic chain next to the runs with
human DVS zones, scene by scene.

Columns (all on all labelled frames of the left-out scene, human RGB labels):
  A human / B2 human   setup 1 with zones from the human DVS annotation (8.6, 8.8)
  A auto / B2 auto     the same with zones from our DVS detector (m3d.slurm)
  free auto            zone + motion detector without training, detector zones (step 17)

Printed: found / right in % at the best cut-off (circle rule and IoU rule), and AP.

Run: PYTHONPATH=src ./venv/bin/python src/dti/evaluation/score_auto_chain.py
"""

import os

import numpy as np

from dti.evaluation.metrics import load_dir, evaluate
from dti.geometry.zone_v3 import ALL_SCENES
from dti.paths import TRAINING_ROOT

GT = f"{TRAINING_ROOT}/dataset_m3d/A/labels/all"
IGNORE = "outputs/step14/ignore"
COLS = [("A human", "outputs/m3/m3_{s}_A/labels"), ("B2 human", "outputs/m3/m3_{s}_B2/labels"),
        ("A auto", "outputs/m3d/m3d_{s}_A/labels"), ("B2 auto", "outputs/m3d/m3d_{s}_B2/labels"),
        ("free auto", "outputs/step17/zone_motion/labels")]


def found_right(per, rule, n_gt):
    conf = np.concatenate([per[k][rule]["pred"][:, 4] for k in per])
    tp = np.concatenate([per[k][rule]["tp"] for k in per]).astype(bool)
    if not len(conf):
        return 0.0, 0.0
    o = np.argsort(-conf)
    t = np.cumsum(tp[o])
    n = np.arange(1, len(t) + 1)
    i = int((2 * t / (n + n_gt)).argmax())
    return t[i] / n_gt, t[i] / n[i]


def main():
    gts = load_dir(GT)
    for rule, title in (("circle", "position only (circle rule)"), ("iou", "position and size (IoU >= 0.5)")):
        print(f"\n{title}: found / right at the best cut-off, AP in brackets; ignore regions used")
        print(f"{'scene':8s} " + " ".join(f"{n:>18s}" for n, _ in COLS))
        for s in ALL_SCENES:
            gt = {k: b for k, b in gts.items() if k.startswith(s + "_frame_")}
            n_gt = sum(len(b) for b in gt.values())
            cells = []
            for _, path in COLS:
                path = path.format(s=s)
                if not os.path.isdir(path):
                    cells.append(f"{'n/a':>18s}")
                    continue
                res, per = evaluate({k: b for k, b in load_dir(path).items() if k in gt}, gt, IGNORE)
                fd, rt = found_right(per, rule, n_gt)
                cells.append(f"{fd:4.0%} / {rt:4.0%} ({res['all']['ap_' + rule]:.2f})")
            print(f"{s:8s} " + " ".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
