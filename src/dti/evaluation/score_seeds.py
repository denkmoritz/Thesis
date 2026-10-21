"""
Step 17c of the zone rebuild: how much do the automatic-chain results move by chance?

Three runs per left-out scene and kind of boxes: the original (m3d.slurm, seed
offset 0) and two repeats with other random seeds (m3d_seeds.slurm, offsets 1, 2),
predictions in outputs/m3d/m3d_<scene>_<A|B2>[_s1|_s2]/labels.

Printed per scene, for A and B2: AP (circle and IoU rule, ignore regions used) of
each seed, and mean +- standard deviation over the three.

Run: PYTHONPATH=src ./venv/bin/python src/dti/evaluation/score_seeds.py
"""

import os

import numpy as np

from dti.evaluation.metrics import load_dir, evaluate
from dti.geometry.zone_v3 import ALL_SCENES
from dti.evaluation.score_auto_chain import GT, IGNORE

PRED = "outputs/m3d"
SEEDS = ["", "_s1", "_s2"]


def main():
    gts = load_dir(GT)
    for rule in ("circle", "iou"):
        print(f"\nAP, {rule} rule: seed 0 / 1 / 2 -> mean +- standard deviation")
        print(f"{'scene':8s} {'A':>36s} {'B2':>36s}")
        for s in ALL_SCENES:
            gt = {k: v for k, v in gts.items() if k.startswith(s + "_frame_")}
            cells = []
            for lab in ("A", "B2"):
                vals = []
                for sd in SEEDS:
                    path = f"{PRED}/m3d_{s}_{lab}{sd}/labels"
                    if os.path.isdir(path):
                        pr = {k: v for k, v in load_dir(path).items() if k in gt}
                        vals.append(evaluate(pr, gt, IGNORE)[0]["all"]["ap_" + rule])
                each = " / ".join(f"{v:.2f}" for v in vals)
                cells.append(f"{each} -> {np.mean(vals):.2f} +- {np.std(vals, ddof=1) if len(vals) > 1 else 0:.2f}")
            print(f"{s:8s} {cells[0]:>36s} {cells[1]:>36s}")


if __name__ == "__main__":
    main()
