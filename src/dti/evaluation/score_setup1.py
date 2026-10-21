"""
Step 15b of the zone rebuild: score the setup-1 runs (grey + motion + zone).

Reads the predictions that m3.slurm saved on PALMA (m3_<scene>_<A|B>/labels),
unpacked into outputs/m3/, and scores each run on its left-out scene
with step11_eval.py against the human RGB labels (training/dataset_m3/A/labels),
without and with the ignore regions of step 14.

Columns per left-out scene:
  A        YOLOv7 trained on human RGB labels of the other five scenes
  B        YOLOv7 trained on zone + motion boxes of the other five scenes
  detector the zone + motion detector without training (step 14), same frames

Run: PYTHONPATH=src ./venv/bin/python src/dti/evaluation/score_setup1.py
"""

import os

from dti.evaluation.metrics import load_dir, evaluate, render
from dti.geometry.zone_v3 import ALL_SCENES
from dti.paths import TRAINING_ROOT

DATASET = f"{TRAINING_ROOT}/dataset_m3"
PRED_DIR = "outputs/m3"
IGNORE = "outputs/step14/ignore"
COLS = [("A", PRED_DIR + "/m3_{scene}_A/labels"), ("B", PRED_DIR + "/m3_{scene}_B/labels"),
        ("B2", PRED_DIR + "/m3_{scene}_B2/labels"),  # step 16: label-free, larger boxes
        ("detector", "outputs/step14/zone_motion/labels"),
        # channel test (step 15c): run A with one input channel removed
        ("nozone", PRED_DIR + "/m3_{scene}_nozone/labels"), ("nomotion", PRED_DIR + "/m3_{scene}_nomotion/labels"),
        ("nogrey", PRED_DIR + "/m3_{scene}_nogrey/labels")]


def main():
    gts = load_dir(f"{DATASET}/A/labels/all")
    tables = [("AP, circle rule", "ap_circle", None), ("AP, circle rule, with ignore regions", "ap_circle", IGNORE),
              ("AP, IoU rule (IoU >= 0.5)", "ap_iou", None), ("AP, IoU rule, with ignore regions", "ap_iou", IGNORE),
              ("best F1, circle rule, with ignore regions (threshold chosen on the test scene, optimistic)",
               "f1_best", IGNORE)]
    for title, key, ignore in tables:
        print(f"\n{title}\n{'left-out':8s} {'frames':>6s} {'labels':>6s} | " + " ".join(f"{n:>10s}" for n, _ in COLS))
        for scene in ALL_SCENES:
            gt = {s: b for s, b in gts.items() if s.startswith(scene + "_frame_")}
            cells = []
            for name, path in COLS:
                path = path.format(scene=scene)
                if not os.path.isdir(path):
                    cells.append(f"{'n/a':>10s}")
                    continue
                res, per_image = evaluate({s: b for s, b in load_dir(path).items() if s in gt}, gt, ignore)
                cells.append(f"{res['all'][key]:10.3f}")
                if key == "f1_best" and name in "AB":
                    out = f"{PRED_DIR}/examples_{name}"
                    os.makedirs(f"{out}/_img", exist_ok=True)
                    busiest = max(per_image, key=lambda s: (len(per_image[s]["gt"]), s))
                    if not os.path.exists(f"{out}/_img/{busiest}.jpg"):
                        os.symlink(f"{DATASET}/A/images/all/{busiest}.png", f"{out}/_img/{busiest}.jpg")
                    render(busiest, per_image[busiest], f"{out}/_img", out, res["all"]["thr_best"])
            print(f"{scene:8s} {len(gt):6d} {sum(len(b) for b in gt.values()):6d} | " + " ".join(cells))
    print(f"\nexample images (busiest test frame per scene, on the 3-channel input image): {PRED_DIR}/examples_<A|B>/")


if __name__ == "__main__":
    main()
