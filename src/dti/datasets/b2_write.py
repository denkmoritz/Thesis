"""
Step 16b of the zone rebuild: write the training boxes B2 (label-free, larger).

Same as B of step 15 (strongest motion peak in each DVS insect's zone, kept if
stronger than MIN_PEAK, peaks closer than 12 px merged), but the box is the
motion blob outlined at 25 % of the peak instead of 50 %. Step 16 chose this rule
leave-one-scene-out, and it was the best rule in every fold, so one rule is used
for all scenes.

Uses the cached peaks and motion crops of step 16 (no motion maps recomputed).
Writes training/dataset_m3/B2/labels/all/{stem}.txt for every image of A,
B2/images -> ../A/images, and folds/{scene}_B2_train.txt, _test.txt, .yaml.

Run: PYTHONPATH=src ./venv/bin/python src/dti/datasets/b2_write.py
"""

import os

import numpy as np

from dti.io.data_sync import RGB_W, RGB_H
from dti.geometry.zone_v3 import ALL_SCENES
from dti.datasets.setup1 import OUT, PALMA_DIR, write_labels
from dti.datasets.b2_box_rule import CACHE, RULES, box

RULE = next(r for r in RULES if r[0] == "blob 0.25")


def main():
    c = np.load(CACHE)
    boxes = {}
    for row, crop in zip(c["rows"], c["crops"]):
        stem = f"{ALL_SCENES[int(row[0])]}_frame_{int(row[1]):06d}"
        b = box(RULE, row, crop)
        kept = boxes.setdefault(stem, [])
        if all(np.hypot(b[0] - q[0], b[1] - q[1]) > 12 for q in kept):
            kept.append(b)
    os.makedirs(f"{OUT}/B2/labels/all", exist_ok=True)
    if not os.path.islink(f"{OUT}/B2/images"):
        os.symlink("../A/images", f"{OUT}/B2/images")
    stems = sorted(f[:-4] for f in os.listdir(f"{OUT}/A/images/all"))
    for stem in stems:
        clipped = [(min(max(x, 0), RGB_W - 1), min(max(y, 0), RGB_H - 1), w, h) for x, y, w, h in boxes.get(stem, [])]
        write_labels(f"{OUT}/B2/labels/all/{stem}.txt", clipped)
    for scene in ALL_SCENES:
        train = [s for s in stems if not s.startswith(scene + "_frame_")]
        test = [s for s in stems if s.startswith(scene + "_frame_")]
        with open(f"{OUT}/folds/{scene}_B2_train.txt", "w") as f:
            f.writelines(f"{PALMA_DIR}/B2/images/all/{s}.png\n" for s in train)
        with open(f"{OUT}/folds/{scene}_B2_test.txt", "w") as f:  # test always with the human labels (A)
            f.writelines(f"{PALMA_DIR}/A/images/all/{s}.png\n" for s in test)
        with open(f"{OUT}/folds/{scene}_B2.yaml", "w") as f:
            f.write(f"# setup 1, training boxes B2 (label-free, blob at 25 % of the peak), left-out scene {scene}\n"
                    f"train: {PALMA_DIR}/folds/{scene}_B2_train.txt\nval: {PALMA_DIR}/folds/{scene}_B2_test.txt\n\n"
                    f"nc: 1\nnames: ['insect']\n")
        n = sum(len(boxes.get(s, [])) for s in test)
        print(f"{scene}: {len(test)} images, {n} B2 boxes")
    print(f"wrote {OUT}/B2 ({len(stems)} label files) and 6 fold yaml files")


if __name__ == "__main__":
    main()
