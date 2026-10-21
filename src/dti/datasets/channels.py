"""
Step 15c of the zone rebuild: channel test for setup 1.

Which of the three inputs does the network need? Three copies of the setup-1
images (step 15) are written, each with one channel set to zero:

  nozone    grey frame + motion     (no DVS information at all)
  nomotion  grey frame + zone       (single frame + DVS zone, like the pilot)
  nogrey    motion + zone           (no picture of the scene itself)

Training boxes are the human RGB labels (A) in all three, so they compare
directly with run A of step 15 (all three channels). Same folds: each of the
six scenes left out in turn, tested on the human labels of that scene.

Added to training/dataset_m3/ (paths inside the lists are PALMA paths):
  {variant}/images/all/{stem}.png     {variant}/labels -> ../A/labels
  folds/{scene}_{variant}_train.txt, _test.txt, .yaml

Run: PYTHONPATH=src ./venv/bin/python src/dti/datasets/channels.py
"""

import os

import cv2
import numpy as np

from dti.geometry.zone_v3 import ALL_SCENES
from dti.datasets.setup1 import OUT, PALMA_DIR

PREVIEW_DIR = "outputs/step15c"
VARIANTS = {"nozone": 0, "nomotion": 1, "nogrey": 2}  # channel set to zero (stored order: blue, green, red)
TEXT = {"nozone": "grey frame (red) + motion (green), zone removed",
        "nomotion": "grey frame (red) + zone (blue), motion removed",
        "nogrey": "motion (green) + zone (blue), grey frame removed"}


def main():
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    stems = sorted(f[:-4] for f in os.listdir(f"{OUT}/A/images/all"))
    for v in VARIANTS:
        os.makedirs(f"{OUT}/{v}/images/all", exist_ok=True)
        if not os.path.islink(f"{OUT}/{v}/labels"):
            os.symlink("../A/labels", f"{OUT}/{v}/labels")
    shown = {s: next(st for st in stems if st.startswith(s)) for s in ("3_m-h-h",)}
    for i, stem in enumerate(stems):
        img = cv2.imread(f"{OUT}/A/images/all/{stem}.png")
        for v, ch in VARIANTS.items():
            out = img.copy()
            out[..., ch] = 0
            cv2.imwrite(f"{OUT}/{v}/images/all/{stem}.png", out)
            if stem in shown.values():
                small = cv2.resize(out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cap = np.zeros((32, small.shape[1], 3), np.uint8)
                cv2.putText(cap, f"{stem}, variant '{v}': {TEXT[v]}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(f"{PREVIEW_DIR}/{stem}_{v}.jpg", np.vstack([small, cap]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if i % 200 == 0:
            print(f"{i}/{len(stems)} images", flush=True)

    by_scene = {s: [st for st in stems if st.startswith(s + "_frame_")] for s in ALL_SCENES}
    for scene in ALL_SCENES:
        train = [st for other in ALL_SCENES if other != scene for st in by_scene[other]]
        for v in VARIANTS:
            for name, items in (("train", train), ("test", by_scene[scene])):
                with open(f"{OUT}/folds/{scene}_{v}_{name}.txt", "w") as f:
                    f.writelines(f"{PALMA_DIR}/{v}/images/all/{st}.png\n" for st in items)
            with open(f"{OUT}/folds/{scene}_{v}.yaml", "w") as f:
                f.write(f"# setup 1 channel test: {TEXT[v]}; training boxes A; left-out scene {scene}\n"
                        f"train: {PALMA_DIR}/folds/{scene}_{v}_train.txt\n"
                        f"val: {PALMA_DIR}/folds/{scene}_{v}_test.txt\n\nnc: 1\nnames: ['insect']\n")
    for v in VARIANTS:
        d = f"{OUT}/{v}/images/all"
        size = sum(os.path.getsize(f"{d}/{f}") for f in os.listdir(d)) / 1e9
        print(f"{v:9s} {len(os.listdir(d))} images, {size:.1f} GB, labels -> {os.readlink(f'{OUT}/{v}/labels')} "
              f"({len(os.listdir(f'{OUT}/{v}/labels/all'))} files)")
    print(f"folds: {len([f for f in os.listdir(f'{OUT}/folds') if f.endswith('.yaml')])} yaml files;  "
          f"previews of one frame per variant: {PREVIEW_DIR}/")


if __name__ == "__main__":
    main()
