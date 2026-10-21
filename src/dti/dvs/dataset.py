"""
Step 12 of the zone rebuild: dataset for our own DVS detector (YOLOv7 on DVS
frames), leave-one-scene-out over the six scenes, as in Gebauer et al.

Images: frames of {scene}_dvs.mp4 (1280x720, events accumulated between two RGB
triggers: grey = no event, white / black = ON / OFF), saved as PNG.

Training frames per scene:
  - frames with at least one certain, hand-placed (keyframe) DVS box, as in
    yolo/prepare.py; labels = ALL certain boxes of that frame (also the filled-in
    ones, so no visible insect is taught as background); frames that contain an
    uncertain box are left out (we cannot say whether it is an insect)
  - empty frames: 20 % as many frames without any DVS annotation within 50
    frames before and after, so the detector learns to stay silent

Folds: for each scene k, train = training frames of the other five scenes,
val = training frames of scene k (only for watching the training). After
training, each model runs on EVERY frame of its left-out scene's video
(dvs.slurm), which gives the detections for the zones.

Writes training/dataset_dvs/ (paths in the lists are PALMA paths):
  images/all/{scene}_frame_{f:06d}.png   labels/all/{scene}_frame_{f:06d}.txt
  folds/{scene}_train.txt, {scene}_val.txt, {scene}.yaml
and one preview per scene in outputs/step12/.

Run: PYTHONPATH=src ./venv/bin/python src/dti/dvs/dataset.py
"""

import csv
import os
from collections import defaultdict

import cv2
import numpy as np

from dti.io.data_sync import DATA_DIR
from dti.geometry.zone_v3 import ALL_SCENES
from dti.paths import TRAINING_ROOT

OUT = f"{TRAINING_ROOT}/dataset_dvs"
PALMA_DIR = "/scratch/tmp/mdenk/data/insect_dataset_dvs"
PREVIEW_DIR = "outputs/step12"
DVS_W, DVS_H = 1280, 720
EMPTY_SHARE = 0.2
GAP = 50


def read_annotations(scene):
    rows = defaultdict(list)
    with open(f"{DATA_DIR}/{scene}_annotation.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows[int(r["frame_index"])].append(r)
    return rows


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists -- remove it first to rebuild")
    for sub in ("images/all", "labels/all", "folds"):
        os.makedirs(f"{OUT}/{sub}")
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    stems = {}
    print(f"{'scene':8s} {'video frames':>12s} {'keyframe frames':>15s} {'left out (uncertain)':>20s} "
          f"{'training frames':>15s} {'boxes':>6s} {'empty frames':>12s}")
    for scene in ALL_SCENES:
        rows = read_annotations(scene)
        cap = cv2.VideoCapture(f"{DATA_DIR}/{scene}_dvs.mp4")
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        key = {f for f, rs in rows.items() if any(r["confidence"] == "certain" and r["is_keyframe"] == "True" for r in rs)}
        with_uncertain = {f for f in key if any(r["confidence"] != "certain" for r in rows[f])}
        train = sorted(f for f in key - with_uncertain if f < n_video)
        has = np.zeros(n_video, bool)
        has[[f for f in rows if f < n_video]] = True
        free = np.flatnonzero(np.convolve(has, np.ones(2 * GAP + 1), mode="same") == 0)
        n_empty = min(len(free), int(round(EMPTY_SHARE * len(train))))
        empty = sorted(int(free[i]) for i in np.unique(np.linspace(0, len(free) - 1, n_empty).round().astype(int))) \
            if n_empty else []
        wanted = {f: [] for f in empty}
        for f in train:
            wanted[f] = [r for r in rows[f] if r["confidence"] == "certain"]
        n_boxes = sum(len(v) for v in wanted.values())

        busiest = max(train, key=lambda f: (len(wanted[f]), -f))
        for f in range(n_video):  # sequential reading: exact frames, no seeking
            ok, img = cap.read()
            if not ok:
                raise SystemExit(f"{scene}: video ended at frame {f} of {n_video}")
            if f not in wanted:
                continue
            stem = f"{scene}_frame_{f:06d}"
            cv2.imwrite(f"{OUT}/images/all/{stem}.png", img)
            with open(f"{OUT}/labels/all/{stem}.txt", "w") as fh:
                for r in wanted[f]:
                    x, y, w, h = (float(r[k]) for k in ("left", "top", "width", "height"))
                    x0, y0, x1, y1 = max(0.0, x), max(0.0, y), min(DVS_W, x + w), min(DVS_H, y + h)
                    if x1 > x0 and y1 > y0:
                        fh.write(f"0 {(x0 + x1) / 2 / DVS_W:.6f} {(y0 + y1) / 2 / DVS_H:.6f} "
                                 f"{(x1 - x0) / DVS_W:.6f} {(y1 - y0) / DVS_H:.6f}\n")
            if f == busiest:
                prev = img.copy()
                for r in wanted[f]:
                    x, y, w, h = (float(r[k]) for k in ("left", "top", "width", "height"))
                    cv2.rectangle(prev, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                cap_txt = np.zeros((32, DVS_W, 3), np.uint8)
                cv2.putText(cap_txt, f"{stem}: DVS training frame, green box = certain DVS annotation "
                                     f"({len(wanted[f])})", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                            cv2.LINE_AA)
                cv2.imwrite(f"{PREVIEW_DIR}/{stem}.jpg", np.vstack([prev, cap_txt]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        cap.release()
        stems[scene] = [f"{scene}_frame_{f:06d}" for f in sorted(wanted)]
        print(f"{scene:8s} {n_video:12d} {len(key):15d} {len(with_uncertain):20d} {len(train):15d} {n_boxes:6d} "
              f"{len(empty):12d}", flush=True)

    print(f"\n{'left-out':8s} {'train images':>12s} {'val images':>10s}")
    for scene in ALL_SCENES:
        train = [s for other in ALL_SCENES if other != scene for s in stems[other]]
        for name, items in (("train", train), ("val", stems[scene])):
            with open(f"{OUT}/folds/{scene}_{name}.txt", "w") as f:
                f.writelines(f"{PALMA_DIR}/images/all/{s}.png\n" for s in items)
        with open(f"{OUT}/folds/{scene}.yaml", "w") as f:
            f.write(f"# DVS detector, left-out scene {scene}\ntrain: {PALMA_DIR}/folds/{scene}_train.txt\n"
                    f"val: {PALMA_DIR}/folds/{scene}_val.txt\n\nnc: 1\nnames: ['insect']\n")
        print(f"{scene:8s} {len(train):12d} {len(stems[scene]):10d}")
    size = sum(os.path.getsize(f"{OUT}/images/all/{f}") for f in os.listdir(f"{OUT}/images/all")) / 1e9
    print(f"\nwrote {OUT}: {sum(len(v) for v in stems.values())} images, {size:.2f} GB; previews in {PREVIEW_DIR}/")


if __name__ == "__main__":
    main()
