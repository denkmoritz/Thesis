"""
Run YOLOv7's train.py with a different random seed -- runs on PALMA.

YOLOv7 fixes its seed inside train.py (init_seeds(2 + rank)), so repeating a run
repeats the same random choices (augmentation, order of the images). This wrapper
shifts that seed by SEED_OFFSET without changing train.py: it replaces
utils.general.init_seeds before train.py is executed, so train.py imports the
shifted version. Everything else, including the zone hooks, is unchanged.

Copy into the YOLOv7 repo root (/scratch/tmp/mdenk/yolov7/) and use it like train.py:

    SEED_OFFSET=1 python3 train_seed.py --weights ... (same arguments as train.py)
"""

import os
import runpy
import sys

import utils.general as general

OFFSET = int(os.environ["SEED_OFFSET"])
_original = general.init_seeds


def init_seeds(seed=0):
    print(f"train_seed.py: seed {seed} -> {seed + OFFSET}")
    _original(seed + OFFSET)


general.init_seeds = init_seeds
sys.argv = ["train.py"] + sys.argv[1:]
runpy.run_path("train.py", run_name="__main__")
