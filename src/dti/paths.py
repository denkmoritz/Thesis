"""
The only place that knows where data lives outside the repository.

Reads config/paths.yaml (not in git; copy config/paths.example.yaml). Simple
"key: value" lines, no YAML library needed. Paths inside the repository (outputs/,
data/output/, data/ictrap-2022-calibration/) stay relative and need the repository
root as working directory, as before.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = os.path.join(ROOT, "config", "paths.yaml")


def _read(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing -- copy config/paths.example.yaml to config/paths.yaml and fill it in")
    out = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                key, value = line.split(":", 1)
                out[key.strip()] = os.path.expanduser(value.strip().strip('"'))
    return out


_cfg = _read(CONFIG)

TRAINING_ROOT = _cfg["training_root"]  # videos, annotation CSVs, CVAT exports, built datasets
RAW_DATA_DIR = _cfg["raw_data_dir"]  # h5 event files (trigger times) and scene-6 privacy masks
