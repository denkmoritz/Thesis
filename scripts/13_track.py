"""Tracker on the DVS detections, settings chosen leave-one-scene-out (results/logs/step20_log.txt).
Old steps 19, 20. Run tracker.py --times once first (needs h5py)."""
from dti.tracking import tune

tune.main()
