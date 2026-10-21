"""
Step 19 of the zone rebuild: a tracker on the DVS detections.

Input: the detections of our DVS detector on every video frame
(outputs/dvs/frames/<scene>.npz, step 12b) and the real time of every
frame (trigger timestamps, so scene 3 at about 50 fps needs no special case).

The tracker, whole video (it may look ahead):
  state        centre and speed of one insect in DVS px; x and y are filtered
               alike (constant-velocity Kalman filter)
  motion       the speed may drift by SPEED_DRIFT box sizes per second within one
               second (process noise)
  measurement  how far a detection centre lies from the true centre, as a share
               of the detection's box size: measured on the hand-placed certain
               annotations of the OTHER five scenes, not tuned
  gate         a detection can join a track only if it lies inside the 99 % region
               of the predicted position (GATE, chi-square with 2 degrees of freedom)
               or inside the track's box at the predicted position (the measured
               centre error is 1-2 px, so the 99 % region alone is smaller than the
               insect and split one insect into several tracks)
  assignment   Hungarian method on these distances (scipy linear_sum_assignment)
  duplicates   two boxes of one frame on the same insect (each holds the other's
               centre): only the more confident one is used
  start        an unassigned detection with confidence >= the start cut-off
  continue     any detection with confidence >= the continue cut-off
  end          no detection for MAX_GAP seconds
  kept         tracks with at least MIN_HITS detections
  gaps         frames between two detections of a track get a box drawn on the
               straight line between the two (by time); nothing is added before
               the first or after the last detection of a track
  confidence   every box of a track gets the track's mean detection confidence

The settings below are first values. Step 20 chooses them on the other five
scenes. The start cut-off is the scene's cut-off of step 17 (already chosen on
the other five scenes).

Output: outputs/step19/tracks/<scene>.npz
  tracks (n, 9) = frame, x1, y1, x2, y2, track confidence, track id,
                  detected (1) or filled in (0), detection confidence (0 if filled in)
Images: outputs/step19/<scene>_tracks.jpg and <scene>_gap.jpg

Run once, to read the frame times (needs h5py):
  PYTHONPATH=src ../yolo/venv/bin/python src/dti/tracking/tracker.py --times
Then:
  PYTHONPATH=src ./venv/bin/python src/dti/tracking/tracker.py
"""

import csv
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from dti.io.data_sync import DATA_DIR, read_frame
from dti.evaluation.metrics import best_f1
from dti.geometry.zone_v3 import ALL_SCENES
from dti.datasets.auto_chain import detections, detector_curve
from dti.paths import RAW_DATA_DIR

OUT_DIR = "outputs/step19"
TIMES = f"{OUT_DIR}/frame_times.npz"
CUTS = "outputs/step17/auto_corrections.json"
H5_DIR = RAW_DATA_DIR

GATE = 9.21  # 99 % of a chi-square distribution with 2 degrees of freedom
START_SPEED = 20.0  # box sizes per second: uncertainty of the speed of a new track
CONTINUE_SHARE = 0.5  # continue cut-off = this share of the start cut-off
MIN_HITS = 3
MAX_GAP = 0.2  # seconds
SPEED_DRIFT = 10.0  # box sizes per second, within one second
SIZE_WINDOW = 5  # detections for the median box size of a track

GREEN, ORANGE, WHITE = (0, 255, 0), (0, 165, 255), (255, 255, 255)


def export_times():
    import h5py
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {}
    for scene in ALL_SCENES:
        name = scene if scene != "6_h-h-h" else "6_h-h-h_filtered"
        with h5py.File(f"{H5_DIR}/{name}.h5", "r") as h:
            trig = h["EXTERNAL_TRIGGERS/corrected_positive"][:]
        t = dict(zip(trig["f"].tolist(), trig["t"].tolist()))  # DVS frame f ends at trigger f
        out[scene] = np.array([t[f] for f in range(max(t) + 1)], float) / 1e6
        d = np.diff(out[scene])
        print(f"{scene}: {len(out[scene])} frames, {1000 * np.median(d):.2f} ms per frame "
              f"(min {1000 * d.min():.2f}, max {1000 * d.max():.2f})")
    np.savez(TIMES, **out)


def centre_error(scene, det, cut):
    """Detection centre minus annotation centre on hand-placed certain boxes, as a share of the detection size."""
    key = defaultdict(list)
    with open(f"{DATA_DIR}/{scene}_annotation.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["is_keyframe"] == "True" and r["confidence"] == "certain":
                key[int(r["frame_index"])].append([float(r[k]) for k in ("left", "top", "width", "height")])
    det = det[det[:, 5] >= cut]
    by_frame = defaultdict(list)
    for row in det:
        by_frame[int(row[0])].append(row)
    err = []
    for fr, boxes in key.items():
        d = np.array(by_frame.get(fr, [])).reshape(-1, 6)
        if not len(d):
            continue
        c = np.c_[(d[:, 1] + d[:, 3]) / 2, (d[:, 2] + d[:, 4]) / 2]
        size = np.sqrt((d[:, 3] - d[:, 1]) * (d[:, 4] - d[:, 2]))
        for x, y, w, h in boxes:
            off = c - (x + w / 2, y + h / 2)
            i = int(np.hypot(*off.T).argmin())
            if np.hypot(*off[i]) <= np.hypot(w, h) / 2:  # the circle rule of the scoring
                err.extend(off[i] / size[i])
    return np.array(err)


def robust_std(e):
    return 1.4826 * float(np.median(np.abs(e - np.median(e))))


def one_per_insect(d):
    """Drop a detection if it and a more confident one of the same frame each hold the other's centre."""
    d = d[np.argsort(-d[:, 5], kind="stable")]
    c = np.c_[(d[:, 1] + d[:, 3]) / 2, (d[:, 2] + d[:, 4]) / 2]
    inside = ((c[:, None, 0] >= d[None, :, 1]) & (c[:, None, 0] <= d[None, :, 3]) &
              (c[:, None, 1] >= d[None, :, 2]) & (c[:, None, 1] <= d[None, :, 4]))  # centre i in box j
    both = inside & inside.T
    keep = []
    for i in range(len(d)):
        if not both[i, keep].any():
            keep.append(i)
    return d[keep]


def track(det, times, start, cont, noise, min_hits=MIN_HITS, max_gap=MAX_GAP, drift=SPEED_DRIFT):
    """det (n, 6) = frame, x1, y1, x2, y2, conf. Returns the tracks array of the module text and the counts."""
    det = det[(det[:, 5] >= cont) & (det[:, 0] < len(times))]
    det = det[np.argsort(det[:, 0], kind="stable")]
    frames, first = np.unique(det[:, 0].astype(int), return_index=True)
    last = np.r_[first[1:], len(det)]
    live, done = [], []
    for fr, a, b in zip(frames, first, last):
        t, d = times[fr], one_per_insect(det[a:b])
        done += [k for k in live if t - k["t_det"] > max_gap]
        live = [k for k in live if t - k["t_det"] <= max_gap]
        for k in live:  # predict to the time of this frame
            dt, q = t - k["t"], (drift * k["size"]) ** 2
            k["x"] = k["x"] + k["v"] * dt
            p11, p12, p22 = k["P"]
            k["P"] = (p11 + 2 * dt * p12 + dt * dt * p22 + q * dt ** 3 / 3, p12 + dt * p22 + q * dt * dt / 2, p22 + q * dt)
            k["t"] = t
        c = np.c_[(d[:, 1] + d[:, 3]) / 2, (d[:, 2] + d[:, 4]) / 2]
        size = np.sqrt((d[:, 3] - d[:, 1]) * (d[:, 4] - d[:, 2]))
        free = np.ones(len(d), bool)
        if live:
            x = np.array([k["x"] for k in live])
            s = np.array([k["P"][0] + (noise * k["size"]) ** 2 for k in live])
            off = np.abs(x[:, None] - c[None])
            d2 = (off ** 2).sum(2) / s[:, None]
            half = np.array([k["size"] for k in live])[:, None] / 2
            ok = (d2 <= GATE) | ((off[:, :, 0] <= half) & (off[:, :, 1] <= half))  # in the gate or in the track's box
            rows, cols = linear_sum_assignment(np.where(ok, d2, 1e6))
            for i, j in zip(rows, cols):
                if not ok[i, j]:
                    continue
                k, (p11, p12, p22) = live[i], live[i]["P"]
                g1, g2, res = p11 / s[i], p12 / s[i], c[j] - live[i]["x"]
                k["x"], k["v"] = k["x"] + g1 * res, k["v"] + g2 * res
                k["P"] = ((1 - g1) * p11, (1 - g1) * p12, p22 - g2 * p12)
                k["hits"].append(d[j])
                k["size"] = float(np.median([np.sqrt((h[3] - h[1]) * (h[4] - h[2])) for h in k["hits"][-SIZE_WINDOW:]]))
                k["t_det"] = t
                free[j] = False
        for j in np.nonzero(free & (d[:, 5] >= start))[0]:
            live.append(dict(x=c[j], v=np.zeros(2), P=((noise * size[j]) ** 2, 0.0, (START_SPEED * size[j]) ** 2),
                             size=float(size[j]), t=t, t_det=t, hits=[d[j]]))
    done += live

    rows, n_id = [], 0
    for k in done:
        h = np.array(k["hits"])
        if len(h) < min_hits:
            continue
        score = float(h[:, 5].mean())
        for i in range(len(h)):
            rows.append([h[i, 0], *h[i, 1:5], score, n_id, 1, h[i, 5]])
            if i + 1 < len(h):
                f0, f1 = int(h[i, 0]), int(h[i + 1, 0])
                for f in range(f0 + 1, f1):
                    w = (times[f] - times[f0]) / (times[f1] - times[f0])
                    rows.append([f, *((1 - w) * h[i, 1:5] + w * h[i + 1, 1:5]), score, n_id, 0, 0.0])
        n_id += 1
    tracks = np.array(rows).reshape(-1, 9)
    return tracks, dict(detections=len(det), started=len(done), kept=n_id)


def best_point(scene, boxes, n_video):
    """Best F1 over all confidence cut-offs, with found (recall) and right (precision) there, in %."""
    conf, hit, n_gt = detector_curve(scene, boxes, n_video)
    f1, thr = best_f1(conf, hit, n_gt)
    tp, n = hit[conf >= thr].sum(), np.sum(conf >= thr)
    return 100 * f1, 100 * tp / n_gt, 100 * tp / max(n, 1)


def draw(img, tracks, f):
    for r in tracks[tracks[:, 0] == f]:
        x1, y1, x2, y2 = r[1:5]
        pad = max(0, (12 - min(x2 - x1, y2 - y1)) / 2)
        cv2.rectangle(img, (int(x1 - pad), int(y1 - pad)), (int(x2 + pad), int(y2 + pad)), GREEN if r[7] else ORANGE, 1)
        cv2.putText(img, str(int(r[6])), (int(x2 + pad) + 3, int(y1 - pad) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    WHITE, 1, cv2.LINE_AA)
    return img


def save(path, img, lines):
    cap = np.zeros((22 * len(lines) + 10, img.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(cap, t, (8, 22 * i + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.imwrite(path, np.vstack([img, cap]), [cv2.IMWRITE_JPEG_QUALITY, 92])


LEGEND = "green box = detection of the track, orange box = filled in (no detection in this frame), number = track id; " \
         "boxes under 12 px are drawn as 12 px"


def images(scene, tracks, times):
    per_frame = np.bincount(tracks[:, 0].astype(int))
    f = int(per_frame.argmax())
    save(f"{OUT_DIR}/{scene}_tracks.jpg", draw(read_frame(scene, "dvs", f), tracks, f),
         [f"{scene}  DVS frame {f} (the frame with most track boxes: {per_frame[f]})", LEGEND])

    # the longest filled gap of the track with most detections that has a gap of 3 or more frames
    best = None
    for tid in np.unique(tracks[:, 6]):
        k = tracks[tracks[:, 6] == tid]
        fr = k[k[:, 7] == 1, 0].astype(int)
        gap = np.diff(fr) - 1
        if len(gap) and gap.max() >= 3 and (best is None or len(fr) > best[0]):
            i = int(gap.argmax())
            best = (len(fr), int(tid), fr[i], fr[i + 1])
    if best is None:
        print(f"  {scene}: no gap of 3 or more frames, no gap image")
        return
    n, tid, f0, f1 = best
    mid = (f0 + f1) // 2
    panels = [draw(read_frame(scene, "dvs", f), tracks, f) for f in (f0, mid, f1)]
    save(f"{OUT_DIR}/{scene}_gap.jpg", np.vstack(panels),
         [f"{scene}  track {tid} ({n} detections). Top: DVS frame {f0}, last detection before the gap.",
          f"Middle: frame {mid}, inside the gap. Bottom: frame {f1}, first detection after the gap.",
          f"The gap is {f1 - f0 - 1} frames = {1000 * (times[f1] - times[f0]):.0f} ms long; the detector had no box "
          f"for this insect there, the tracker filled one in.", LEGEND])


def main():
    if "--times" in sys.argv:
        return export_times()
    os.makedirs(f"{OUT_DIR}/tracks", exist_ok=True)
    times, cuts = np.load(TIMES), json.load(open(CUTS))
    det = {s: detections(s) for s in ALL_SCENES}
    err = {s: centre_error(s, det[s][0], cuts[s]["confidence_cutoff"]) for s in ALL_SCENES}

    print("Centre error of the detector on hand-placed certain boxes, as a share of the detection's box size")
    print(f"{'scene':8s} {'pairs':>6s} {'own scene':>9s} {'other five (used)':>17s}")
    noise = {}
    for s in ALL_SCENES:
        noise[s] = robust_std(np.concatenate([err[o] for o in ALL_SCENES if o != s]))
        print(f"{s:8s} {len(err[s]) // 2:6d} {robust_std(err[s]):9.3f} {noise[s]:17.3f}")

    print(f"\nTracker with the first settings: continue cut-off = {CONTINUE_SHARE} x start cut-off, at least {MIN_HITS} "
          f"detections, gap up to {1000 * MAX_GAP:.0f} ms, speed drift {SPEED_DRIFT:g}")
    print(f"{'scene':8s} {'start':>5s} {'cont.':>5s} | {'det.':>6s} {'tracks':>6s} {'kept':>5s} | {'boxes':>6s} "
          f"{'filled':>6s} {'share':>5s} | {'median track':>12s} | {'detector: F1 found right':>24s} | "
          f"{'tracker: F1 found right':>23s}")
    for s in ALL_SCENES:
        d, n_video = det[s]
        start = cuts[s]["confidence_cutoff"]
        tracks, n = track(d, times[s], start, CONTINUE_SHARE * start, noise[s])
        np.savez_compressed(f"{OUT_DIR}/tracks/{s}.npz", tracks=tracks)
        filled = int(np.sum(tracks[:, 7] == 0))
        length = [np.ptp(times[s][tracks[tracks[:, 6] == i, 0].astype(int)]) for i in range(n["kept"])]
        before, after = best_point(s, d, n_video), best_point(s, tracks[:, :6], n_video)
        print(f"{s:8s} {start:5.2f} {CONTINUE_SHARE * start:5.2f} | {n['detections']:6d} {n['started']:6d} {n['kept']:5d} | "
              f"{len(tracks):6d} {filled:6d} {100 * filled / max(len(tracks), 1):4.0f}% | {1000 * np.median(length):9.0f} ms | "
              f"{before[0]:12.1f} {before[1]:5.0f}% {before[2]:4.0f}% | {after[0]:11.1f} {after[1]:5.0f}% {after[2]:4.0f}%")
        images(s, tracks, times[s])
    print("\ndet. = detections above the continue cut-off; tracks = tracks started; kept = with enough detections")
    print("found = of 100 certain DVS annotations (all video frames), how many got a box; right = of 100 boxes, how "
          "many are on an annotation")
    print("F1 = one number for both (high only if found and right are both high); all three at the confidence cut-off "
          "with the best F1 of the scene, as in step 12b")
    print("detector = the detections alone; tracker = the boxes of the kept tracks. First settings, not yet chosen on "
          "the other scenes (step 20).")


if __name__ == "__main__":
    main()
