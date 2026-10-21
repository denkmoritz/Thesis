"""
Step 11d of the zone rebuild: can RGB motion tell where the insect is ALONG the zone?

The pilot showed that the network looks inside the zone but cannot find small
insects along it (scenes 3-5). Here we test, without any training, whether the
change between RGB frames marks the insect's position inside the zone.

Test frames: the last 40 % of scenes 1-5 (same frames as the pilot), zones as
stored in dataset_zone_v2 (weight > 0.5). Each connected piece of the zone is
treated on its own; a label belongs to the piece that contains its centre.

Motion maps (per pixel: largest absolute difference over the three colour
channels, frames slightly blurred first):
  prev      |I(f) - I(f-1)|
  three1    min(|I(f) - I(f-1)|, |I(f) - I(f+1)|)   three-frame differencing
  three5    the same with f-5 and f+5 (50 ms apart; 100 ms in scene 3)
  median    |I(f) - median of 11 frames from f-10 to f+10|   background model

Printed per map and scene:
  top-1 circle   strongest motion peak of the zone piece lies inside the smallest
                 circle around the label box (same rule as the scoring script)
  top-1 20px     ... lies within 20 px of the label centre
  top-3 20px     one of the three strongest peaks (at least 25 px apart) does
  chance 20px    share of the zone piece within 20 px of the label centre
                 = hit rate of a random position in the zone
  rank           how the strongest motion in the label box ranks among all pixels
                 of the zone piece (median; 1 % = among the strongest 1 %)

Run: PYTHONPATH=src ./venv/bin/python src/dti/motion/cue.py
"""

import glob
import os

import cv2
import numpy as np

from dti.io.data_sync import DATA_DIR, SCENES, RGB_W, RGB_H
from dti.paths import TRAINING_ROOT

DATASET = f"{TRAINING_ROOT}/dataset_zone_v2"
OUT_DIR = "outputs/step11d"
OFFSETS = sorted({-10, -8, -6, -5, -4, -2, -1, 0, 1, 2, 4, 5, 6, 8, 10})
MEDIAN_OFFSETS = list(range(-10, 11, 2))
MAPS = ["prev", "three1", "three5", "median"]
NEAR_PX = 20.0
PEAK_SEP = 25
MIN_PIECE = 50  # px
WHITE, GREEN, CYAN = (255, 255, 255), (0, 255, 0), (255, 255, 0)


def read_frames(cap, f, n_video):
    """Blurred frames f + offset for all OFFSETS (None if outside the video), read in one pass."""
    start = max(0, f + OFFSETS[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = {}
    for i in range(start, min(n_video - 1, f + OFFSETS[-1]) + 1):
        ok, img = cap.read()
        if not ok:
            break
        if i - f in OFFSETS:
            frames[i - f] = img
    return frames


def diff(a, b):
    return cv2.absdiff(a, b).max(axis=2)


def motion_maps(frames):
    blur = {k: cv2.GaussianBlur(v, (0, 0), 1.5) for k, v in frames.items()}
    out = {}
    if -1 in blur:
        out["prev"] = diff(blur[0], blur[-1])
    for name, g in (("three1", 1), ("three5", 5)):
        if -g in blur and g in blur:
            out[name] = np.minimum(diff(blur[0], blur[-g]), diff(blur[0], blur[g]))
    stack = [blur[k] for k in MEDIAN_OFFSETS if k in blur]
    if len(stack) >= 5:  # at the very start / end of a video only one side is available
        out["median"] = diff(blur[0], np.median(np.stack(stack), axis=0).astype(np.uint8))
    return {k: cv2.blur(v, (5, 5)) for k, v in out.items()}


def peaks(m, piece, n=3):
    """The n strongest positions of map m inside the zone piece, at least PEAK_SEP px apart: (x, y)."""
    v = np.where(piece, m.astype(np.float32), -1.0)
    out = []
    for _ in range(n):
        y, x = np.unravel_index(int(v.argmax()), v.shape)
        if v[y, x] < 0:
            break
        out.append((x, y))
        cv2.circle(v, (int(x), int(y)), PEAK_SEP, -1.0, -1)
    return out


def load_labels(stem):
    boxes = []
    for line in open(f"{DATASET}/labels/val/{stem}.txt"):
        if line.strip():
            _, cx, cy, w, h = map(float, line.split()[:5])
            boxes.append((cx * RGB_W, cy * RGB_H, w * RGB_W, h * RGB_H))
    return boxes


def render(scene, f, frame, mask, boxes, m, top1, name):
    left = frame.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(left, contours, -1, WHITE, 2)
    scale = 255.0 / max(1.0, float(np.percentile(m[mask], 99.9)))
    right = cv2.applyColorMap(np.uint8(np.clip(m * scale, 0, 255)), cv2.COLORMAP_INFERNO)
    right[~mask] = (right[~mask] * 0.25).astype(np.uint8)
    cv2.drawContours(right, contours, -1, WHITE, 2)
    for img in (left, right):
        for cx, cy, w, h in boxes:
            pad = max(0, (14 - min(w, h)) / 2)
            cv2.rectangle(img, (int(cx - w / 2 - pad), int(cy - h / 2 - pad)),
                          (int(cx + w / 2 + pad), int(cy + h / 2 + pad)), GREEN, 2)
    for x, y in top1:
        cv2.circle(right, (int(x), int(y)), 20, CYAN, 2)
    canvas = cv2.resize(np.hstack([left, np.full((RGB_H, 16, 3), 40, np.uint8), right]), None, fx=0.5, fy=0.5,
                        interpolation=cv2.INTER_AREA)
    lines = [
        f"{scene}  RGB frame {f}   left: RGB frame   right: motion map '{name}' (bright = strong change; "
        "outside the zone darkened)",
        f"white outline = zone (weight > 0.5);  green box = RGB label ({len(boxes)}), tiny boxes padded;  "
        "cyan circle (20 px) = strongest motion in each zone piece",
    ]
    cap = np.zeros((22 * len(lines) + 10, canvas.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(cap, t, (8, 22 * i + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.imwrite(f"{OUT_DIR}/{scene}_frame_{f:06d}_{name}.jpg", np.vstack([canvas, cap]), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    yy, xx = np.mgrid[0:RGB_H, 0:RGB_W]
    results = {}
    for scene in SCENES:
        stems = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{DATASET}/images/val/{scene}_frame_*.jpg"))
        cap = cv2.VideoCapture(f"{DATA_DIR}/{scene}_rgb.mp4")
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rec = {k: dict(circle=[], near=[], near3=[], rank=[]) for k in MAPS}
        chance, n_out, n_nozone, seek_err, kept = [], 0, 0, [], []
        for stem in stems:
            f = int(stem[-6:])
            boxes = load_labels(stem)
            zone = cv2.imread(f"{DATASET}/zones/val/{stem}.png", cv2.IMREAD_GRAYSCALE)
            if zone.min() == 255 or not boxes:  # no DVS insect -> no zone
                n_nozone += len(boxes)
                continue
            frames = read_frames(cap, f, n_video)
            seek_err.append(float(cv2.absdiff(frames[0], cv2.imread(f"{DATASET}/images/val/{stem}.jpg")).mean()))
            maps = motion_maps(frames)
            n_pieces, lab = cv2.connectedComponents((zone > 153).astype(np.uint8))
            top1 = {k: [] for k in maps}
            for cx, cy, w, h in boxes:
                piece_id = lab[int(np.clip(cy, 0, RGB_H - 1)), int(np.clip(cx, 0, RGB_W - 1))]
                piece = lab == piece_id
                if piece_id == 0 or piece.sum() < MIN_PIECE:
                    n_out += 1
                    continue
                dist = np.hypot(xx - cx, yy - cy)
                chance.append(float(np.mean(dist[piece] <= NEAR_PX)))
                box = (np.abs(xx - cx) <= w / 2) & (np.abs(yy - cy) <= h / 2)
                for k, m in maps.items():
                    pk = peaks(m, piece)
                    d = [float(np.hypot(x - cx, y - cy)) for x, y in pk]
                    rec[k]["circle"].append(d[0] <= np.hypot(w, h) / 2)
                    rec[k]["near"].append(d[0] <= NEAR_PX)
                    rec[k]["near3"].append(min(d) <= NEAR_PX)
                    rec[k]["rank"].append(float(np.mean(m[piece] >= m[box].max())))
                    top1[k].append(pk[0])
            kept.append((len(boxes), stem, f, frames[0], zone > 153, boxes, maps, top1))
        cap.release()
        results[scene] = (rec, chance, n_out, n_nozone, seek_err)
        # example: the test frame with most labels, shown with the background-model map
        if kept:
            _, stem, f, frame, mask, boxes, maps, top1 = max(kept, key=lambda t: (t[0], t[1]))
            name = "median" if "median" in maps else next(iter(maps))
            render(scene, f, frame, mask, boxes, maps[name], top1[name], name)
        print(f"{scene}: {len(kept)} frames done, video frame vs dataset image mean abs difference "
              f"{np.mean(seek_err):.2f} grey levels (max {np.max(seek_err):.2f})", flush=True)

    print(f"\n{'scene':8s} {'labels':>6s} {'outside zone':>12s} {'chance 20px':>11s} | {'map':7s} {'top-1 circle':>12s} "
          f"{'top-1 20px':>10s} {'top-3 20px':>10s} {'rank':>6s}")
    for scene, (rec, chance, n_out, n_nozone, _) in results.items():
        for i, k in enumerate(MAPS):
            r = rec[k]
            head = (f"{scene:8s} {len(chance):6d} {n_out:12d} {np.mean(chance):11.0%}" if i == 0 else f"{'':40s}")
            print(f"{head} | {k:7s} {np.mean(r['circle']):12.0%} {np.mean(r['near']):10.0%} {np.mean(r['near3']):10.0%} "
                  f"{np.median(r['rank']):6.1%}")
    print("\nlabels = RGB labels whose centre lies in a zone piece; outside zone = labels outside every piece (not counted)")
    print("chance 20px = hit rate of a random position in the zone piece; rank = median share of zone pixels that are")
    print("at least as strong as the strongest pixel in the label box (small = the insect stands out)")


if __name__ == "__main__":
    main()
