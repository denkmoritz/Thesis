"""
Step 10 of the zone rebuild: which time interval does DVS video frame f show?

Label-free timing check. The .h5 files hold the raw events and the corrected RGB
trigger timestamps with the RGB frame index. A DVS video frame is the events
accumulated between two triggers (Gebauer et al.). Here we rebuild the frame
from the raw events for several candidate intervals and compare it with frame f
of {scene}_dvs.mp4. The interval that reproduces the video frame is the answer;
no RGB label and no DVS annotation is used.

Trigger pulses: the positive (rising) edge labelled f is written P[f], the
negative (falling) edge labelled f is N[f]. The first positive edge is P[-1].

Rendering of the video (found by inspection): grey 127 = no event, white = last
event at the pixel was ON, black = last event was OFF.

Printed per scene:
  - trigger period, pulse length (P -> N), and N[f] - P[f]
  - for every candidate interval: median number of pixels that differ from the
    video frame (event / no event), and share of frames reproduced exactly
  - for the best interval: share of pixels with the right colour (ON / OFF)

Needs h5py, which is only in the yolo venv:
Run: PYTHONPATH=src ../yolo/venv/bin/python src/dti/io/trigger_timing.py
"""

import os

import cv2
import h5py
import numpy as np

from dti.io.data_sync import SCENES, DATA_DIR
from dti.paths import RAW_DATA_DIR

H5_DIR = RAW_DATA_DIR
OUT_DIR = "outputs/step10"
ALL_SCENES = SCENES + ["6_h-h-h"]
H5_NAME = {s: s for s in SCENES} | {"6_h-h-h": "6_h-h-h_filtered"}
DVS_W, DVS_H = 1280, 720
N_FRAMES = 60  # per scene, spread over the frames with most events
WHITE = (255, 255, 255)

# name -> (start edge, start offset, end edge, end offset), relative to video frame f
CANDIDATES = {
    "P[f-2] .. P[f-1]": ("P", -2, "P", -1),
    "P[f-1] .. P[f]": ("P", -1, "P", 0),
    "P[f] .. P[f+1]": ("P", 0, "P", 1),
    "N[f-1] .. N[f]": ("N", -1, "N", 0),
    "N[f] .. N[f+1]": ("N", 0, "N", 1),
    "N[f+1] .. N[f+2]": ("N", 1, "N", 2),
    "N[f] .. P[f]": ("N", 0, "P", 0),
    "N[f+1] .. P[f+1]": ("N", 1, "P", 1),
}


def load(scene):
    with h5py.File(f"{H5_DIR}/{H5_NAME[scene]}.h5", "r") as h:
        ev = h["CD/events"][:]
        pos = h["EXTERNAL_TRIGGERS/corrected_positive"][:]
        neg = h["EXTERNAL_TRIGGERS/corrected_negative"][:]
    edges = {"P": dict(zip(pos["f"].tolist(), pos["t"].tolist())),
             "N": dict(zip(neg["f"].tolist(), neg["t"].tolist()))}
    return ev, edges


def render(ev, t0, t1):
    """Accumulated frame of the events in [t0, t1): 0 = none, 1 = last event OFF, 2 = last event ON."""
    a, b = np.searchsorted(ev["t"], [t0, t1])
    e = ev[a:b]
    img = np.zeros((DVS_H, DVS_W), np.uint8)
    img[e["y"], e["x"]] = e["p"] + 1  # events are time-sorted, so the last one wins
    return img


def classes(frame_bgr):
    """Video frame -> 0 = grey, 1 = black, 2 = white."""
    g = frame_bgr[..., 0].astype(np.int16)
    return np.where(g > 191, 2, np.where(g < 64, 1, 0)).astype(np.uint8)


def to_bgr(cls):
    return cv2.cvtColor(np.array([127, 0, 255], np.uint8)[cls], cv2.COLOR_GRAY2BGR)


def pick_frames(ev, edges, n_video):
    """Frames with many events, spread over the recording (busy frames separate the candidates best)."""
    frames = np.arange(3, n_video - 3)
    t = np.array([edges["P"][f] for f in range(2, n_video - 2)])
    counts = np.diff(np.searchsorted(ev["t"], t))[:len(frames)]
    chunks = np.array_split(np.arange(len(frames)), N_FRAMES)
    return [int(frames[c[np.argmax(counts[c])]]) for c in chunks if len(c)]


def figure(scene, f, video_cls, ev, edges, ranked, path):
    panels = [("video frame", video_cls)]
    for name, _ in ranked[:2]:
        s_e, s_o, e_e, e_o = CANDIDATES[name]
        panels.append((name, render(ev, edges[s_e][f + s_o], edges[e_e][f + e_o])))
    imgs = [cv2.resize(to_bgr(c), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST) for _, c in panels]
    gap = np.full((imgs[0].shape[0], 8, 3), 40, np.uint8)
    canvas = np.hstack([imgs[0], gap, imgs[1], gap, imgs[2]])
    diff = [int(np.sum((c > 0) != (video_cls > 0))) for _, c in panels[1:]]
    lines = [
        f"{scene}  DVS frame {f}   left: frame {f} of the DVS video   middle: rebuilt from raw events, "
        f"interval {panels[1][0]}   right: interval {panels[2][0]}",
        "grey = no event, white = last event ON, black = last event OFF;  P[f] / N[f] = rising / falling trigger "
        "edge labelled with RGB frame f",
        f"pixels that differ from the video frame (event / no event): middle {diff[0]}, right {diff[1]};  "
        f"event pixels in the video frame: {int(np.sum(video_cls > 0))}",
    ]
    cap = np.zeros((22 * len(lines) + 10, canvas.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(cap, t, (8, 22 * i + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.imwrite(path, np.vstack([canvas, cap]), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for scene in ALL_SCENES:
        ev, edges = load(scene)
        cap = cv2.VideoCapture(f"{DATA_DIR}/{scene}_dvs.mp4")
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fs = np.array(sorted(set(edges["P"]) & set(edges["N"])))
        p, n = (np.array([edges[k][f] for f in fs]) for k in "PN")
        print(f"\n{scene}: {len(ev):,} events, video frames {n_video}, trigger labels P {min(edges['P'])}.."
              f"{max(edges['P'])}, N {min(edges['N'])}..{max(edges['N'])}")
        print(f"  trigger period (P[f+1]-P[f]) median {np.median(np.diff(p)) / 1000:.3f} ms;  "
              f"N[f]-P[f] median {np.median(n - p) / 1000:+.3f} ms;  "
              f"N[f]-P[f-1] (pulse length) median {np.median(n[1:] - p[:-1]) / 1000:+.3f} ms")

        frames = pick_frames(ev, edges, n_video)
        mask_diff = {name: [] for name in CANDIDATES}
        colour_ok = {name: [] for name in CANDIDATES}
        video = {}
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            assert ok, (scene, f)
            video[f] = v = classes(img)
            for name, (s_e, s_o, e_e, e_o) in CANDIDATES.items():
                r = render(ev, edges[s_e][f + s_o], edges[e_e][f + e_o])
                mask_diff[name].append(int(np.sum((r > 0) != (v > 0))))
                both = (r > 0) & (v > 0)
                colour_ok[name].append(float(np.mean(r[both] == v[both])) if both.any() else np.nan)
        cap.release()

        n_ev = np.median([int(np.sum(video[f] > 0)) for f in frames])
        ranked = sorted(((name, float(np.median(d))) for name, d in mask_diff.items()), key=lambda x: x[1])
        print(f"  {len(frames)} frames, median event pixels in the video frame: {n_ev:.0f}")
        print(f"  {'interval':18s} {'median diff px':>14s} {'exact frames':>12s} {'colour right':>12s}")
        for name, med in ranked:
            d = np.array(mask_diff[name])
            print(f"  {name:18s} {med:14.0f} {np.mean(d == 0):12.0%} {np.nanmean(colour_ok[name]):12.1%}")
        print(f"  -> best: {ranked[0][0]}")

        busiest = max(frames, key=lambda f: int(np.sum(video[f] > 0)))
        figure(scene, busiest, video[busiest], ev, edges, ranked, f"{OUT_DIR}/{scene}_frame_{busiest:06d}.jpg")
        print(f"  image: frame {busiest}")


if __name__ == "__main__":
    main()
