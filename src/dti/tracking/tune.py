"""
Step 20 of the zone rebuild: choose the tracker settings of step 19 on the other
five scenes, then measure on the left-out scene whether tracking helps.

1. Settings search. Every combination of
     CONTINUE  continue cut-off as a share of the start cut-off
     MIN_HITS  detections a track needs to be kept
     MAX_GAP   seconds without a detection before a track ends
     DRIFT     how fast the speed may change (box sizes per second, within one second)
   is run on every scene (start cut-off and centre error as in step 19). For each
   left-out scene the combination with the best mean F1 on the OTHER five scenes is
   chosen. F1 as in step 12b: circle rule, certain DVS annotations, all video frames,
   best confidence cut-off of the scene.
   Small leak, said once: the start cut-off and the centre error of a training scene
   were themselves measured on the five scenes other than it, which includes the
   left-out scene. Neither is tuned on F1.

2. Scoring on the left-out scene, detector alone vs tracker with the chosen settings:
   a. DVS F1 / found / right, as above (all certain annotations).
   b. The same on hand-placed frames only: frames whose every annotation row is a
      keyframe. Check against a bias: the annotation tool fills in boxes between
      keyframes by drawing straight lines, as the tracker does in its gaps, so (a)
      could reward the tracker for copying the tool.
   c. RGB labels inside a zone (zone weight > 0.5 at the label centre) on the
      labelled RGB frames, zones from boxes at DVS frames f-1 and f as in step 17;
      detector = detections above the cut-off, tracker = all boxes of kept tracks.
      Zone area (share of the RGB image with weight > 0.5) next to it, because more
      zones always cover more labels.

Output: outputs/step20/search.json (F1 of every combination and scene),
outputs/step20/tracks/<scene>.npz (chosen settings, columns as step 19),
images outputs/step20/<scene>_tracks.jpg and <scene>_gap.jpg.

Run: PYTHONPATH=src ./venv/bin/python src/dti/tracking/tune.py
"""

import csv
import itertools
import json
import os
from collections import defaultdict

import numpy as np

from dti.tracking import tracker as s19
from dti.io import scene6 as s6
from dti.io.data_sync import DATA_DIR, RGB_W, RGB_H, load_rgb, center
from dti.geometry.rectify import load_dvs, setup
from dti.geometry.zone import Zone, raw_to_rect
from dti.evaluation.metrics import evaluate, best_f1
from dti.geometry.zone_v3 import ALL_SCENES, Z_NEAR, NEG_SHARE_MAX, DISP_MARGIN, scene_stats, sigma_from
from dti.datasets.auto_chain import detections, records, K

OUT_DIR = "outputs/step20"
GRID = dict(CONTINUE=[0.25, 0.5, 1.0], MIN_HITS=[2, 3, 5], MAX_GAP=[0.05, 0.1, 0.2, 0.4], DRIFT=[5.0, 10.0, 20.0])
AREA_STEP = 4  # zone area measured on every 4th RGB pixel


def annotations(scene):
    """Certain boxes per frame, uncertain boxes per frame, frames whose every row is a keyframe."""
    certain, uncertain, rows, keys = defaultdict(list), defaultdict(list), defaultdict(int), defaultdict(int)
    with open(f"{DATA_DIR}/{scene}_annotation.csv", newline="") as f:
        for r in csv.DictReader(f):
            fr = int(r["frame_index"])
            x, y, w, h = (float(r[k]) for k in ("left", "top", "width", "height"))
            (certain if r["confidence"] == "certain" else uncertain)[fr].append([x, y, x + w, y + h, 1.0])
            rows[fr] += 1
            keys[fr] += r["is_keyframe"] == "True"
    return certain, uncertain, {fr for fr in rows if keys[fr] == rows[fr]}


def score(ann, boxes, n_video, frames=None):
    """F1, found, right in % at the best cut-off (circle rule), like step19.best_point; optional frame subset."""
    certain, uncertain, _ = ann
    by_frame = defaultdict(list)
    for row in boxes:
        by_frame[int(row[0])].append(row[1:6])
    gts, preds = {}, {}
    for fr in (range(n_video) if frames is None else frames):
        gts[f"x_frame_{fr}"] = np.array(certain.get(fr, []), float).reshape(-1, 5)
        p = np.array(by_frame.get(fr, []), float).reshape(-1, 5)
        if fr in uncertain and len(p):
            c, u = np.c_[(p[:, 0] + p[:, 2]) / 2, (p[:, 1] + p[:, 3]) / 2], np.array(uncertain[fr])
            p = p[~((c[:, None, 0] >= u[None, :, 0]) & (c[:, None, 0] <= u[None, :, 2]) &
                    (c[:, None, 1] >= u[None, :, 1]) & (c[:, None, 1] <= u[None, :, 3])).any(1)]
        preds[f"x_frame_{fr}"] = p
    _, per = evaluate(preds, gts)
    conf = np.concatenate([per[s]["circle"]["pred"][:, 4] for s in per])
    hit = np.concatenate([per[s]["circle"]["tp"] for s in per]).astype(bool)
    n_gt = sum(len(g) for g in gts.values())
    f1, thr = best_f1(conf, hit, n_gt)
    tp, n = hit[conf >= thr].sum(), np.sum(conf >= thr)
    return 100 * f1, 100 * tp / max(n_gt, 1), 100 * tp / max(n, 1)


def run(det, times, start, noise, s):
    return s19.track(det, times, start, s["CONTINUE"] * start, noise, min_hits=s["MIN_HITS"],
                     max_gap=s["MAX_GAP"], drift=s["DRIFT"])[0]


def zone_check(scenes_boxes, cuts):
    """Per scene and box source: share of RGB labels inside a zone and mean zone area (both in %)."""
    rect = setup()
    optics = rect["KR"][1, 1] / np.load("data/output/stereo_calibration.npz")["KL"][1, 1]
    sy, sx = np.mgrid[0:RGB_H:AREA_STEP, 0:RGB_W:AREA_STEP]
    area_pts = raw_to_rect(rect, np.c_[sx.ravel(), sy.ravel()])
    human = {s: load_dvs(s) for s in ALL_SCENES}
    rgb = {s: s6.load_rgb() if s == s6.SCENE else load_rgb(s) for s in ALL_SCENES}
    coef = {s: np.array(cuts[s]["coef_rect_px"]) for s in ALL_SCENES}
    stats = {s: scene_stats(rect, human[s], rgb[s], coef[s]) for s in ALL_SCENES}
    out = {}
    for scene in ALL_SCENES:
        others = [s for s in ALL_SCENES if s != scene]
        zone = Zone(rect, *sigma_from(np.concatenate([stats[s][0] for s in others]),
                                      np.concatenate([stats[s][1] for s in others])), optics)
        if cuts[scene]["negative_disparity_share"] > NEG_SHARE_MAX:
            p5, _, p95 = cuts[scene]["disparity_p5_p50_p95"]
            zone.d_lo, zone.d_hi = p5 - DISP_MARGIN, p95 + DISP_MARGIN
        else:
            zone.d_lo, zone.d_hi = 0.0, rect["fB_mm"] / Z_NEAR
        for name, recs in scenes_boxes[scene].items():
            covered, areas, n_zones = [], [], 0
            for f in sorted(f for f in rgb[scene] if rgb[scene][f]):
                r = recs.get(f + K, []) + recs.get(f, [])
                n_zones += len(r)
                labels = rgb[scene][f]
                px = [[int(np.clip(v, 0, lim - 1)) for v, lim in zip(center(b), (RGB_W, RGB_H))] for b in labels]
                pts = raw_to_rect(rect, np.array(px, float))  # label centre pixel, as in step 17
                covered.extend(zone.weights(pts, r, coef[scene], True) > 0.5)
                areas.append(np.mean(zone.weights(area_pts, r, coef[scene], True) > 0.5))
            out[(scene, name)] = (100 * np.mean(covered), 100 * np.mean(areas), n_zones)
    return out


def main():
    os.makedirs(f"{OUT_DIR}/tracks", exist_ok=True)
    times, cuts = np.load(s19.TIMES), json.load(open(s19.CUTS))
    det = {s: detections(s) for s in ALL_SCENES}
    ann = {s: annotations(s) for s in ALL_SCENES}
    err = {s: s19.centre_error(s, det[s][0], cuts[s]["confidence_cutoff"]) for s in ALL_SCENES}
    noise = {s: s19.robust_std(np.concatenate([err[o] for o in ALL_SCENES if o != s])) for s in ALL_SCENES}
    print("centre error used (other five scenes): " + ", ".join(f"{s} {noise[s]:.3f}" for s in ALL_SCENES))

    # 1. every combination on every scene
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    f1 = np.zeros((len(combos), len(ALL_SCENES)))
    print(f"\n1. {len(combos)} combinations x {len(ALL_SCENES)} scenes", flush=True)
    for i, c in enumerate(combos):
        for j, s in enumerate(ALL_SCENES):
            tr = run(det[s][0], times[s], cuts[s]["confidence_cutoff"], noise[s], c)
            f1[i, j] = score(ann[s], tr[:, :6], det[s][1])[0]
        if (i + 1) % 12 == 0:
            print(f"   {i + 1} done", flush=True)
    with open(f"{OUT_DIR}/search.json", "w") as fh:
        json.dump(dict(scenes=ALL_SCENES, combos=combos, f1=f1.round(2).tolist()), fh, indent=1)

    chosen = {}
    print("\nchosen on the other five scenes (mean F1 there):")
    for j, s in enumerate(ALL_SCENES):
        i = int(np.delete(f1, j, axis=1).mean(1).argmax())
        chosen[s] = combos[i]
        print(f"   {s}: {combos[i]}  ({np.delete(f1, j, axis=1)[i].mean():.1f})")

    # 2. detector vs tracker on the left-out scene
    boxes, recs = {}, {}
    print("\n2a/2b. DVS F1 / found / right in % at the best cut-off; all certain annotations, and hand-placed frames only")
    print(f"{'scene':8s} | {'all: detector':>17s} | {'all: tracker':>17s} | {'keyframes: detector':>19s} | "
          f"{'keyframes: tracker':>18s} | frames")
    for s in ALL_SCENES:
        d, n_video = det[s]
        start = cuts[s]["confidence_cutoff"]
        tr = run(d, times[s], start, noise[s], chosen[s])
        np.savez_compressed(f"{OUT_DIR}/tracks/{s}.npz", tracks=tr)
        boxes[s] = tr
        kf = sorted(f for f in ann[s][2] if f < n_video)
        cells = [score(ann[s], b, n_video, fr) for fr in (None, kf) for b in (d, tr[:, :6])]
        print(f"{s:8s} | " + " | ".join(f"{a:5.1f} {b:4.0f}% {c:4.0f}%" for a, b, c in cells) + f" | {len(kf)}")
        recs[s] = {"detector": records(d, start), "tracker": records(tr[:, :6], 0.0)}
        s19.OUT_DIR = OUT_DIR
        s19.images(s, tr, times[s])

    print("\n2c. RGB labels inside a zone (weight > 0.5) / mean zone area, labelled RGB frames, zones from f-1 and f")
    zc = zone_check(recs, cuts)
    print(f"{'scene':8s} | {'detector: in zone, area, zones':>30s} | {'tracker: in zone, area, zones':>29s}")
    for s in ALL_SCENES:
        print(f"{s:8s} | " + " | ".join(f"{zc[(s, n)][0]:9.0f}% {zc[(s, n)][1]:6.2f}% {zc[(s, n)][2]:6d}"
                                         for n in ("detector", "tracker")))
    print("\nfound = of 100 certain DVS annotations, how many got a box; right = of 100 boxes, how many are on an "
          "annotation; F1 = one number for both. The cut-off is chosen on the scored scene itself (as in step 12b "
          "and the paper), so all three are optimistic, for detector and tracker alike.")


if __name__ == "__main__":
    main()
