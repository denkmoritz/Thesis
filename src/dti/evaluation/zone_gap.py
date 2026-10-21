"""
Step 21 of the zone rebuild: why do zones from our DVS detector cover fewer RGB
labels than zones from the human DVS annotation?

Rule fixed before running (2026-10-20):
  population  RGB labels whose centre lies outside the detector zone (weight <= 0.5);
              detector zone as in step 17 (detections above the scene's cut-off at
              DVS frames f-1 and f, correction from detector boxes)
  matching    the human DVS box (certain, frame f-1) whose human zone (step 14) holds
              the label centre with the highest weight; if no human zone holds it
              (weight <= 0.5): "no DVS counterpart"
  outcome, checked in this order:
    recall     no detection (above the cut-off, frame f-1 or f) has its centre within
               the circle of the matched human box (radius = half its diagonal, the
               circle rule of the scoring)
    size       a detection is there, and a zone built from the detection's centre with
               the human box's size holds the label
    placement  everything else: a detection is there, but its centre or the
               detector-based correction puts the zone off the label

Printed per scene: labels, share inside the human zone and inside the detector zone,
and of the labels outside the detector zone how many fall in each outcome.

Run: PYTHONPATH=src ./venv/bin/python src/dti/evaluation/zone_gap.py
"""

import json

import numpy as np

from dti.io import scene6 as s6
from dti.io.data_sync import RGB_W, RGB_H, load_rgb, center
from dti.geometry.rectify import load_dvs, setup
from dti.geometry.zone import Zone, raw_to_rect
from dti.geometry.zone_v3 import (ALL_SCENES, CORR, Z_NEAR, NEG_SHARE_MAX, DISP_MARGIN, scene_stats,
                                         sigma_from)
from dti.datasets.auto_chain import detections, records, K
from dti.tracking.tracker import CUTS

OUTCOMES = ("recall", "size", "placement", "no DVS counterpart")


def make_zone(rect, optics, scene, stats, corr):
    others = [s for s in ALL_SCENES if s != scene]
    zone = Zone(rect, *sigma_from(np.concatenate([stats[s][0] for s in others]),
                                  np.concatenate([stats[s][1] for s in others])), optics)
    if corr[scene]["negative_disparity_share"] > NEG_SHARE_MAX:
        p5, _, p95 = corr[scene]["disparity_p5_p50_p95"]
        zone.d_lo, zone.d_hi = p5 - DISP_MARGIN, p95 + DISP_MARGIN
    else:
        zone.d_lo, zone.d_hi = 0.0, rect["fB_mm"] / Z_NEAR
    return zone


def main():
    rect = setup()
    optics = rect["KR"][1, 1] / np.load("data/output/stereo_calibration.npz")["KL"][1, 1]
    corr_h, corr_d = json.load(open(CORR)), json.load(open(CUTS))
    human = {s: load_dvs(s) for s in ALL_SCENES}
    rgb = {s: s6.load_rgb() if s == s6.SCENE else load_rgb(s) for s in ALL_SCENES}
    coef_h = {s: np.array(corr_h[s]["coef_rect_px"]) for s in ALL_SCENES}
    coef_d = {s: np.array(corr_d[s]["coef_rect_px"]) for s in ALL_SCENES}
    stats_h = {s: scene_stats(rect, human[s], rgb[s], coef_h[s]) for s in ALL_SCENES}
    stats_d = {s: scene_stats(rect, human[s], rgb[s], coef_d[s]) for s in ALL_SCENES}

    print(f"{'scene':8s} {'labels':>6s} {'in human zone':>13s} {'in detector zone':>16s} {'outside':>7s} | "
          + " ".join(f"{o:>18s}" for o in OUTCOMES))
    total = dict.fromkeys(OUTCOMES, 0)
    for scene in ALL_SCENES:
        zh = make_zone(rect, optics, scene, stats_h, corr_h)
        zd = make_zone(rect, optics, scene, stats_d, corr_d)
        auto = records(detections(scene)[0], corr_d[scene]["confidence_cutoff"])
        n_lab = in_h = in_d = 0
        count = dict.fromkeys(OUTCOMES, 0)
        for f in sorted(f for f in rgb[scene] if rgb[scene][f]):
            dets = auto.get(f + K, []) + auto.get(f, [])
            hum = human[scene].get(f + K, [])
            for b in rgb[scene][f]:
                px = [int(np.clip(v, 0, lim - 1)) for v, lim in zip(center(b), (RGB_W, RGB_H))]
                pt = raw_to_rect(rect, np.array([px], float))
                wh = [zh.weights(pt, [r], coef_h[scene], True)[0] for r in hum]
                n_lab += 1
                in_h += bool(wh) and max(wh) > 0.5
                if zd.weights(pt, dets, coef_d[scene], True)[0] > 0.5:
                    in_d += 1
                    continue
                if not wh or max(wh) <= 0.5:
                    count["no DVS counterpart"] += 1
                    continue
                hx, hy, hw, hh = hum[int(np.argmax(wh))]["box"]
                hc, radius = (hx + hw / 2, hy + hh / 2), np.hypot(hw, hh) / 2
                near = [d for d in dets if np.hypot(d["c"][0] - hc[0], d["c"][1] - hc[1]) <= radius]
                if not near:
                    count["recall"] += 1
                    continue
                resized = [dict(d, box=(d["c"][0] - hw / 2, d["c"][1] - hh / 2, hw, hh)) for d in near]
                count["size" if zd.weights(pt, resized, coef_d[scene], True)[0] > 0.5 else "placement"] += 1
        out = n_lab - in_d
        print(f"{scene:8s} {n_lab:6d} {100 * in_h / n_lab:12.0f}% {100 * in_d / n_lab:15.0f}% {out:7d} | "
              + " ".join(f"{count[o]:7d} ({100 * count[o] / max(out, 1):3.0f} %)".rjust(18) for o in OUTCOMES),
              flush=True)
        for o in OUTCOMES:
            total[o] += count[o]
    n_out = sum(total.values())
    print(f"{'all':8s} {'':>6s} {'':>13s} {'':>16s} {n_out:7d} | "
          + " ".join(f"{total[o]:7d} ({100 * total[o] / n_out:3.0f} %)".rjust(18) for o in OUTCOMES))
    print("\noutside = labels outside the detector zone. recall = the detector had no box on that insect; size = it "
          "had one, but the zone was too narrow because the detector box was smaller than the human box; placement = "
          "it had one, but the zone lies off the label (box centre or correction).")


if __name__ == "__main__":
    main()
