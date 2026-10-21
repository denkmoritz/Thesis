"""
Step 18 of the zone rebuild: DVS-only against fusion, on the same ground truth.

Research question 2: how much does fusion improve over the DVS-only detector?
Both are scored on the same frames (all labelled RGB frames) against the same
labels (the human RGB labels), with the same DVS detections (our DVS detector,
frames f-1 and f, every detection kept so the cut-off can be swept).

What DVS-only can say about an insect in the RGB image, without using RGB:
  DVS line    the insect lies on the line (zone) of a DVS detection; where along
              the line is unknown. Counted as found if the RGB label centre lies
              inside the detection's zone. This is the most DVS-only can claim,
              so it is an upper bound for DVS-only, not a box.
  DVS box     a box anyway: placed on the line at the typical disparity of the
              scene (median of the label-free matches of step 17, no labels), size
              DVS box x the RGB/DVS size ratio of the other scenes.
Fusion (from steps 17 / m3d.slurm): A and B2 trained on detector zones, and the
zone + motion rule without training.

A DVS detection that finds no RGB label is a false alarm, unless a human DVS
annotation confirms an insect there (then it is ignored: an insect that is not
labelled in RGB) -- the counterpart of the ignore regions used for fusion.

Printed per scene: found / right in % at the best cut-off, and AP in brackets.

Run: PYTHONPATH=src ./venv/bin/python src/dti/evaluation/dvs_vs_fusion.py
"""

import json
from collections import defaultdict

import numpy as np

from dti.io.data_sync import RGB_W, RGB_H, load_rgb
from dti.geometry.rectify import load_dvs, setup, rect_dvs
from dti.geometry import scene_correction as s4
from dti.geometry.epiline import rect_to_raw
from dti.geometry.zone import Zone, raw_to_rect
from dti.io import scene6 as s6
from dti.evaluation.metrics import load_dir, evaluate, average_precision, score_matrix
from dti.dvs.score import DVS_W, DVS_H
from dti.geometry.zone_v3 import ALL_SCENES, Z_NEAR, NEG_SHARE_MAX, DISP_MARGIN, scene_stats, sigma_from
from dti.datasets.auto_chain import DET, OUT_DIR as S17
from dti.evaluation.score_auto_chain import found_right
from dti.paths import TRAINING_ROOT

GT = f"{TRAINING_ROOT}/dataset_m3d/A/labels/all"
IGNORE = "outputs/step14/ignore"
FUSION = [("A fusion", "outputs/m3d/m3d_{s}_A/labels"), ("B2 fusion", "outputs/m3d/m3d_{s}_B2/labels"),
          ("no training", f"{S17}/zone_motion/labels")]
K = s4.K


def frame_detections(scene):
    d = np.load(f"{DET}/{scene}.npz")["det"]
    out = defaultdict(list)
    for row in d:
        x1, y1, x2, y2 = (float(np.clip(v, 0, m)) for v, m in zip(row[1:5], (DVS_W, DVS_H, DVS_W, DVS_H)))
        out[int(row[0])].append(dict(box=(x1, y1, x2 - x1, y2 - y1), c=((x1 + x2) / 2, (y1 + y2) / 2),
                                     conf=float(row[5])))
    return out


def merged(dets, f):
    """Detections of DVS frames f-1 and f; the same insect seen in both is kept once (stronger one)."""
    keep = []
    for r in sorted(dets.get(f + K, []) + dets.get(f, []), key=lambda r: -r["conf"]):
        if all(np.hypot(r["c"][0] - q["c"][0], r["c"][1] - q["c"][1]) > 10 for q in keep):
            keep.append(r)
    return keep


def confirmed_by_human(r, human_recs):
    """True if a certain human DVS box (frame f-1 or f) has its circle around this detection's centre."""
    for h in human_recs:
        x, y, w, hh = h["box"]
        if np.hypot(r["c"][0] - (x + w / 2), r["c"][1] - (y + hh / 2)) <= np.hypot(w, hh) / 2:
            return True
    return False


def main():
    rect = setup()
    corr = json.load(open(f"{S17}/auto_corrections.json"))
    optics = rect["KR"][1, 1] / np.load("data/output/stereo_calibration.npz")["KL"][1, 1]
    human = {s: load_dvs(s) for s in ALL_SCENES}
    rgb = {s: s6.load_rgb() if s == s6.SCENE else load_rgb(s) for s in ALL_SCENES}
    stats = {s: scene_stats(rect, human[s], rgb[s], np.array(corr[s]["coef_rect_px"])) for s in ALL_SCENES}
    gts = load_dir(GT)
    rows = {}
    for scene in ALL_SCENES:
        coef = np.array(corr[scene]["coef_rect_px"])
        a, b, c = coef
        others = [s for s in ALL_SCENES if s != scene]
        zone = Zone(rect, *sigma_from(np.concatenate([stats[s][0] for s in others]),
                                      np.concatenate([stats[s][1] for s in others])), optics)
        if corr[scene]["negative_disparity_share"] > NEG_SHARE_MAX:
            p5, _, p95 = corr[scene]["disparity_p5_p50_p95"]
            zone.d_lo, zone.d_hi = p5 - DISP_MARGIN, p95 + DISP_MARGIN
        else:
            zone.d_lo, zone.d_hi = 0.0, rect["fB_mm"] / Z_NEAR
        d_typ = corr[scene]["disparity_p5_p50_p95"][1]
        ratio = float(np.median(np.concatenate([stats[s][2] for s in others])))
        dets = frame_detections(scene)
        gt = {k: v for k, v in gts.items() if k.startswith(scene + "_frame_")}
        n_gt = sum(len(v) for v in gt.values())

        line_conf, line_tp, box_preds = [], [], {}
        for stem, g in gt.items():
            f = int(stem[-6:])
            recs = merged(dets, f)
            hum = human[scene].get(f + K, []) + human[scene].get(f, [])
            centres = np.c_[(g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2]
            rect_pts = raw_to_rect(rect, centres)
            free = np.ones(len(g), bool)
            boxes = []
            for r in recs:  # highest confidence first
                inside = zone.weights(rect_pts, [r], coef, True) > 0.5
                cand = inside & free
                if cand.any():
                    free[int(np.argmax(cand))] = False
                    line_conf.append(r["conf"])
                    line_tp.append(True)
                elif not confirmed_by_human(r, hum):
                    line_conf.append(r["conf"])
                    line_tp.append(False)
                xd, yd = rect_dvs(rect, [r["c"]])[0]
                x = xd - d_typ
                raw = rect_to_raw(rect, np.array([[x, (yd + a + b * x) / (1 - c)]]))
                if len(raw):
                    cx, cy = raw[0]
                    w, h = r["box"][2] * ratio, r["box"][3] * ratio
                    if 0 <= cx < RGB_W and 0 <= cy < RGB_H:
                        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, r["conf"]])
            box_preds[stem] = np.array(boxes, float).reshape(-1, 5)

        line_conf, line_tp = np.array(line_conf), np.array(line_tp, bool)
        o = np.argsort(-line_conf)
        t = np.cumsum(line_tp[o])
        n = np.arange(1, len(t) + 1)
        i = int((2 * t / (n + n_gt)).argmax())
        cells = {"DVS line": {"circle": (t[i] / n_gt, t[i] / n[i], average_precision(line_conf, line_tp, n_gt))}}
        res, per = evaluate(box_preds, gt, IGNORE)
        cells["DVS box"] = {r_: (*found_right(per, r_, n_gt), res["all"]["ap_" + r_]) for r_ in ("circle", "iou")}
        for name, path in FUSION:
            pr = {k: v for k, v in load_dir(path.format(s=scene)).items() if k in gt}
            res, per = evaluate(pr, gt, IGNORE)
            cells[name] = {r_: (*found_right(per, r_, n_gt), res["all"]["ap_" + r_]) for r_ in ("circle", "iou")}
        rows[scene] = (n_gt, cells, d_typ, ratio)

    names = ["DVS line", "DVS box", "A fusion", "B2 fusion", "no training"]
    for rule, title in (("circle", "position only (circle rule)"), ("iou", "position and size (IoU >= 0.5)")):
        print(f"\n{title}: found / right in % at the best cut-off, AP in brackets")
        print(f"{'scene':8s} {'labels':>6s} " + " ".join(f"{n:>18s}" for n in names))
        for scene, (n_gt, cells, _, _) in rows.items():
            out = []
            for n in names:
                v = cells[n].get(rule)
                out.append(f"{'-':>18s}" if v is None else f"{v[0]:4.0%} / {v[1]:4.0%} ({v[2]:.2f})")
            print(f"{scene:8s} {n_gt:6d} " + " ".join(f"{x:>18s}" for x in out))
    print("\nDVS line = label lies in the zone of a DVS detection (no position along the line: upper bound for "
          "DVS-only);\nDVS box = DVS detection placed at the typical disparity of the scene:")
    for scene, (_, _, d_typ, ratio) in rows.items():
        print(f"  {scene}: typical disparity {d_typ:+.0f} rect px, box = DVS box x {ratio:.2f}")


if __name__ == "__main__":
    main()
