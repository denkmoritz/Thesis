"""
Step 14 of the zone rebuild: zone v3 for all six scenes, and the zone+motion
detector on it. No RGB label of the tested scene is used anywhere.

What changed against zone v2 (step 7):
  row correction  label-free (step 13), instead of a fit on 60 % of the labels
  zone width      sigma per image row from the leftover row errors of the OTHER
                  five scenes (their labels, their label-free correction)
  box size        DVS box x the RGB/DVS size ratio measured in the OTHER scenes
  along the line  0.2 m to infinity; if more than 5 % of the label-free matches
                  have a negative disparity (scene 3), the range of disparities
                  of those matches (5th-95th percentile +- 20 px) is used instead
  test frames     ALL labelled frames of the scene (v2: only the last 40 %)
Unchanged: DVS side = human annotation of frame f-1 (certain), flat band of
DVS box height x 1.37 across the line, Gaussian fall-off outside.

Detector (step 11e): the three strongest motion peaks (background model) in each
DVS insect's zone (weight > 0.5); confidence = peak strength.

Ignore regions (written as PNG, 1/4 size): the zone of every DVS insect that has
no RGB label in it (insect not visible / not labelled in RGB), and the privacy
masks of scene 6. Scores are printed without and with them.

Printed per scene: share of label centres inside the zone, zone area, AP (circle
and IoU rule), best F1; and for scenes 1-5 on the last 40 % the AP next to the
same detector on the v2 zone (label-fitted correction, step 11e).

Run: PYTHONPATH=src ./venv/bin/python src/dti/geometry/zone_v3.py
"""

import glob
import json
import os

import cv2
import numpy as np

from dti.io.data_sync import DATA_DIR, DATASET_DIR, SCENES, RGB_W, RGB_H, load_rgb, center
from dti.geometry.rectify import load_dvs, setup
from dti.geometry import scene_correction as s4
from dti.geometry.zone import Zone, ROW_BIN, MIN_BIN_N, SIGMA_FLOOR, raw_to_rect
from dti.io import scene6 as s6
from dti.evaluation.metrics import load_dir, evaluate, render
from dti.motion.cue import DATASET, read_frames, motion_maps, peaks
from dti.motion.detector import refine, dedupe, N_ZONE_PEAKS

OUT_DIR = "outputs/step14"
CORR = "outputs/step13/labelfree_corrections.json"
V2_PRED = "outputs/step11e/zone_motion_median/labels"
ALL_SCENES = SCENES + [s6.SCENE]
K = s4.K
Z_NEAR = 200.0  # mm
NEG_SHARE_MAX = 0.05
DISP_MARGIN = 20.0  # rect px


def scene_stats(rect, dvs, rgb, coef):
    """Leftover row error (rect row, RGB px) and RGB/DVS height ratios of one scene, with its label-free correction."""
    frames = sorted(f for f in rgb if rgb[f])
    xy, off = s4.collect(rect, dvs, rgb, frames, coef)
    res = (off - s4.predict(coef, xy)) * rect["to_rgb"]
    ratios = [rgb[f][0][3] / dvs[f + K][0]["box"][3] for f in frames
              if len(rgb[f]) == 1 and len(dvs.get(f + K, [])) == 1]
    return xy[:, 1], res, ratios


def sigma_from(rows, res):
    edges = np.arange(0, 1200 + ROW_BIN, ROW_BIN)
    centres, vals = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (rows >= lo) & (rows < hi)
        if sel.sum() >= MIN_BIN_N:
            centres.append((lo + hi) / 2)
            vals.append(max(SIGMA_FLOOR, float(np.percentile(np.abs(res[sel]), 68))))
    return np.array(centres), np.array(vals)


def image_path(scene, f):
    if scene == s6.SCENE:
        return f"{os.path.dirname(os.path.dirname(s6.LABEL_DIR))}/images/train/frame_{f:06d}.jpg"
    return glob.glob(f"{DATASET_DIR}/images/*/{scene}_frame_{f:06d}.jpg")[0]


def main():
    for sub in ("zone_motion/labels", "ignore", "_img"):
        os.makedirs(f"{OUT_DIR}/{sub}", exist_ok=True)
    rect = setup()
    corr = json.load(open(CORR))
    optics = rect["KR"][1, 1] / np.load("data/output/stereo_calibration.npz")["KL"][1, 1]
    fy, fx = np.mgrid[0:RGB_H, 0:RGB_W]
    grid = raw_to_rect(rect, np.c_[fx.ravel(), fy.ravel()])
    data = {s: (load_dvs(s), s6.load_rgb() if s == s6.SCENE else load_rgb(s)) for s in ALL_SCENES}
    stats = {s: scene_stats(rect, *data[s], np.array(corr[s]["coef_rect_px"])) for s in ALL_SCENES}
    masks6 = s6.load_masks("rgb")

    gts, info = {}, {}
    for scene in ALL_SCENES:
        dvs, rgb = data[scene]
        coef = np.array(corr[scene]["coef_rect_px"])
        others = [s for s in ALL_SCENES if s != scene]
        rows, vals = sigma_from(np.concatenate([stats[s][0] for s in others]),
                                np.concatenate([stats[s][1] for s in others]))
        ratio = float(np.median(np.concatenate([stats[s][2] for s in others])))
        zone = Zone(rect, rows, vals, optics)
        shifted = corr[scene]["negative_disparity_share"] > NEG_SHARE_MAX
        if shifted:
            p5, _, p95 = corr[scene]["disparity_p5_p50_p95"]
            zone.d_lo, zone.d_hi = p5 - DISP_MARGIN, p95 + DISP_MARGIN
        else:
            zone.d_lo, zone.d_hi = 0.0, rect["fB_mm"] / Z_NEAR
        print(f"{scene}: box size = DVS box x {ratio:.2f}; sigma (RGB px) at rows {rows.astype(int).tolist()}: "
              f"{np.round(vals, 1).tolist()}; along the line: "
              + (f"disparity {zone.d_lo:+.0f}..{zone.d_hi:+.0f} rect px (sign check failed, "
                 f"{corr[scene]['negative_disparity_share']:.0%} negative)" if shifted else "0.2 m .. infinity"),
              flush=True)

        cap = cv2.VideoCapture(f"{DATA_DIR}/{scene}_rgb.mp4")
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        in_zone, areas, n_ign, n_dvs = [], [], 0, 0
        for f in sorted(f for f in rgb if rgb[f]):
            stem = f"{scene}_frame_{f:06d}"
            boxes = rgb[f]
            gts[stem] = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3], 1.0] for b in boxes])
            m = motion_maps(read_frames(cap, f, n_video))["median"]
            ignore = np.zeros((RGB_H, RGB_W), np.uint8)
            if scene == s6.SCENE:
                for mk in masks6.values():
                    if any(g in mk["frames"] for g in range(f - 10, f + 11)):
                        y0, y1, x0, x1 = max(0, mk["top"] - 10), mk["bottom"] + 10, max(0, mk["left"] - 10), mk["right"] + 10
                        m[y0:y1, x0:x1] = 0
                        ignore[y0:y1, x0:x1] = 255
            recs = dvs.get(f + K, [])
            cx = np.clip(np.array([center(b)[0] for b in boxes]).astype(int), 0, RGB_W - 1)
            cy = np.clip(np.array([center(b)[1] for b in boxes]).astype(int), 0, RGB_H - 1)
            covered, union, preds = np.zeros(len(boxes), bool), np.zeros((RGB_H, RGB_W), bool), []
            for r in recs:
                piece = zone.weights(grid, [r], coef, True).reshape(RGB_H, RGB_W) > 0.5
                hit = piece[cy, cx]
                covered |= hit
                union |= piece
                if not hit.any():  # DVS insect without an RGB label in its zone
                    ignore[piece] = 255
                    n_ign += 1
                w, h = r["box"][2] * ratio, r["box"][3] * ratio
                for x, y in peaks(m, piece, N_ZONE_PEAKS):
                    if m[y, x] > 0:
                        preds.append((*refine(m, x, y, max(w, h, 8)), w, h, m[y, x] / 255))
            n_dvs += len(recs)
            in_zone += list(covered)
            areas.append(union.mean())
            with open(f"{OUT_DIR}/zone_motion/labels/{stem}.txt", "w") as fh:
                for px, py, w, h, conf in dedupe(preds):
                    fh.write(f"0 {px / RGB_W:.6f} {py / RGB_H:.6f} {w / RGB_W:.6f} {h / RGB_H:.6f} {conf:.6f}\n")
            cv2.imwrite(f"{OUT_DIR}/ignore/{stem}.png", cv2.resize(ignore, (RGB_W // 4, RGB_H // 4),
                                                                   interpolation=cv2.INTER_NEAREST))
        cap.release()
        info[scene] = (float(np.mean(in_zone)), float(np.mean(areas)), n_dvs, n_ign)

    preds = load_dir(f"{OUT_DIR}/zone_motion/labels")
    v2 = load_dir(V2_PRED)
    print(f"\n{'scene':8s} {'frames':>6s} {'labels':>6s} {'DVS':>5s} {'no label':>8s} | {'label in zone':>13s} {'zone area':>9s} | "
          f"{'AP circle':>9s} {'with ignore':>11s} {'AP IoU':>7s} {'with ignore':>11s} {'best F1':>7s} {'with ignore':>11s} | "
          f"{'last 40 %: AP circle':>20s} {'v2 zone':>8s}")
    for scene in ALL_SCENES:
        gt = {s: b for s, b in gts.items() if s.startswith(scene + "_frame_")}
        pr = {s: b for s, b in preds.items() if s in gt}
        a = evaluate(pr, gt)[0]["all"]
        res, per_image = evaluate(pr, gt, ignore_dir=f"{OUT_DIR}/ignore")
        b = res["all"]
        last = {s: g for s, g in gt.items() if os.path.exists(f"{DATASET}/images/val/{s}.jpg")}
        tail = f"{'n/a':>20s} {'n/a':>8s}"
        if last:
            c = evaluate({s: p for s, p in pr.items() if s in last}, last)[0]["all"]
            d = evaluate({s: p for s, p in v2.items() if s in last}, last)[0]["all"]
            tail = f"{c['ap_circle']:20.3f} {d['ap_circle']:8.3f}"
        cov, area, n_dvs, n_ign = info[scene]
        print(f"{scene:8s} {len(gt):6d} {a['labels']:6d} {n_dvs:5d} {n_ign:8d} | {cov:13.0%} {area:9.1%} | "
              f"{a['ap_circle']:9.3f} {b['ap_circle']:11.3f} {a['ap_iou']:7.3f} {b['ap_iou']:11.3f} {a['f1_best']:7.2f} "
              f"{b['f1_best']:11.2f} | {tail}")
        busiest = max(per_image, key=lambda s: (len(per_image[s]["gt"]), s))
        link = f"{OUT_DIR}/_img/{busiest}.jpg"
        if not os.path.exists(link):
            os.symlink(image_path(scene, int(busiest[-6:])), link)
        render(busiest, per_image[busiest], f"{OUT_DIR}/_img", OUT_DIR, b["thr_best"])
    print("\nDVS = certain DVS boxes at f-1 on the labelled frames; no label = of these, zones without an RGB label inside")
    print("(ignore regions); label in zone = RGB label centres with zone weight > 0.5; best F1: threshold chosen on the")
    print("test scene (optimistic); v2 zone = same detector on the zone with the label-fitted correction (step 11e).")
    print(f"predictions: {OUT_DIR}/zone_motion/labels, ignore regions: {OUT_DIR}/ignore, one image per scene: {OUT_DIR}/")


if __name__ == "__main__":
    main()
