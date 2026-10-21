"""
Step 7 of the zone rebuild: the zone map.

For each DVS insect (annotation of frame f-1) the zone in RGB frame f is

    w(p) = across(p) * along(p)

  across: flat band of weight 1 as tall as the insect (DVS box height converted
          to RGB px, h), then a Gaussian fall-off outside it:
          exp(-max(|dist| - h/2, 0)^2 / (2 sigma(row)^2)); sigma = geometric
          uncertainty, grows toward the top/bottom of the image (option 1),
          estimated per image row from the leftover row errors of the FIT part
          (first 60% of every scene, all scenes pooled)
  along:  1 on the depth segment 0.2 .. 3.5 m (disparity f*B/Z), Gaussian
          fall-off with the same sigma outside it; scene 3: 1 along the whole
          line (its sideways shift cannot be measured, step 6)

Several DVS insects: zone = maximum over them. Everything is computed in
rectified coordinates (line = row there) and sampled at raw RGB pixels.

Evaluated on the TEST part only (last 40% of every scene):
  - zone weight at each RGB label centre (no matching needed: max over insects)
  - box coverage: share of the RGB label box with zone weight > 0.5
  - height check: share of labels whose box height fits inside the flat band
  - share of the image with weight > 0.5, mean weight over the image
  - lift = mean weight at labels / mean weight over the image

Run: PYTHONPATH=src ./venv/bin/python src/dti/geometry/zone.py
"""

import json
import os

import cv2
import numpy as np

from dti.io.data_sync import SCENES, load_rgb, read_frame, center
from dti.geometry.rectify import load_dvs, setup, rect_dvs, rect_rgb
from dti.geometry.scene_correction import K, collect, predict
from dti.geometry.epiline import CORR, RGB_W, RGB_H

OUT_DIR = "outputs/step7"
Z_NEAR, Z_FAR = 200.0, 3500.0  # mm
NO_DEPTH_LIMIT = {"3_m-h-h"}
ROW_BIN = 150  # rectified px
MIN_BIN_N = 15
SIGMA_FLOOR = 4.0  # RGB px
GRID_STEP = 4  # raw px, for image-wide statistics
WHITE, GREEN = (255, 255, 255), (0, 255, 0)


class Zone:
    def __init__(self, rect, sigma_rows, sigma_vals, h_scale=None):
        self.rect = rect
        self.h_scale = h_scale  # RGB px per DVS px of box height; None = no flat band (old zone)
        self.to_rgb = rect["to_rgb"]
        self.sigma_rows, self.sigma_vals = sigma_rows, sigma_vals  # sigma in RGB px at rectified rows
        self.d_lo, self.d_hi = rect["fB_mm"] / Z_FAR, rect["fB_mm"] / Z_NEAR

    def sigma(self, rect_rows):
        return np.interp(rect_rows, self.sigma_rows, self.sigma_vals)

    def weights(self, rect_pts, dvs_recs, coef, depth_limit):
        """Zone weight at rectified RGB points (n, 2) for a list of DVS records."""
        w = np.zeros(len(rect_pts))
        if not dvs_recs:
            return w
        a, b, c = coef
        x, y = rect_pts[:, 0], rect_pts[:, 1]
        sig = self.sigma(y)
        for rec, (xd, yd) in zip(dvs_recs, rect_dvs(self.rect, [r["c"] for r in dvs_recs])):
            across = (y - (yd + a + b * x) / (1 - c)) * self.to_rgb  # RGB px (line is ~horizontal)
            half = 0.0 if self.h_scale is None else rec["box"][3] * self.h_scale / 2
            excess = np.maximum(np.abs(across) - half, 0)
            wa = np.exp(-excess ** 2 / (2 * sig ** 2))
            if depth_limit:
                d = xd - x
                outside = np.maximum(self.d_lo - d, 0) + np.maximum(d - self.d_hi, 0)
                wa *= np.exp(-(outside * self.to_rgb) ** 2 / (2 * sig ** 2))
            w = np.maximum(w, wa)
        return w


def raw_to_rect(rect, raw):
    return cv2.undistortPoints(np.float32(raw)[:, None], rect["KR"], rect["dR"], R=rect["R2"],
                               P=rect["P2"]).reshape(-1, 2)


def estimate_sigma(rect, corr):
    """Robust sigma (RGB px) of the corrected row error per rectified row bin, FIT part, all scenes pooled."""
    ys, res = [], []
    for scene in SCENES:
        dvs, rgb = load_dvs(scene), load_rgb(scene)
        c = corr[scene]
        coef = np.array(c["coef_rect_px"])
        fit_frames = sorted(f for f in rgb if rgb[f] and c["fit_frames"][0] <= f <= c["fit_frames"][1])
        xy, off = collect(rect, dvs, rgb, fit_frames, coef)
        ys.append(xy[:, 1])
        res.append((off - predict(coef, xy)) * rect["to_rgb"])
    ys, res = np.concatenate(ys), np.concatenate(res)
    edges = np.arange(0, 1200 + ROW_BIN, ROW_BIN)
    rows, vals = [], []
    print(f"sigma per rectified row bin (FIT part, all scenes, 68th percentile of |row error|, RGB px):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ys >= lo) & (ys < hi)
        if sel.sum() < MIN_BIN_N:
            print(f"  rows {lo:4d}-{hi:4d}: n={sel.sum():4d}  (too few, skipped)")
            continue
        s = max(SIGMA_FLOOR, float(np.percentile(np.abs(res[sel]), 68)))
        rows.append((lo + hi) / 2)
        vals.append(s)
        print(f"  rows {lo:4d}-{hi:4d}: n={sel.sum():4d}  sigma {s:5.1f}")
    return np.array(rows), np.array(vals)


def estimate_h_scale(rect, corr):
    """RGB px per DVS px of insect height: from optics, and from single-insect pairs of the FIT part."""
    optics = rect["KR"][1, 1] / np.load("data/output/stereo_calibration.npz")["KL"][1, 1]
    ratios = []
    for scene in SCENES:
        dvs, rgb = load_dvs(scene), load_rgb(scene)
        c = corr[scene]
        for f, labs in rgb.items():
            d = dvs.get(f + K, [])
            if c["fit_frames"][0] <= f <= c["fit_frames"][1] and len(labs) == 1 and len(d) == 1:
                ratios.append(labs[0][3] / d[0]["box"][3])
    r = np.array(ratios)
    print(f"insect height RGB/DVS: optics (focal ratio) {optics:.2f};  FIT part n={len(r)}: median {np.median(r):.2f}, "
          f"IQR {np.percentile(r, 25):.2f}..{np.percentile(r, 75):.2f}")
    return optics, float(np.median(r))


def box_samples(b, step=2.0):
    xs = np.arange(b[0], b[0] + b[2] + 1e-6, step)
    ys = np.arange(b[1], b[1] + b[3] + 1e-6, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.c_[gx.ravel(), gy.ravel()]


def render(zone, scene, f, dvs, rgb, coef, depth_limit, grid_rect_full):
    recs = dvs.get(f + K, [])
    w = zone.weights(grid_rect_full, recs, coef, depth_limit).reshape(RGB_H, RGB_W)
    r_img = read_frame(scene, "rgb", f)
    heat = cv2.applyColorMap(np.uint8(255 * w), cv2.COLORMAP_INFERNO)
    over = np.uint8(np.clip(r_img * (0.35 + 0.65 * (1 - w[..., None])) + heat * 0.65 * w[..., None], 0, 255))
    for b in rgb[f]:
        pad = max(0, (14 - min(b[2], b[3])) / 2)
        cv2.rectangle(over, (int(b[0] - pad), int(b[1] - pad)), (int(b[0] + b[2] + pad), int(b[1] + b[3] + pad)),
                      GREEN, 3)
    d_img = read_frame(scene, "dvs", f + K)
    for r in recs:
        x, y, bw, bh = r["box"]
        pad = max(0, (10 - min(bw, bh)) / 2)
        cv2.rectangle(d_img, (int(x - pad), int(y - pad)), (int(x + bw + pad), int(y + bh + pad)), (255, 255, 0), 2)
    s = RGB_H / 720
    d_img = cv2.resize(d_img, None, fx=s, fy=s)
    canvas = cv2.resize(np.hstack([d_img, np.full((RGB_H, 16, 3), 40, np.uint8), over]), None, fx=0.5, fy=0.5,
                        interpolation=cv2.INTER_AREA)
    labs = [float(v) for v in zone.weights(raw_to_rect(zone.rect, [center(b) for b in rgb[f]]), recs, coef,
                                           depth_limit)] if recs else []
    lines = [
        f"{scene}  RGB frame {f} (test part)   left: DVS frame {f + K} with its annotations (cyan, enlarged)   "
        "right: RGB frame with zone map",
        f"zone weight 0 (image unchanged) .. 1 (bright yellow), flat band = DVS box height x {zone.h_scale:.2f};  "
        "green box = RGB label;  "
        + ("depth segment 0.2 .. 3.5 m" if depth_limit else "no depth limit (scene 3)"),
        "zone weight at each RGB label: " + (", ".join(f"{v:.2f}" for v in labs) if labs else "n/a (no DVS insect)"),
    ]
    cap = np.zeros((22 * len(lines) + 10, canvas.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(cap, t, (8, 22 * i + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.imwrite(f"{OUT_DIR}/{scene}_frame_{f:06d}.jpg", np.vstack([canvas, cap]), [cv2.IMWRITE_JPEG_QUALITY, 90])


def evaluate(zone, rect, corr, grid_rect):
    print(f"{'scene':10s} {'labels':>6s} | {'median w':>8s} {'w>0.5':>6s} {'w>0.1':>6s} | "
          f"{'box cov.':>8s} {'cov>=90%':>8s} {'fits h':>6s} | {'area w>0.5':>10s} {'lift':>6s}")
    for scene in SCENES:
        dvs, rgb = load_dvs(scene), load_rgb(scene)
        c = corr[scene]
        coef = np.array(c["coef_rect_px"])
        depth_limit = scene not in NO_DEPTH_LIMIT
        test = sorted(f for f in rgb if rgb[f] and c["test_frames"][0] <= f <= c["test_frames"][1])
        w_lab, cov, fits, areas, means = [], [], [], [], []
        for f in test:
            recs = dvs.get(f + K, [])
            if not recs:
                continue
            w_lab.extend(zone.weights(raw_to_rect(rect, [center(b) for b in rgb[f]]), recs, coef, depth_limit))
            for b in rgb[f]:
                cov.append(np.mean(zone.weights(raw_to_rect(rect, box_samples(b)), recs, coef, depth_limit) > 0.5))
                band = 0 if zone.h_scale is None else max(r["box"][3] for r in recs) * zone.h_scale
                fits.append(b[3] <= band)
            wg = zone.weights(grid_rect, recs, coef, depth_limit)
            areas.append(np.mean(wg > 0.5))
            means.append(wg.mean())
        w_lab, cov = np.array(w_lab), np.array(cov)
        print(f"{scene:10s} {len(w_lab):6d} | {np.median(w_lab):8.2f} {np.mean(w_lab > 0.5):6.0%} "
              f"{np.mean(w_lab > 0.1):6.0%} | {np.median(cov):8.0%} {np.mean(cov >= 0.9):8.0%} {np.mean(fits):6.0%} | "
              f"{np.mean(areas):10.1%} {w_lab.mean() / np.mean(means):5.0f}x")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rect = setup()
    corr = json.load(open(CORR))
    rows, vals = estimate_sigma(rect, corr)
    optics, fitted = estimate_h_scale(rect, corr)
    gy, gx = np.mgrid[0:RGB_H:GRID_STEP, 0:RGB_W:GRID_STEP]
    grid_rect = raw_to_rect(rect, np.c_[gx.ravel(), gy.ravel()])

    variants = [("old: Gaussian only (no flat band)", None),
                (f"flat band, height = DVS height x {optics:.2f} (optics)", optics),
                (f"flat band, height = DVS height x {fitted:.2f} (FIT-part median)", fitted)]
    print("\nTEST part (last 40% per scene)")
    for name, h in variants:
        print(f"\n[{name}]")
        evaluate(Zone(rect, rows, vals, h), rect, corr, grid_rect)
    print("\nbox cov. = median share of the RGB label box with zone weight > 0.5; cov>=90% = share of labels with "
          "at least 90% of their box covered;\nfits h = label height <= flat band height; lift = mean weight at "
          "label centres / mean weight over the image")

    zone = Zone(rect, rows, vals, max(optics, fitted))
    fy, fx = np.mgrid[0:RGB_H, 0:RGB_W]
    grid_rect_full = raw_to_rect(rect, np.c_[fx.ravel(), fy.ravel()])
    for scene in SCENES:
        dvs, rgb = load_dvs(scene), load_rgb(scene)
        c = corr[scene]
        test = sorted(f for f in rgb if rgb[f] and c["test_frames"][0] <= f <= c["test_frames"][1])
        busiest = max(test, key=lambda f: (len(rgb[f]) + len(dvs.get(f + K, [])), -f))
        picks = [busiest] + [f for f in (test[len(test) // 3], test[2 * len(test) // 3]) if f != busiest]
        for f in picks[:3]:
            render(zone, scene, f, dvs, rgb, np.array(c["coef_rect_px"]), scene not in NO_DEPTH_LIMIT,
                   grid_rect_full)
    print(f"images drawn with flat band height = DVS height x {max(optics, fitted):.2f}")


if __name__ == "__main__":
    main()
