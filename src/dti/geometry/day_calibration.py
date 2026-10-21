"""
Step 9 of the zone rebuild: do the calibration recordings of the recording days
remove the row offset of the scenes?

So far one calibration recording (24 Oct 2022) was used for all six scenes, and
the RGB insects sat up to 17 px off the DVS line. On 2026-10-02 the calibration
recordings of the three recording days arrived
(~/Downloads/ictrap-2022-calibration, info.txt):
  scenes 1, 2     27 Oct 2022   2022-10-27_15-11-37, 2022-10-27_15-44-14
  scenes 3, 4, 5  28 Oct 2022   2022-10-28_15-26-49  (16-03-01 has no DVS video)
  scene 6          4 Oct 2022   2022-10-04_16-18-06, _16-44-05, _16-44-53

For every recording (board = asymmetric circle grid, every 10th frame, DVS frame
f paired with RGB frame f, worst 20 % of pairs dropped, as src/calibrate.py):
  full        new intrinsics and new relative pose from this recording
  pose only   intrinsics of the old calibration, only the relative pose new
              (more robust if the board covers little of the image)

Then, for every scene, the row offset between the DVS line and my RGB labels
(all labelled frames, best-case pairing, DVS annotation of frame f-1; no label is
used to fit anything) with: the old calibration, the old calibration plus the
label-free correction of step 13, and each calibration of the scene's day.

Board detections are cached in outputs/step9/ (detection is slow).
Run: PYTHONPATH=src ./venv/bin/python src/dti/geometry/day_calibration.py
"""

import json
import os

import cv2
import numpy as np

from dti.geometry import board_calibration as cb
from dti.io.data_sync import load_rgb
from dti.geometry import calibration as s2
from dti.geometry import rectify as s3
from dti.geometry import scene_correction as s4
from dti.io import scene6 as s6

SRC = os.path.expanduser("~/Downloads/ictrap-2022-calibration")
OUT_DIR = "outputs/step9"
LABELFREE = "outputs/step13/labelfree_corrections.json"
STRIDE = 10
DROP = 0.2
DAYS = {"27 Oct": (["1_l-l-l", "2_l-h-l"], ["2022-10-27_15-11-37", "2022-10-27_15-44-14"]),
        "28 Oct": (["3_m-h-h", "4_m-m-h", "5_h-l-h"], ["2022-10-28_15-26-49"]),
        "4 Oct": (["6_h-h-h"], ["2022-10-04_16-18-06", "2022-10-04_16-44-05", "2022-10-04_16-44-53"])}
SIZE_L, SIZE_R = (1280, 720), (1920, 1200)


def detect(rec, cam):
    """frame index -> (44, 2) circle centres, every STRIDE-th frame; cached."""
    cache = f"{OUT_DIR}/det_{rec}_{cam}.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        return dict(zip(d["frames"].tolist(), d["pts"]))
    cap, blobs, det, i = cv2.VideoCapture(f"{SRC}/{rec}_{cam}.mp4"), s2.blob_detector(), {}, 0
    while True:
        if i % STRIDE:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        found, c = cv2.findCirclesGrid(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (s2.COLS, s2.ROWS),
                                       flags=cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector=blobs)
        if found:
            det[i] = c.reshape(-1, 2)
        i += 1
    cap.release()
    np.savez(cache, frames=np.array(sorted(det)), pts=np.array([det[k] for k in sorted(det)]).reshape(-1, 44, 2))
    return det


def calibrate(obj, pl, pr, old):
    """Returns {'full': cal, 'pose only': cal}; each after dropping the worst DROP share of pairs."""
    out = {}
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    def pose_only(o, l, r):
        rms, *_, R, T, E, F = cv2.stereoCalibrate(o, l, r, old["KL"], old["dL"], old["KR"], old["dR"], SIZE_L,
                                                  flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)
        return dict(rms=rms, KL=old["KL"], dL=old["dL"], KR=old["KR"], dR=old["dR"], R=R, T=T, E=E, F=F)

    for name, fn in (("full", lambda o, l, r: cb.stereo_calibrate(o, l, r, SIZE_L, SIZE_R)), ("pose only", pose_only)):
        cal = fn(obj, pl, pr)
        err = cb.per_pair_epipolar_error(pl, pr, cal["F"])
        keep = np.argsort(err)[:len(err) - int(DROP * len(err))]
        cal = fn([obj[i] for i in keep], [pl[i] for i in keep], [pr[i] for i in keep])
        cal["epi"] = float(cb.per_pair_epipolar_error([pl[i] for i in keep], [pr[i] for i in keep], cal["F"]).mean())
        cal["n"] = len(keep)
        cal["sizeL"], cal["sizeR"] = np.array(SIZE_L), np.array(SIZE_R)
        out[name] = cal
    return out


def make_rect(cal):
    rect = s2.build_rectification(cal)
    rect.update(dL=cal["dL"], KR=cal["KR"], dR=cal["dR"])
    rect["to_rgb"] = cal["KR"][1, 1] / rect["P1"][1, 1]
    rect["fB_mm"] = abs(rect["P2"][0, 3]) / np.linalg.norm(cal["T"]) * s3.BASELINE_MM
    return rect


def coverage(pts, size):
    hull = cv2.convexHull(np.float32(np.concatenate(pts).reshape(-1, 2)))
    return cv2.contourArea(hull) / (size[0] * size[1])


def scene_row(name, rect, dvs, rgb, coef=None):
    frames = sorted(f for f in rgb if rgb[f])
    xy, off = s4.collect(rect, dvs, rgb, frames, coef)
    res = (off - s4.predict(coef, xy)) * rect["to_rgb"]
    _, disp, _ = s6.best_case(rect, dvs, rgb, s4.K)
    z = rect["fB_mm"] / disp[disp > 0] / 1000
    a = np.abs(res)
    return (f"  {name:32s} {len(a):4d} {np.median(a):9.1f} {np.mean(a < 5):7.0%} {np.mean(a < 10):8.0%} "
            f"{np.median(res):+9.1f} {np.mean(disp <= 0):9.0%} {np.median(z) if len(z) else float('nan'):9.2f}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    s3.OUT_DIR = OUT_DIR
    old = dict(np.load(s2.CALIB))
    old_rect = make_rect(old)
    lf = json.load(open(LABELFREE))
    rot = lambda R: np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))
    print(f"old calibration (24 Oct): focal DVS {old['KL'][0, 0]:.0f} px, RGB {old['KR'][0, 0]:.0f} px, rotation "
          f"DVS->RGB {rot(old['R']):.2f} deg, baseline direction {np.round(old['T'].ravel() / np.linalg.norm(old['T']), 3)}")

    cals = {}
    print(f"\n{'recording':22s} {'board DVS':>9s} {'board RGB':>9s} {'pairs':>5s} | {'variant':9s} {'kept':>4s} {'RMS px':>6s} "
          f"{'epipolar px':>11s} {'focal DVS':>9s} {'focal RGB':>9s} {'rotation deg':>12s} {'baseline direction':>22s} "
          f"{'cover DVS':>9s} {'cover RGB':>9s}")
    for day, (_, recs) in DAYS.items():
        for rec in recs:
            dl, dr = detect(rec, "dvs"), detect(rec, "rgb")
            common = sorted(set(dl) & set(dr))
            head = f"{rec:22s} {len(dl):9d} {len(dr):9d} {len(common):5d}"
            if len(common) < 10:
                print(f"{head} | too few pairs, not used", flush=True)
                continue
            obj = [cb.OBJECT_POINTS.copy() for _ in common]
            pl = [np.float32(dl[i]).reshape(-1, 1, 2) for i in common]
            pr = [np.float32(dr[i]).reshape(-1, 1, 2) for i in common]
            cals[rec] = calibrate(obj, pl, pr, old)
            for i, (name, c) in enumerate(cals[rec].items()):
                print(f"{head if i == 0 else '':47s} | {name:9s} {c['n']:4d} {c['rms']:6.2f} {c['epi']:11.2f} "
                      f"{c['KL'][0, 0]:9.0f} {c['KR'][0, 0]:9.0f} {rot(c['R']):12.2f} "
                      f"{str(np.round(c['T'].ravel() / np.linalg.norm(c['T']), 3)):>22s} "
                      f"{coverage(pl, SIZE_L):9.0%} {coverage(pr, SIZE_R):9.0%}", flush=True)

    print("\nRow offset between the DVS line and my RGB labels (RGB px), all labelled frames, nothing fitted to labels")
    for day, (scenes, recs) in DAYS.items():
        for scene in scenes:
            dvs = s3.load_dvs(scene)
            rgb = s6.load_rgb() if scene == s6.SCENE else load_rgb(scene)
            print(f"\n{scene} (recorded {day})\n  {'calibration':32s} {'n':>4s} {'median |.|':>9s} {'< 5 px':>7s} "
                  f"{'< 10 px':>8s} {'signed':>9s} {'disp <= 0':>9s} {'depth m':>9s}")
            print(scene_row("old (24 Oct)", old_rect, dvs, rgb))
            print(scene_row("old + label-free correction", old_rect, dvs, rgb, np.array(lf[scene]["coef_rect_px"])))
            best = None
            for rec in recs:
                for name, c in cals.get(rec, {}).items():
                    rect = make_rect(c)
                    print(scene_row(f"{rec[5:]} {name}", rect, dvs, rgb), flush=True)
                    if name == "pose only" and best is None:
                        best = rect
            if best is not None:
                frames = sorted(f for f in rgb if rgb[f])
                f = max(frames, key=lambda f: (len(rgb[f]), -f))
                s3.render(best, scene, f, dvs, rgb)
    print(f"\nimages: one rectified pair per scene with the first 'pose only' calibration of its day, in {OUT_DIR}/")
    print("median |.| = median distance label - line; signed = median with sign; disp <= 0 = share of impossible")
    print("(negative) disparities; depth = median insect depth from the positive disparities")


if __name__ == "__main__":
    main()
