"""
Step 2 of the zone rebuild: re-check the saved stereo calibration.

Uses data/output/stereo_calibration.npz as it is (no refit) and tests it on the
calibration video (asymmetric circle grid 4x11, same detector as calibrate.py):

  1. Reprojection error per camera: board pose from the camera's own points
     (solvePnP), project the grid back, distance to the detected circles.
  2. Transfer error: board pose from the DVS points only, move it into the RGB
     camera with the stereo R, T, project with the RGB intrinsics. This tests the
     extrinsics directly -- it is what the zone relies on.
  3. Epipolar error: distance of each RGB point to the epipolar line of its DVS
     point (after removing lens distortion from both).
  4. Rectified row difference: rectify both point sets; matching points should
     end up on the same image row.

Rectification: the DVS intrinsics are scaled to the RGB image size (DVS frame is
resized to 1920x1200 first), so both cameras share one output grid. This is the
set-up step 3 builds on.

Run: PYTHONPATH=src ./venv/bin/python src/dti/geometry/calibration.py
"""

import os

import cv2
import numpy as np

CALIB = "data/output/stereo_calibration.npz"
DVS_MP4 = "data/ictrap-2022-calibration/dvs.mp4"
RGB_MP4 = "data/ictrap-2022-calibration/rgb.mp4"
OUT_DIR = "outputs/step2"
STRIDE = 5
COLS, ROWS = 4, 11
WHITE, GREEN, CYAN = (255, 255, 255), (0, 255, 0), (255, 255, 0)
RECT_TO_RGB = 1.0  # set in main: RGB focal / rectified focal
SHIFTS = [-2, -1, 0, 1, 2]

# datasheet focal lengths in px (focal mm / pixel pitch)
DVS_F_DATASHEET = 4.5e3 / 4.86
RGB_F_DATASHEET = 6.0e3 / 4.8


def object_points():
    pts = [((2 * c + r % 2), r, 0.0) for r in range(ROWS) for c in range(COLS)]
    return np.float32(pts)


def blob_detector():
    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea, p.minArea, p.maxArea = True, 10, 100_000
    p.filterByCircularity, p.minCircularity = True, 0.6
    p.filterByConvexity, p.minConvexity = True, 0.8
    p.filterByInertia, p.minInertiaRatio = True, 0.3
    return cv2.SimpleBlobDetector_create(p)


def detect(path):
    """frame index -> (44, 2) circle centres, every STRIDE-th frame."""
    det, cap, i = {}, cv2.VideoCapture(path), 0
    blobs = blob_detector()
    while True:
        if i % STRIDE:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        found, c = cv2.findCirclesGrid(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (COLS, ROWS),
                                       flags=cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector=blobs)
        if found:
            det[i] = c.reshape(-1, 2)
        i += 1
    return det


def read_frame(path, index):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    assert ok, (path, index)
    return frame


def point_line_dist(pts, lines):
    a, b, c = lines.T
    return np.abs(a * pts[:, 0] + b * pts[:, 1] + c) / np.hypot(a, b)


def evaluate(cal, rect, obj, pl, pr):
    """All four error measures for one board pair (arrays per point)."""
    KL, dL, KR, dR, R, T, F = (cal[k] for k in ("KL", "dL", "KR", "dR", "R", "T", "F"))
    _, rl, tl = cv2.solvePnP(obj, pl, KL, dL)
    _, rr, tr = cv2.solvePnP(obj, pr, KR, dR)
    reproj_l = np.linalg.norm(cv2.projectPoints(obj, rl, tl, KL, dL)[0].reshape(-1, 2) - pl, axis=1)
    reproj_r = np.linalg.norm(cv2.projectPoints(obj, rr, tr, KR, dR)[0].reshape(-1, 2) - pr, axis=1)

    # DVS board pose -> RGB camera: X_rgb = R X_dvs + T
    Rl = cv2.Rodrigues(rl)[0]
    r_in_rgb = cv2.Rodrigues(R @ Rl)[0]
    t_in_rgb = R @ tl + T
    transfer = np.linalg.norm(cv2.projectPoints(obj, r_in_rgb, t_in_rgb, KR, dR)[0].reshape(-1, 2) - pr, axis=1)

    ul = cv2.undistortPoints(pl[:, None], KL, dL, P=KL).reshape(-1, 2)
    ur = cv2.undistortPoints(pr[:, None], KR, dR, P=KR).reshape(-1, 2)
    epi = point_line_dist(ur, cv2.computeCorrespondEpilines(ul[:, None], 1, F).reshape(-1, 3))

    yl = rect_points_dvs(rect, pl)[:, 1]
    yr = rect_points_rgb(rect, pr)[:, 1]
    return reproj_l, reproj_r, transfer, epi, yr - yl


def build_rectification(cal):
    size_l, size_r = tuple(int(v) for v in cal["sizeL"]), tuple(int(v) for v in cal["sizeR"])
    S = np.diag([size_r[0] / size_l[0], size_r[1] / size_l[1], 1.0])
    KLs = S @ cal["KL"]
    R1, R2, P1, P2, _, roi1, roi2 = cv2.stereoRectify(KLs, cal["dL"], cal["KR"], cal["dR"], size_r,
                                                      cal["R"], cal["T"], alpha=1, newImageSize=size_r)
    return dict(S=S, KLs=KLs, R1=R1, R2=R2, P1=P1, P2=P2, size=size_r,
                mapL=cv2.initUndistortRectifyMap(KLs, cal["dL"], R1, P1, size_r, cv2.CV_32FC1),
                mapR=cv2.initUndistortRectifyMap(cal["KR"], cal["dR"], R2, P2, size_r, cv2.CV_32FC1))


def rect_points_dvs(rect, pts):
    scaled = pts * np.diag(rect["S"])[:2]
    return cv2.undistortPoints(np.float32(scaled)[:, None], rect["KLs"], rect["dL"],
                               R=rect["R1"], P=rect["P1"]).reshape(-1, 2)


def rect_points_rgb(rect, pts):
    return cv2.undistortPoints(np.float32(pts)[:, None], rect["KR"], rect["dR"],
                               R=rect["R2"], P=rect["P2"]).reshape(-1, 2)


def caption(img, lines):
    cap = np.zeros((22 * len(lines) + 10, img.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(cap, t, (8, 22 * i + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return np.vstack([img, cap])


def colors(n):
    hsv = np.uint8([[[int(180 * i / n), 255, 255] for i in range(n)]])
    return [tuple(int(v) for v in c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]]


def render_rectified(rect, f, pl, pr, err):
    size = rect["size"]
    dvs = cv2.resize(read_frame(DVS_MP4, f), size, interpolation=cv2.INTER_LINEAR)
    dvs = cv2.remap(dvs, *rect["mapL"], cv2.INTER_LINEAR)
    rgb = cv2.remap(read_frame(RGB_MP4, f), *rect["mapR"], cv2.INTER_LINEAR)
    ql, qr = rect_points_dvs(rect, pl), rect_points_rgb(rect, pr)
    canvas = np.hstack([dvs, np.full((size[1], 16, 3), 40, np.uint8), rgb])
    off = size[0] + 16
    order = np.argsort(ql[:, 1])
    chosen = order[np.linspace(0, len(order) - 1, 8).astype(int)]
    col = colors(len(chosen))
    for k in range(len(ql)):
        cv2.circle(canvas, (int(ql[k, 0]), int(ql[k, 1])), 5, (150, 150, 150), 2)
        cv2.circle(canvas, (int(qr[k, 0]) + off, int(qr[k, 1])), 5, (150, 150, 150), 2)
    for k, c in zip(chosen, col):
        y = int(round(ql[k, 1]))
        cv2.line(canvas, (0, y), (canvas.shape[1], y), c, 1)
        cv2.circle(canvas, (int(ql[k, 0]), int(ql[k, 1])), 7, c, 3)
        cv2.circle(canvas, (int(qr[k, 0]) + off, int(qr[k, 1])), 7, c, 3)
    canvas = cv2.resize(canvas, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img = caption(canvas, [
        f"calibration video frame {f}, rectified   left: DVS (resized to 1920x1200, then rectified)   right: RGB (rectified)",
        "grey circles = all 44 grid centres after rectification; 8 centres spread over the board height in colour "
        "(same colour = same grid point), line = horizontal line through the coloured DVS centre",
        f"row difference RGB - DVS for this pair: mean {err.mean():+.2f}, max |.| {np.abs(err).max():.2f} rectified px "
        f"(x {RECT_TO_RGB:.2f} = RGB px)",
    ])
    cv2.imwrite(f"{OUT_DIR}/rectified_frame_{f:05d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def render_raw_epilines(cal, f, pl, pr, epi):
    """Unrectified RGB frame with epipolar lines of the 4 grid corners and all RGB centres."""
    KL, dL, KR, dR, F = (cal[k] for k in ("KL", "dL", "KR", "dR", "F"))
    rgb = read_frame(RGB_MP4, f)
    dvs = read_frame(DVS_MP4, f)
    corners = [0, COLS - 1, COLS * (ROWS - 1), COLS * ROWS - 1]
    col = colors(len(corners))
    ul = cv2.undistortPoints(pl[corners][:, None], KL, dL, P=KL)
    lines = cv2.computeCorrespondEpilines(ul, 1, F).reshape(-1, 3)
    w, h = rgb.shape[1], rgb.shape[0]
    for (a, b, c), color, k in zip(lines, col, corners):
        # line in undistorted RGB px; sample it and re-apply RGB distortion so it sits on the raw image
        xs = np.linspace(0, w, 200)
        ys = -(a * xs + c) / b
        norm = cv2.undistortPoints(np.float32(np.c_[xs, ys])[:, None], KR, None).reshape(-1, 2)
        pts3 = np.c_[norm, np.ones(len(norm))]
        raw = cv2.projectPoints(pts3, np.zeros(3), np.zeros(3), KR, dR)[0].reshape(-1, 2)
        cv2.polylines(rgb, [np.int32(raw)], False, color, 2)
        cv2.circle(dvs, (int(pl[k, 0]), int(pl[k, 1])), 8, color, 2)
    for p in pr:
        cv2.circle(rgb, (int(p[0]), int(p[1])), 5, GREEN, 2)
    s = 1200 / 720
    dvs = cv2.resize(dvs, None, fx=s, fy=s)
    canvas = cv2.resize(np.hstack([dvs, np.full((h, 16, 3), 40, np.uint8), rgb]), None, fx=0.5, fy=0.5,
                        interpolation=cv2.INTER_AREA)
    img = caption(canvas, [
        f"calibration video frame {f}, NOT rectified   left: DVS 1280x720 (shown enlarged)   right: RGB 1920x1200",
        "left: coloured circles = 4 grid corners in DVS; right: curve of the same colour = their epipolar lines "
        "(lens distortion applied), green circles = detected RGB centres",
        f"epipolar error for this pair (all 44 points): mean {epi.mean():.2f} px, max {epi.max():.2f} px (RGB px)",
    ])
    cv2.imwrite(f"{OUT_DIR}/raw_epilines_frame_{f:05d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def render_coverage(dl, dr, frames):
    """All detected grid centres of all pairs on one DVS and one RGB frame: where the calibration has data."""
    dvs, rgb = read_frame(DVS_MP4, frames[0]), read_frame(RGB_MP4, frames[0])
    dvs[:], rgb[:] = dvs // 3, rgb // 3
    for f in frames:
        for p in dl[f]:
            cv2.circle(dvs, (int(p[0]), int(p[1])), 3, CYAN, -1)
        for p in dr[f]:
            cv2.circle(rgb, (int(p[0]), int(p[1])), 4, GREEN, -1)
    s = 1200 / 720
    dvs = cv2.resize(dvs, None, fx=s, fy=s)
    canvas = cv2.resize(np.hstack([dvs, np.full((1200, 16, 3), 40, np.uint8), rgb]), None, fx=0.5, fy=0.5,
                        interpolation=cv2.INTER_AREA)
    cov = [coverage(np.concatenate([dl[f] for f in frames]), 1280, 720),
           coverage(np.concatenate([dr[f] for f in frames]), 1920, 1200)]
    img = caption(canvas, [
        f"calibration coverage: all grid centres of the {len(frames)} pairs (frame {frames[0]} darkened as background)",
        "left: DVS 1280x720 (cyan, shown enlarged)   right: RGB 1920x1200 (green)",
        f"share of image (10x10 grid of cells) containing at least one centre: DVS {cov[0]:.0%}, RGB {cov[1]:.0%}",
    ])
    cv2.imwrite(f"{OUT_DIR}/coverage.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return cov


def coverage(pts, w, h, n=10):
    cells = {(int(x * n / w), int(y * n / h)) for x, y in pts}
    return len(cells) / n ** 2


def timing_on_board(cal, dr, frames):
    """Epipolar error of DVS frame f+k vs RGB frame f, board only (no insect labels)."""
    KL, dL, KR, dR, F = (cal[k] for k in ("KL", "dL", "KR", "dR", "F"))
    need = sorted({f + k for f in frames for k in SHIFTS})
    blobs, cap, det = blob_detector(), cv2.VideoCapture(DVS_MP4), {}
    for i in need:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        found, c = cv2.findCirclesGrid(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (COLS, ROWS),
                                       flags=cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector=blobs)
        if found:
            det[i] = c.reshape(-1, 2)
    usable = [f for f in frames if all(f + k in det for k in SHIFTS)]
    motion = np.array([np.linalg.norm(det[f] - det[f - 1], axis=1).mean() for f in usable])
    out = {}
    for k in SHIFTS:
        e = []
        for f in usable:
            ul = cv2.undistortPoints(det[f + k][:, None], KL, dL, P=KL)
            ur = cv2.undistortPoints(dr[f][:, None], KR, dR, P=KR).reshape(-1, 2)
            e.append(point_line_dist(ur, cv2.computeCorrespondEpilines(ul, 1, F).reshape(-1, 3)).mean())
        out[k] = np.array(e)
    return usable, motion, out


def stats(name, v, unit="px"):
    v = np.abs(np.concatenate(v))
    print(f"  {name:34s} mean {v.mean():6.2f}  median {np.median(v):6.2f}  p95 {np.percentile(v, 95):6.2f}  "
          f"max {v.max():6.2f} {unit}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cal = dict(np.load(CALIB))
    rect = build_rectification(cal)
    rect.update(dL=cal["dL"], KR=cal["KR"], dR=cal["dR"])
    np.set_printoptions(precision=3, suppress=True)

    print("Saved calibration vs datasheet")
    print(f"  DVS fx, fy = {cal['KL'][0, 0]:.1f}, {cal['KL'][1, 1]:.1f}   datasheet {DVS_F_DATASHEET:.1f}   "
          f"cx, cy = {cal['KL'][0, 2]:.1f}, {cal['KL'][1, 2]:.1f} (centre 640, 360)")
    print(f"  RGB fx, fy = {cal['KR'][0, 0]:.1f}, {cal['KR'][1, 1]:.1f}   datasheet {RGB_F_DATASHEET:.1f}   "
          f"cx, cy = {cal['KR'][0, 2]:.1f}, {cal['KR'][1, 2]:.1f} (centre 960, 600)")
    angle = np.degrees(np.linalg.norm(cv2.Rodrigues(cal["R"])[0]))
    t = cal["T"].ravel()
    print(f"  rotation DVS->RGB {angle:.2f} deg;  T = {t} grid units, direction {t / np.linalg.norm(t)}")
    print(f"  distortion DVS {cal['dL'].ravel()}")
    print(f"  distortion RGB {cal['dR'].ravel()}")
    global RECT_TO_RGB
    RECT_TO_RGB = cal["KR"][1, 1] / rect["P1"][1, 1]
    print(f"  rectified image scale: 1 rectified px = {RECT_TO_RGB:.2f} RGB px")
    print(f"  rectification: P1 fy, cy = {rect['P1'][1, 1]:.2f}, {rect['P1'][1, 2]:.2f}   "
          f"P2 fy, cy = {rect['P2'][1, 1]:.2f}, {rect['P2'][1, 2]:.2f}  (must be equal)")

    print("Detecting grids in the calibration video ...")
    dl, dr = detect(DVS_MP4), detect(RGB_MP4)
    frames = sorted(set(dl) & set(dr))
    print(f"  DVS {len(dl)}  RGB {len(dr)}  pairs {len(frames)} (every {STRIDE}th frame)")

    obj = object_points()
    res = {f: evaluate(cal, rect, obj, dl[f], dr[f]) for f in frames}
    print("Errors over all pairs (includes the pairs the calibration was fitted on)")
    names = ["reprojection DVS (DVS px)", "reprojection RGB (RGB px)", "transfer DVS->RGB (RGB px)",
             "epipolar distance (RGB px)", "rectified row diff (rect. px)"]
    for i, n in enumerate(names):
        stats(n, [res[f][i] for f in frames])
    rowdiff = np.concatenate([res[f][4] for f in frames])
    print(f"  rectified row diff signed: mean {rowdiff.mean():+.2f} rect. px = {rowdiff.mean() * RECT_TO_RGB:+.2f} RGB px "
          "(a constant offset would show here)")
    cov = render_coverage(dl, dr, frames)
    print(f"Coverage (share of 10x10 image cells with a grid centre): DVS {cov[0]:.0%}, RGB {cov[1]:.0%}")

    usable, motion, tim = timing_on_board(cal, dr, frames)
    moving = motion > 3
    print(f"Timing on the board: DVS frame f+k vs RGB frame f, mean epipolar error per pair (RGB px)")
    print(f"  {len(usable)} pairs with the board found at all shifts; {moving.sum()} with board motion > 3 DVS px/frame "
          f"(median motion {np.median(motion):.1f})")
    for k in SHIFTS:
        print(f"  k={k:+d}: all pairs median {np.median(tim[k]):5.2f}   moving pairs median "
              f"{np.median(tim[k][moving]) if moving.any() else float('nan'):5.2f}")

    per_pair = np.array([res[f][3].mean() for f in frames])
    order = np.argsort(per_pair)
    picks = [frames[order[0]], frames[order[len(order) // 2]], frames[order[-1]]]
    print("Per-pair mean epipolar error: best / median / worst frame")
    for f in picks:
        e = res[f]
        print(f"  frame {f:5d}: epipolar {e[3].mean():.2f}  transfer {e[2].mean():.2f}  "
              f"row diff {e[4].mean():+.2f} px")
        render_rectified(rect, f, dl[f], dr[f], e[4])
    render_raw_epilines(cal, picks[1], dl[picks[1]], dr[picks[1]], res[picks[1]][3])
    print(f"Images in {OUT_DIR}")


if __name__ == "__main__":
    main()
