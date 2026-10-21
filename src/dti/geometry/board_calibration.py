"""
Script for Stereo-Calibration of RGB and DVS Cameras.
"""

import cv2
import numpy as np

# Video sources
DVS_MP4 = "data/ictrap-2022-calibration/dvs.mp4"
RGB_MP4 = "data/ictrap-2022-calibration/rgb.mp4"

# Calibration target
COLS = 4
ROWS = 11
SPACING = 1.0 # inter-circle spacing (arbitrary units)
PATTERN = (COLS, ROWS)
FRAME_STRIDE = 5 # sample every Nth frame

# Known optics: Prophesee EVK3 HD + Kowa LM5NCL 4.5 mm
DVS_FOCAL_MM   = 4.5
DVS_PIXEL_UM   = 4.86
DVS_RESOLUTION = (1280, 720)

# Known optics: Basler acA1920-150uc + Kowa LM6JC 6 mm
RGB_FOCAL_MM   = 6.0
RGB_PIXEL_UM   = 4.8
RGB_RESOLUTION = (1920, 1200)

# Filtering: fraction of worst-epipolar-error pairs to discard
DROP_FRACTION = 0.2

# Below this many synchronized pairs, calibration quality is unreliable
MIN_RECOMMENDED_PAIRS = 10


# Setup

def make_asymmetric_object_points(cols, rows, spacing):
    """Build 3-D object points for an asymmetric circle grid."""
    pts = np.zeros((rows * cols, 3), np.float32)
    k = 0
    for row in range(rows):
        for col in range(cols):
            pts[k] = ((2 * col + (row % 2)) * spacing, row * spacing, 0.0)
            k += 1
    return pts


OBJECT_POINTS = make_asymmetric_object_points(COLS, ROWS, SPACING)


def make_intrinsic_matrix(focal_mm, pixel_um, resolution):
    """Construct a pinhole camera matrix from known optics."""
    focal_px = focal_mm * 1000.0 / pixel_um
    cx, cy   = resolution[0] / 2.0, resolution[1] / 2.0
    return np.array(
        [[focal_px, 0.0,      cx],
         [0.0,      focal_px, cy],
         [0.0,      0.0,      1.0]],
        dtype=np.float64,
    )


K_DVS_INIT = make_intrinsic_matrix(DVS_FOCAL_MM, DVS_PIXEL_UM, DVS_RESOLUTION)
K_RGB_INIT = make_intrinsic_matrix(RGB_FOCAL_MM, RGB_PIXEL_UM, RGB_RESOLUTION)


def make_blob_detector():
    """Configure a SimpleBlobDetector tuned for calibration circles."""
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea        = True
    params.minArea             = 10
    params.maxArea             = 100_000
    params.filterByCircularity = True
    params.minCircularity      = 0.6
    params.filterByConvexity   = True
    params.minConvexity        = 0.8
    params.filterByInertia     = True
    params.minInertiaRatio     = 0.3
    return cv2.SimpleBlobDetector_create(params)


BLOB_DETECTOR = make_blob_detector()


def detect_grid(gray):
    """Detect the asymmetric circle grid in a grayscale frame."""
    return cv2.findCirclesGrid(
        gray, PATTERN,
        flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
        blobDetector=BLOB_DETECTOR,
    )


# Detection

def detect_all(path):
    """Detect the calibration grid in every sampled frame of a video.

    Returns:
        detections: dict mapping frame index -> corner array
        image_size: (width, height) of the video frames
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    detections = {}
    image_size = None
    frame_idx = 0

    while True:
        if frame_idx % FRAME_STRIDE != 0:
            # Skip decoding non-sampled frames; grab() only advances the stream.
            if not cap.grab():
                break
            frame_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_size = gray.shape[::-1] # (width, height)
        if image_size is not None and frame_size != image_size:
            raise ValueError(
                f"{path}: frame {frame_idx} has size {frame_size}, "
                f"expected {image_size} (frame size changed mid-video)"
            )
        image_size = frame_size
        found, centers = detect_grid(gray)
        if found:
            detections[frame_idx] = centers
        frame_idx += 1

    cap.release()
    return detections, image_size


def pair_frames(detections_left, detections_right):
    """Return synchronized frame pairs where both cameras detected the grid.

    Pairing is done purely by integer frame index: frame N of the left
    video is assumed to correspond to frame N of the right video. This
    relies on the two recordings already being frame-rate-matched and
    phase-aligned (e.g. hardware-triggered capture) upstream of this script.
    """
    object_pts, left_pts, right_pts = [], [], []
    for idx, corners_left in detections_left.items():
        if idx in detections_right:
            object_pts.append(OBJECT_POINTS.copy())
            left_pts.append(corners_left)
            right_pts.append(detections_right[idx])
    return object_pts, left_pts, right_pts


# Calibration

def stereo_calibrate(object_pts, left_pts, right_pts, size_left, size_right,
                      rational=False):
    """Run per-camera calibration, then joint stereo calibration.

    Args:
        object_pts:  list of object-point arrays (one per frame pair)
        left_pts:    detected corners in the left (DVS) camera
        right_pts:   detected corners in the right (RGB) camera
        size_left:   (width, height) of left-camera images
        size_right:  (width, height) of right-camera images
        rational:    add CALIB_RATIONAL_MODEL for extra distortion coefficients

    Returns:
        dict with keys rms, KL, dL, KR, dR, R, T, E, F
    """
    mono_flags = cv2.CALIB_USE_INTRINSIC_GUESS
    if rational:
        mono_flags |= cv2.CALIB_RATIONAL_MODEL

    _, KL, dL, *_ = cv2.calibrateCamera(
        object_pts, left_pts, size_left, K_DVS_INIT.copy(), None, flags=mono_flags
    )
    _, KR, dR, *_ = cv2.calibrateCamera(
        object_pts, right_pts, size_right, K_RGB_INIT.copy(), None, flags=mono_flags
    )

    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    if rational:
        stereo_flags |= cv2.CALIB_RATIONAL_MODEL

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    rms, KL, dL, KR, dR, R, T, E, F = cv2.stereoCalibrate(
        object_pts, left_pts, right_pts,
        KL, dL, KR, dR, size_left,
        flags=stereo_flags, criteria=criteria,
    )

    return dict(rms=rms, KL=KL, dL=dL, KR=KR, dR=dR, R=R, T=T, E=E, F=F)


def per_pair_epipolar_error(left_pts, right_pts, F):
    """Compute the mean symmetric epipolar error for each frame pair.

    For every correspondence (pL, pR), the error is the average of:
      - distance from pR to its epipolar line in the right image
      - distance from pL to its epipolar line in the left image
    """
    # computeCorrespondEpilines' 2nd arg is *which image the input points are
    # from* (1=left, 2=right); it returns lines in the other image.
    POINTS_FROM_LEFT_IMAGE = 1
    POINTS_FROM_RIGHT_IMAGE = 2

    errors = []
    for corners_left, corners_right in zip(left_pts, right_pts):
        pts_left  = corners_left.reshape(-1, 2)
        pts_right = corners_right.reshape(-1, 2)

        # Epipolar lines in right image (projected from left points)
        lines_in_right = cv2.computeCorrespondEpilines(
            corners_left.reshape(-1, 1, 2), POINTS_FROM_LEFT_IMAGE, F
        ).reshape(-1, 3)

        # Epipolar lines in left image (projected from right points)
        lines_in_left = cv2.computeCorrespondEpilines(
            corners_right.reshape(-1, 1, 2), POINTS_FROM_RIGHT_IMAGE, F
        ).reshape(-1, 3)

        point_errors = []
        for pt, (a, b, c) in zip(pts_right, lines_in_right):
            point_errors.append(abs(a * pt[0] + b * pt[1] + c) / np.hypot(a, b))
        for pt, (a, b, c) in zip(pts_left, lines_in_left):
            point_errors.append(abs(a * pt[0] + b * pt[1] + c) / np.hypot(a, b))

        errors.append(np.mean(point_errors))

    return np.array(errors)


def run_calibration_step(object_pts, left_pts, right_pts, size_left, size_right, label):
    """Run stereo_calibrate + per_pair_epipolar_error and print a summary line."""
    result = stereo_calibrate(object_pts, left_pts, right_pts, size_left, size_right)
    epi_errors = per_pair_epipolar_error(left_pts, right_pts, result["F"])
    print(f"{label} pairs  stereo RMS = {result['rms']:.3f} px "
          f"mean epipolar error = {epi_errors.mean():.3f} px")
    return result, epi_errors


# Main

def main():
    print("Detecting calibration grids in both videos...")
    detections_left, size_left = detect_all(DVS_MP4)
    detections_right, size_right = detect_all(RGB_MP4)
    print(f"DVS frames with grid: {len(detections_left)}  "
          f"RGB frames with grid: {len(detections_right)}")

    object_pts, left_pts, right_pts = pair_frames(detections_left, detections_right)
    n_pairs = len(object_pts)
    print(f"Synchronized pairs: {n_pairs}")
    if n_pairs == 0:
        raise RuntimeError(
            "No synchronized frame pairs found: neither video produced a grid "
            "detection at a matching frame index. Check the video paths, "
            "FRAME_STRIDE, and that both recordings cover overlapping frame indices."
        )
    if n_pairs < MIN_RECOMMENDED_PAIRS:
        print(f"WARNING: only {n_pairs} synchronized pairs found "
              f"(recommended minimum: {MIN_RECOMMENDED_PAIRS}); "
              "calibration may be unstable.")

    # Step 1: Baseline calibration
    print("\n[1] Baseline (standard distortion model):")
    result, epi_errors = run_calibration_step(
        object_pts, left_pts, right_pts, size_left, size_right, n_pairs
    )

    # Step 2: Drop worst pairs and refit
    n_drop = int(n_pairs * DROP_FRACTION)
    n_keep = n_pairs - n_drop
    if n_keep == 0:
        raise RuntimeError(
            f"DROP_FRACTION={DROP_FRACTION} would drop all {n_pairs} pairs "
            "(n_keep=0). Lower DROP_FRACTION or collect more synchronized pairs."
        )
    print(f"\n[2] Dropping worst {int(DROP_FRACTION * 100)}% ({n_drop} pairs), "
          f"keeping {n_keep}:")

    keep_idx = np.argsort(epi_errors)[:n_keep]
    object_pts_f = [object_pts[i] for i in keep_idx]
    left_pts_f = [left_pts[i] for i in keep_idx]
    right_pts_f = [right_pts[i] for i in keep_idx]

    result, epi_errors_f = run_calibration_step(
        object_pts_f, left_pts_f, right_pts_f, size_left, size_right, n_keep
    )

    # Rectify and save
    KL, dL = result["KL"], result["dL"]
    KR, dR = result["KR"], result["dR"]
    R, T, E, F = result["R"], result["T"], result["E"], result["F"]

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        KL, dL, KR, dR, size_left, R, T, alpha=0
    )
    mapLx, mapLy = cv2.initUndistortRectifyMap(KL, dL, R1, P1, size_left,  cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(KR, dR, R2, P2, size_right, cv2.CV_32FC1)

    out_path = "data/output/stereo_calibration_v2.npz"
    np.savez(
        out_path,
        KL=KL, dL=dL, KR=KR, dR=dR,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        mapLx=mapLx, mapLy=mapLy, mapRx=mapRx, mapRy=mapRy,
        sizeL=size_left, sizeR=size_right,
    )

    baseline_mm = 60.0 # known physical baseline between cameras
    t_norm = np.linalg.norm(T)
    print(f"\nSaved {out_path}")
    print(f"||T|| = {t_norm:.3f} units")
    if t_norm == 0:
        print("scale (60 mm baseline) = undefined (||T|| is zero; calibration is degenerate)")
    else:
        scale = baseline_mm / t_norm
        print(f"scale (60 mm baseline) = {scale:.3f} mm/unit")


if __name__ == "__main__":
    main()