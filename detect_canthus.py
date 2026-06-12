#!/usr/bin/env python3
"""
Two-stage thermal inner canthus detector.

Stage 1 — face detection  (detector_weights, default: thermal_detector.pt)
  Robust to distance and outdoor conditions (trained on SF-TL54 + TFW).
  Finds the face bounding box regardless of range.

Stage 2 — canthus regression  (regressor_weights, default: thermal_canthus_yolo.pt)
  High-precision keypoint model (trained on SF-TL54 only).
  Runs on the cropped + upscaled face region for sub-pixel accuracy.
  Skipped when the face is too small (person too far → canthi not resolvable).

CLI usage
---------
    python detect_canthus.py image.png
    python detect_canthus.py image.png --out result.png
    python detect_canthus.py image.png --no-save
    python detect_canthus.py image.png --min-face 40   # lower distance threshold

Module usage
------------
    from detect_canthus import load_models, detect

    models = load_models()
    result = detect(models, "image.png")

    result.left        # (x, y) or None
    result.right       # (x, y) or None
    result.face_box    # (x1, y1, x2, y2) or None
    result.too_far     # True if face found but too small for canthus
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HERE             = Path(__file__).parent
DETECTOR_WEIGHTS  = HERE / "thermal_detector.pt"     # v3: face detector
REGRESSOR_WEIGHTS = HERE / "thermal_canthus_yolo.pt" # v1: canthus regressor

DOT_R    = 6
DOT_CLR  = (255, 0, 0)
RING_CLR = (255, 255, 255)

# Face bbox width (px in original image) below which canthi are not resolvable.
# At this size the inter-canthus distance is ~1-2px — not meaningful to report.
MIN_FACE_PX = 80

# Padding added around the detected face bbox before cropping for stage 2.
CROP_PAD = 0.25


# ── result type ───────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    left:     tuple[int, int] | None  # inner canthus, left side of image
    right:    tuple[int, int] | None  # inner canthus, right side of image
    face_box: tuple[int, int, int, int] | None  # (x1,y1,x2,y2) in original image
    too_far:  bool  # face detected but too small for canthus localisation


# ── preprocessing ─────────────────────────────────────────────────────────────

def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2RGB)


# ── model loading ─────────────────────────────────────────────────────────────

def load_models(
    detector:  str | Path = DETECTOR_WEIGHTS,
    regressor: str | Path = REGRESSOR_WEIGHTS,
) -> dict:
    """
    Load both stage models. Call once, reuse for many images.

    Falls back to a single-model mode if detector weights are missing
    (e.g. first-run before thermal_detector.pt is provided).
    """
    from ultralytics import YOLO
    det_path = Path(detector)
    reg_path = Path(regressor)

    if not reg_path.exists():
        raise FileNotFoundError(f"Regressor weights not found: {reg_path}")

    if not det_path.exists():
        print(f"[warn] Detector weights not found ({det_path}). "
              "Using regressor for both stages (single-model fallback).")
        m = YOLO(str(reg_path))
        return {"detector": m, "regressor": m, "single_model": True}

    return {
        "detector":     YOLO(str(det_path)),
        "regressor":    YOLO(str(reg_path)),
        "single_model": False,
    }


# ── single-model compat shim (old API) ───────────────────────────────────────

def load_model(weights: str | Path = REGRESSOR_WEIGHTS):
    """Legacy single-model loader. Returns a models dict usable with detect()."""
    from ultralytics import YOLO
    m = YOLO(str(weights))
    return {"detector": m, "regressor": m, "single_model": True}


# ── stage 1: face detection ───────────────────────────────────────────────────

def _detect_face(detector, img_bgr: np.ndarray) -> tuple[int,int,int,int] | None:
    """Return (x1,y1,x2,y2) of highest-confidence face, or None."""
    results = detector(_preprocess(img_bgr), verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None
    best = int(results.boxes.conf.cpu().numpy().argmax())
    x1, y1, x2, y2 = results.boxes.xyxy[best].cpu().numpy().astype(int)
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return (x1, y1, x2, y2)


# ── stage 2: canthus regression on cropped face ───────────────────────────────

def _regress_canthus(regressor, img_bgr: np.ndarray, box: tuple, min_face_px: int):
    """
    Crop the face region (with padding), run regressor, map coords back.
    Returns (left, right, too_far).
    """
    x1, y1, x2, y2 = box
    face_w = x2 - x1
    face_h = y2 - y1

    if face_w < min_face_px:
        return None, None, True   # too far — canthi not resolvable

    # padded crop
    h, w = img_bgr.shape[:2]
    pad_x = int(face_w * CROP_PAD)
    pad_y = int(face_h * CROP_PAD)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    crop = img_bgr[cy1:cy2, cx1:cx2]

    results = regressor(_preprocess(crop), verbose=False)[0]
    if results.keypoints is None or len(results.keypoints) == 0:
        return None, None, False

    best = int(results.boxes.conf.cpu().numpy().argmax())
    kpts = results.keypoints.xy[best].cpu().numpy()
    if kpts.shape[0] < 2:
        return None, None, False

    # map crop coords back to original image
    left  = (int(kpts[0, 0]) + cx1, int(kpts[0, 1]) + cy1)
    right = (int(kpts[1, 0]) + cx1, int(kpts[1, 1]) + cy1)
    return left, right, False


# ── public API ────────────────────────────────────────────────────────────────

def detect(
    models,
    image: str | Path | np.ndarray,
    min_face_px: int = MIN_FACE_PX,
) -> DetectionResult:
    """
    Two-stage detection: find face → localise canthi on crop.

    Parameters
    ----------
    models       : dict returned by load_models() or load_model()
    image        : file path or BGR numpy array
    min_face_px  : face bbox width threshold below which canthi are not reported

    Returns
    -------
    DetectionResult with .left, .right, .face_box, .too_far
    """
    if isinstance(image, (str, Path)):
        img_bgr = cv2.imread(str(image))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    else:
        img_bgr = image

    # stage 1
    box = _detect_face(models["detector"], img_bgr)
    if box is None:
        return DetectionResult(None, None, None, False)

    # stage 2
    left, right, too_far = _regress_canthus(
        models["regressor"], img_bgr, box, min_face_px
    )
    return DetectionResult(left, right, box, too_far)


def annotate(
    image: str | Path | np.ndarray,
    result: DetectionResult,
) -> np.ndarray:
    """Draw face box, canthus dots, and status label on the image."""
    img = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image.copy()

    # face bbox
    if result.face_box:
        x1, y1, x2, y2 = result.face_box
        color = (0, 180, 255) if result.too_far else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        label = "too far" if result.too_far else "face"
        cv2.putText(img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1)

    # canthus dots
    for pt in (result.left, result.right):
        if pt is not None:
            cv2.circle(img, pt, DOT_R,     DOT_CLR,  -1)
            cv2.circle(img, pt, DOT_R + 2, RING_CLR,  1)

    return img


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--detector",  default=str(DETECTOR_WEIGHTS))
    ap.add_argument("--regressor", default=str(REGRESSOR_WEIGHTS))
    ap.add_argument("--min-face",  type=int, default=MIN_FACE_PX,
                    help="min face width in px to attempt canthus detection")
    ap.add_argument("--out",      default=None)
    ap.add_argument("--no-save",  action="store_true")
    args = ap.parse_args()

    src = Path(args.image)
    if not src.exists():
        print(f"Error: {src} not found", file=sys.stderr); sys.exit(1)

    models = load_models(args.detector, args.regressor)
    result = detect(models, src, min_face_px=args.min_face)

    if result.face_box is None:
        print("No face detected."); sys.exit(1)

    if result.too_far:
        print(f"Face detected at {result.face_box} but too far — canthi not resolvable.")
    else:
        print(f"Left  inner canthus: {result.left}")
        print(f"Right inner canthus: {result.right}")

    if not args.no_save:
        out = Path(args.out) if args.out else src.parent / f"{src.stem}_canthus{src.suffix}"
        cv2.imwrite(str(out), annotate(src, result))
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
