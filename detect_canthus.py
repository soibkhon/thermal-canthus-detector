#!/usr/bin/env python3
"""
Thermal inner canthus detector.

Detects the two inner eye corners (inner canthi) in thermal infrared face images
using a YOLOv8-pose model fine-tuned on the IS2AI SF-TL54 thermal dataset.

CLI usage
---------
    python detect_canthus.py image.png
    python detect_canthus.py image.png --out result.png
    python detect_canthus.py image.png --no-save          # print coords only

Module usage
------------
    from detect_canthus import load_model, detect

    model = load_model()                          # load once
    left, right = detect(model, "image.png")      # or pass a numpy BGR array
    # left  = (x, y) pixel coords — inner canthus on the LEFT  side of image
    # right = (x, y) pixel coords — inner canthus on the RIGHT side of image
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

WEIGHTS  = Path(__file__).parent / "thermal_canthus_yolo.pt"
DOT_R    = 6
DOT_CLR  = (255, 0, 0)       # blue (BGR)
RING_CLR = (255, 255, 255)


# ── preprocessing ─────────────────────────────────────────────────────────────

def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """False-colour thermal → CLAHE-enhanced grayscale, returned as 3-ch RGB."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enh   = clahe.apply(gray)
    return cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)


# ── public API ────────────────────────────────────────────────────────────────

def load_model(weights: str | Path = WEIGHTS):
    """Load the YOLO model. Call once and reuse for multiple images."""
    from ultralytics import YOLO
    return YOLO(str(weights))


def detect(
    model,
    image: str | Path | np.ndarray,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """
    Detect inner canthi in a thermal face image.

    Parameters
    ----------
    model   : YOLO model returned by load_model()
    image   : file path (str/Path) or BGR numpy array

    Returns
    -------
    (left_canthus, right_canthus)
        Each is an (x, y) pixel tuple, or None if detection failed.
        left  = inner canthus on the LEFT  side of the image (person's right eye)
        right = inner canthus on the RIGHT side of the image (person's left  eye)
    """
    if isinstance(image, (str, Path)):
        img_bgr = cv2.imread(str(image))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    else:
        img_bgr = image

    results = model(_preprocess(img_bgr), verbose=False)[0]

    if results.keypoints is None or len(results.keypoints) == 0:
        return None, None

    best = int(results.boxes.conf.cpu().numpy().argmax())
    kpts = results.keypoints.xy[best].cpu().numpy()   # (2, 2)

    if kpts.shape[0] < 2:
        return None, None

    left  = (int(kpts[0, 0]), int(kpts[0, 1]))
    right = (int(kpts[1, 0]), int(kpts[1, 1]))
    return left, right


def annotate(
    image: str | Path | np.ndarray,
    left:  tuple[int, int] | None,
    right: tuple[int, int] | None,
) -> np.ndarray:
    """Draw blue dots at the detected canthus locations. Returns annotated BGR array."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
    else:
        img = image.copy()

    for pt in (left, right):
        if pt is not None:
            cv2.circle(img, pt, DOT_R,     DOT_CLR,  -1)
            cv2.circle(img, pt, DOT_R + 2, RING_CLR,  1)
    return img


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Detect inner canthi in a thermal face image."
    )
    ap.add_argument("image",    help="path to thermal image (PNG/JPG)")
    ap.add_argument("--weights", default=str(WEIGHTS),
                    help="path to YOLO weights (default: thermal_canthus_yolo.pt)")
    ap.add_argument("--out",    default=None,
                    help="output image path (default: <stem>_canthus.<ext>)")
    ap.add_argument("--no-save", action="store_true",
                    help="print coordinates only, do not save output image")
    args = ap.parse_args()

    src = Path(args.image)
    if not src.exists():
        print(f"Error: file not found: {src}", file=sys.stderr)
        sys.exit(1)

    model = load_model(args.weights)
    left, right = detect(model, src)

    if left is None and right is None:
        print("No face detected.", file=sys.stderr)
        sys.exit(1)

    print(f"Left  inner canthus: {left}")
    print(f"Right inner canthus: {right}")

    if not args.no_save:
        out_path = Path(args.out) if args.out else src.parent / f"{src.stem}_canthus{src.suffix}"
        cv2.imwrite(str(out_path), annotate(src, left, right))
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
