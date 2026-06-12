#!/usr/bin/env python3
"""
Live two-stage inner canthus detection.

Stage 1 (thermal_detector.pt)    — detects face at any distance
Stage 2 (thermal_canthus_yolo.pt) — localises canthi on the cropped face

Usage
-----
    python live_canthus.py                      # webcam 0
    python live_canthus.py --source 1           # second camera / thermal USB
    python live_canthus.py --source video.mp4
    python live_canthus.py --device cuda
    python live_canthus.py --min-face 40        # report canthi for smaller faces
    python live_canthus.py --save out.mp4
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from detect_canthus import load_models, detect, annotate, DetectionResult


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",    default="0")
    ap.add_argument("--detector",  default=None)
    ap.add_argument("--regressor", default=None)
    ap.add_argument("--device",    default="cpu")
    ap.add_argument("--min-face",  type=int, default=80,
                    help="face width px threshold for canthus detection (default 80)")
    ap.add_argument("--save",      default=None, metavar="OUT.mp4")
    return ap.parse_args()


def open_source(source: str):
    try:
        src = int(source)
    except ValueError:
        src = source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")
    return cap


def overlay_info(frame, fps, result: DetectionResult):
    h, w = frame.shape[:2]

    # FPS
    cv2.putText(frame, f"{fps:.0f} fps", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0),     3)
    cv2.putText(frame, f"{fps:.0f} fps", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80,255,80),  1)

    # status line
    if result.face_box is None:
        status = "no face"
        col = (80, 80, 255)
    elif result.too_far:
        bw = result.face_box[2] - result.face_box[0]
        status = f"too far  (face {bw}px wide)"
        col = (0, 180, 255)
    else:
        status = f"L:{result.left}  R:{result.right}"
        col = (255, 200, 80)

    cv2.putText(frame, status, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),  2)
    cv2.putText(frame, status, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col,       1)


def main():
    args = parse_args()

    from detect_canthus import DETECTOR_WEIGHTS, REGRESSOR_WEIGHTS
    det_w = args.detector  or str(DETECTOR_WEIGHTS)
    reg_w = args.regressor or str(REGRESSOR_WEIGHTS)

    print("Loading models …")
    models = load_models(det_w, reg_w)
    if args.device != "cpu":
        models["detector"].to(args.device)
        models["regressor"].to(args.device)

    cap    = open_source(args.source)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w_in   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source : {args.source}  ({w_in}×{h_in} @ {fps_in:.0f} fps)")
    print(f"Min face width for canthus: {args.min_face}px")
    print("Press Q or Esc to quit.\n")

    writer  = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (w_in, h_in))
        print(f"Recording → {args.save}")

    t_prev  = time.perf_counter()
    fps_avg = fps_in

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = detect(models, frame, min_face_px=args.min_face)
        out    = annotate(frame, result)

        t_now   = time.perf_counter()
        fps_avg = 0.9 * fps_avg + 0.1 / max(t_now - t_prev, 1e-6)
        t_prev  = t_now

        overlay_info(out, fps_avg, result)
        cv2.imshow("Thermal Canthus  (Q = quit)", out)

        if writer:
            writer.write(out)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved → {args.save}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
