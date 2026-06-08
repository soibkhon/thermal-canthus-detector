#!/usr/bin/env python3
"""
Live inner canthus detection from webcam, thermal camera, or video file.

Usage
-----
    python live_canthus.py                      # webcam 0
    python live_canthus.py --source 1           # second camera
    python live_canthus.py --source video.mp4   # video file
    python live_canthus.py --device cuda        # run model on GPU
    python live_canthus.py --save out.mp4       # record output
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from detect_canthus import load_model, detect, annotate


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="camera index (0,1,…) or path to video file")
    ap.add_argument("--weights", default=None,
                    help="override YOLO weights path")
    ap.add_argument("--device", default="cpu",
                    help="inference device: cpu | cuda | 0")
    ap.add_argument("--save", default=None, metavar="OUT.mp4",
                    help="save annotated video to file")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="minimum face detection confidence (default 0.25)")
    return ap.parse_args()


def open_source(source: str):
    try:
        src = int(source)
    except ValueError:
        src = source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    return cap


def overlay_info(frame, fps, left, right):
    h, w = frame.shape[:2]
    # FPS badge
    cv2.putText(frame, f"{fps:.0f} fps", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),    3)
    cv2.putText(frame, f"{fps:.0f} fps", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 80), 1)
    # coordinates
    def coord_str(pt, label):
        return f"{label}: {pt}" if pt else f"{label}: --"
    cv2.putText(frame, coord_str(left,  "L"), (8, h - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),      2)
    cv2.putText(frame, coord_str(left,  "L"), (8, h - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 80),  1)
    cv2.putText(frame, coord_str(right, "R"), (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),      2)
    cv2.putText(frame, coord_str(right, "R"), (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 80),  1)


def main():
    args = parse_args()

    print("Loading model …")
    from detect_canthus import WEIGHTS
    weights = args.weights or WEIGHTS
    model = load_model(weights)
    # push model to requested device
    if args.device != "cpu":
        model.to(args.device)

    cap = open_source(args.source)
    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30
    w_in    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source  : {args.source}  ({w_in}×{h_in} @ {fps_in:.0f} fps)")
    print("Press Q or Esc to quit.\n")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (w_in, h_in))
        print(f"Recording → {args.save}")

    # rolling average FPS
    t_prev   = time.perf_counter()
    fps_avg  = fps_in

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        left, right = detect(model, frame)
        out = annotate(frame, left, right)

        # FPS
        t_now   = time.perf_counter()
        fps_avg = 0.9 * fps_avg + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev  = t_now

        overlay_info(out, fps_avg, left, right)

        cv2.imshow("Thermal Canthus Detector  (Q = quit)", out)

        if writer:
            writer.write(out)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):   # Q or Esc
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved → {args.save}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
