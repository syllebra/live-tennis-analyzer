#!/usr/bin/env python3
"""Smoke-test the TensorRT models on a single video frame.

Example:
    python tools/verify_models.py --video videos/sample_tennis.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception import ObjectDetector, KeypointDetector  # noqa: E402
from utils import filter_detections  # noqa: E402

COCO_NAMES = {0: "person", 32: "sports ball"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="videos/sample_tennis.mp4")
    parser.add_argument("--object-model", default="models/rtdetrv2_r50vd_6x_coco_ema.trt")
    parser.add_argument("--keypoint-model", default="models/keypoint_detector.trt")
    parser.add_argument("--frame", type=int, default=0, help="frame index to test")
    parser.add_argument("--save", default="", help="optional path for an annotated preview JPEG")
    args = parser.parse_args()

    for path, label in ((args.object_model, "object"), (args.keypoint_model, "keypoint")):
        if not Path(path).exists():
            print(f"ERROR: {label} model not found: {path}")
            return 1

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}")
        return 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"ERROR: cannot read frame {args.frame} from {args.video}")
        return 1

    print(f"frame {args.frame}: {frame.shape[1]}x{frame.shape[0]}")

    object_detector = ObjectDetector(args.object_model)
    keypoint_detector = KeypointDetector(args.keypoint_model)

    detections = object_detector.predict(frame)
    keypoints = keypoint_detector.predict(frame)

    for class_id, name in COCO_NAMES.items():
        boxes, scores, labels = filter_detections(detections, 0.3, [class_id])
        print(f"{name:12s}: {len(boxes)} detection(s)")
        for box, score in zip(boxes.tolist(), scores.tolist()):
            print(f"  score={score:.3f} box={[round(v, 1) for v in box]}")

    print("keypoints (x, y):")
    for index in range(0, len(keypoints), 2):
        print(f"  {index // 2:2d}: ({keypoints[index]:8.2f}, {keypoints[index + 1]:8.2f})")

    if args.save:
        frame = object_detector.draw_ball(frame, detections)
        frame = keypoint_detector.draw_keypoints(frame, keypoints)
        boxes, scores, labels = filter_detections(detections, 0.6, [0])
        for box, score in zip(boxes.tolist(), scores.tolist()):
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"person {score:.2f}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.save, frame)
        print(f"saved preview -> {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
