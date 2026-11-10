import argparse
import queue
import threading
import time
from pathlib import Path

import cv2

from perception import ObjectDetector, KeypointDetector
from streaming import FPSMonitor, StreamHandler
from minimap import MiniMap


def _check_model(path: str, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{label} model not found: {path}\n"
            "Run `python tools/setup_models.py --all` first, or pass the correct "
            "path on the command line."
        )


def main(
    stream_url,
    object_model_path,
    keypoint_model_path,
    output_path="output.mp4",
    display=True,
    max_frames=None,
    minimap_width=200,
    output_fps=13.0,
    draw_keypoints=True,
    draw_court_lines=True,
    show_court_status=True,
):
    _check_model(object_model_path, "Object detector")
    _check_model(keypoint_model_path, "Keypoint detector")

    # Bounded queue: this prevents a local file from being decoded entirely
    # into RAM and makes the decoder wait for the consumer instead.
    frame_queue = queue.Queue(maxsize=30)
    stop_flag = threading.Event()

    # handle stream, skip frames based on how fast processing is
    fps_feedback = FPSMonitor()
    stream_handler = StreamHandler(stream_url, frame_queue, stop_flag, fps_feedback)

    # start stream handler thread
    decode_thread = threading.Thread(target=stream_handler.start, daemon=True)
    decode_thread.start()

    object_detector = ObjectDetector(object_model_path)
    keypoint_detector = KeypointDetector(keypoint_model_path)

    writer = None
    minimap = None
    window_name = "Prediction"

    start = time.perf_counter()
    frames_processed = 0

    try:
        while True:
            try:
                frame = frame_queue.get(block=True, timeout=2)
            except queue.Empty:
                print("Frame buffer is empty, exiting...")
                break

            if minimap is None:
                height, width = frame.shape[:2]
                court_aspect = MiniMap.court_length / MiniMap.court_width
                max_width = max(40, int((height - 40) / court_aspect))
                width_px = max(40, min(int(minimap_width), max_width, max(40, width // 3)))
                position = (100, 250) if 250 + int(width_px * court_aspect) <= height else (20, 20)
                minimap = MiniMap(position=position, width=width_px, alpha=0.6)
                if output_path:
                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open video writer for {output_path}")
                    print(f"Writing annotated video to {output_path} ({width}x{height} @ {output_fps} fps)")

            keypoint_detections = keypoint_detector.predict(frame)
            object_detections = object_detector.predict(frame)

            # Draw the court terrain/keypoints before the HUD so they remain
            # visible and make a bad court fit obvious in diagnostic videos.
            if draw_keypoints:
                frame = keypoint_detector.draw_keypoints(
                    frame,
                    keypoint_detections,
                    draw_court=draw_court_lines,
                )

            frame = object_detector.draw_ball(frame, object_detections)
            frame = minimap.draw_minimap(frame)
            frame = minimap.draw_players(
                frame,
                object_detections,
                keypoint_detections,
                (181, 91, 241),
                (212, 245, 0),
            )
            frame = minimap.draw_ball(frame, object_detections, keypoint_detections, (64, 228, 254))

            if show_court_status:
                homography = minimap.find_homography(keypoint_detections)
                status = "court keypoints: OK" if homography is not None else "court keypoints: FIT FAILED"
                status_color = (0, 255, 0) if homography is not None else (0, 0, 255)
                status_origin = (12, frame.shape[0] - 16)
                cv2.putText(frame, status, status_origin, cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, status, status_origin, cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, status_color, 2, cv2.LINE_AA)

            if writer is not None:
                writer.write(frame)
            if display:
                cv2.imshow(window_name, frame)

            frames_processed += 1
            fps_feedback.update()  # notify stream handler thread of processing speed

            if frames_processed % 25 == 0:
                elapsed = time.perf_counter() - start
                print(f"processed {frames_processed} frames ({frames_processed / elapsed:.1f} FPS)")

            if max_frames is not None and frames_processed >= max_frames:
                print(f"Reached --max-frames={max_frames}")
                break

            if display and cv2.waitKey(1) == 27:
                break

    finally:
        total = time.perf_counter() - start
        stop_flag.set()
        decode_thread.join(timeout=5)
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    if total > 0:
        print(f"frames_processed={frames_processed}, total={total:.2f}s, fps={frames_processed / total:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream_url', type=str, help='URL of the stream to process', default='http://127.0.0.1:8080/stream')
    parser.add_argument('--object_model_path', type=str, help='Path to the object detection model (tensorrt engine)', default='models/rtdetrv2_r50vd_6x_coco_ema.trt')
    parser.add_argument('--keypoint_model_path', type=str, help='Path to the keypoint detection model (tensorrt engine)', default='models/keypoint_detector.trt')
    parser.add_argument('--output', type=str, default='output.mp4', help='Annotated output video; pass "" to disable')
    parser.add_argument('--output_fps', type=float, default=13.0, help='FPS of the output video')
    parser.add_argument('--no-display', action='store_true', help='Do not open a GUI window (use on headless WSL servers)')
    parser.add_argument('--max-frames', type=int, default=None, help='Stop after N processed frames')
    parser.add_argument('--minimap-width', type=int, default=200, help='Desired minimap width in pixels')
    parser.add_argument('--no-keypoints', action='store_true',
                        help='Do not draw the 14 court keypoints / court skeleton')
    parser.add_argument('--no-court-lines', action='store_true',
                        help='Draw numbered keypoints only, without connecting court lines')
    parser.add_argument('--no-court-status', action='store_true',
                        help='Do not draw the court-fit OK/FAILED status text')
    args = parser.parse_args()
    main(
        args.stream_url,
        args.object_model_path,
        args.keypoint_model_path,
        output_path=args.output,
        display=not args.no_display,
        max_frames=args.max_frames,
        minimap_width=args.minimap_width,
        output_fps=args.output_fps,
        draw_keypoints=not args.no_keypoints,
        draw_court_lines=not args.no_court_lines,
        show_court_status=not args.no_court_status,
    )
