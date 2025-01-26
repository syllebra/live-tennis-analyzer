import time
import pycuda.autoinit 
import queue
import threading
import cv2
import argparse

from perception import ObjectDetector, KeypointDetector
from streaming import FPSMonitor, StreamHandler
from minimap import MiniMap


def main(srtream_url, object_model_path, keypoint_model_path):    
    
    frame_queue = queue.Queue()
    stop_flag = threading.Event()
    
    # handle stream, skip frames based on how fast processing is
    fps_feedback = FPSMonitor()
    stream_handler = StreamHandler(srtream_url, frame_queue, stop_flag, fps_feedback)
    
    # start stream handler thread
    decode_thread = threading.Thread(target=stream_handler.start)
    decode_thread.daemon = True
    decode_thread.start()
    
    object_detector = ObjectDetector(object_model_path)
    keypoint_detector = KeypointDetector(keypoint_model_path)
    minimap = MiniMap(position=(100, 250), width=200, alpha=0.6)
    
    start = time.perf_counter()
    frames_processed = 0
    writer = cv2.VideoWriter(
                    'output.mp4',
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    13,  # fps
                    (1920, 1080)
                )
    
    while True:      
        try:
            frame = frame_queue.get(block=True, timeout=1)
            
            keypoint_detections = keypoint_detector.predict(frame)
            object_detections = object_detector.predict(frame)
            
            frame = object_detector.draw_ball(frame, object_detections)
            frame = minimap.draw_minimap(frame)
            frame = minimap.draw_players(frame, object_detections, keypoint_detections, (181, 91, 241), (212, 245, 0)) 
            frame = minimap.draw_ball(frame, object_detections, keypoint_detections, (64, 228, 254))
            writer.write(frame)
            cv2.imshow('Prediction', frame)

            frames_processed += 1
            fps_feedback.update() # notify stream handler thread of processing speed
            
            if cv2.waitKey(1) == 27:
                break
            
        except queue.Empty:
            print("Frame buffer is empty, exiting...")
            break
        
    total = time.perf_counter() - start
    print(f"{frames_processed=}, {total=}, {frames_processed / total}")
    stop_flag.set()
    decode_thread.join()
    writer.release()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream_url', type=str, help='URL of the stream to process', default='http://127.0.0.1:8080/stream')
    parser.add_argument('--object_model_path', type=str, help='Path to the object detection model (tensorrt engine)', default='models/rtdetrv2_r34vd_120e_coco_ema.trt')
    parser.add_argument('--keypoint_model_path', type=str, help='Path to the keypoint detection model (tensorrt engine)', default='models/keypoint_detector.trt')
    args = parser.parse_args()
    main(args.stream_url, args.object_model_path, args.keypoint_model_path)

