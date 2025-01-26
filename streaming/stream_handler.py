import cv2 
import time
from collections import deque

class StreamHandler:
    def __init__(self, stream_url, frame_queue, stop_flag, fps_feedback):
        self.stream_url = stream_url
        self.frame_queue = frame_queue
        self.stop_flag = stop_flag
        self.fps_feedback = fps_feedback
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            print("Error: Unable to open RTSP stream")
            self.stop_flag.set()
            return

        last_decode_time = time.time()
        while not self.stop_flag.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Unable to read frame from stream")
                self.stop_flag.set()
                break
            
            # check if decoding can be skipped based on fps feedback of processing thread
            frame_delay = self.fps_feedback.get_delay()
            if time.time() - last_decode_time < frame_delay and self.frame_queue.qsize() > 5:
                # print(f"skipping frame, {frame_delay=}")
                continue
                   

            self.frame_queue.put(frame)
            last_decode_time = time.time()

        self.cap.release()
        print("Decoding thread stopped")
        
        
        
class FPSMonitor:
    def __init__(self, window_size=10):
        self.frame_count = 0
        
        self.window = deque()
        self.window_sum = 0
        self.window_size = window_size
        self.latest_update = time.time()
        
    def update(self):
        current = time.time()
        delay = current - self.latest_update
        
        if len(self.window) > self.window_size:
            old = self.window.popleft()
            self.window_sum -= old
        
        self.window.append(delay)
        self.window_sum += delay
        
        self.latest_update = current
        self.frame_count += 1
        
    def get_delay(self):
        return self.window_sum / len(self.window) if len(self.window) > 0 else 0
        