import cv2
import numpy as np

from typing import Iterator


def video_frame_generator(video_path: str) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {video_path}")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        yield frame
    
    cap.release()
    cv2.destroyAllWindows()