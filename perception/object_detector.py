import torch
import cv2
import numpy as np
from typing import Optional

from utils import TRTInferenceObject, filter_detections, preprocess_detection, postprocess_detections_for_tracker, remove_non_players


class SimpleIoUTracker:
    """Very small IoU tracker used by :meth:`ObjectDetector.track_players`.

    The original project referenced ``BYTETracker`` but did not ship it.  The
    tracking path is not used by ``main.py``; this implementation keeps that
    optional API functional without pulling in another dependency.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []  # [x1, y1, x2, y2, track_id, last_frame]

    @staticmethod
    def _iou(box_a, box_b) -> float:
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
        area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections, frame_id):
        detections = np.asarray(detections)
        output = []
        assigned = set()
        for det in detections:
            best_index, best_iou = None, self.iou_threshold
            for index, track in enumerate(self.tracks):
                if index in assigned:
                    continue
                iou = self._iou(det[:4], track[:4])
                if iou >= best_iou:
                    best_index, best_iou = index, iou
            if best_index is None:
                track = [*det[:4], len(self.tracks) + len(output), frame_id]
                self.tracks.append(track)
                best_index = len(self.tracks) - 1
            else:
                self.tracks[best_index][:4] = det[:4]
                self.tracks[best_index][5] = frame_id
            assigned.add(best_index)
            track = self.tracks[best_index]
            output.append([*track[:4], track[4]])

        self.tracks = [t for t in self.tracks if frame_id - t[5] <= self.max_age]
        return np.asarray(output, dtype=np.float32)


class ObjectDetector:
    def __init__(self, model_path):
        
        self.predictor = TRTInferenceObject(model_path)
        self.tracker = SimpleIoUTracker()
    
    def predict(self, im):
        im, orig_shape = preprocess_detection(im)
        return self.predictor.predict({'images': im, 'orig_target_sizes': orig_shape})
    
    def track_players(self, prediction, frame_id, keypoint_detections):
        # tracker_input = postprocess_detections_for_tracker(trt_output, (0))
        boxes, scores, labels = filter_detections(prediction, 0.6, [0])
        if scores.shape[0] > 2:
            boxes, scores, labels = remove_non_players(boxes, scores, labels, keypoint_detections)
        # print(boxes.shape, scores.shape, labels.shape)
        tracker_input = torch.cat((boxes, scores.unsqueeze(1), labels.unsqueeze(1)), dim=1)
        return self.tracker.update(tracker_input, frame_id)
    
    @staticmethod
    def draw_players(im, tracked_objects):
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj[:5]
            x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return im
    
    @staticmethod
    def draw_ball(im, prediction, threshold=0.5):
        boxes, scores, labels = filter_detections(prediction, threshold, [32])
        for j, box in enumerate(boxes):
            # Calculate top center point of ball
            top_x = (box[0] + box[2]) / 2
            top_y = box[1] - 3 
            
            # Define triangle points
            triangle_height = 12 
            triangle_width = 10   
            triangle_points = np.array([
                [top_x - triangle_width/2, top_y - triangle_height], 
                [top_x + triangle_width/2, top_y - triangle_height], 
                [top_x, top_y] 
            ], np.int32)
            
            # Draw filled triangle
            cv2.fillPoly(im, [triangle_points.reshape((-1,1,2))], (0, 0, 255))
            # cv2.putText(im, f'Score: {scores[j]:.2f}', (int(top_x), int(top_y - triangle_height - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return im
    
    