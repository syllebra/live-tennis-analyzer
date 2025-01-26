import torch
import cv2
import numpy as np
from typing import Optional

from utils import TRTInferenceObject, filter_detections, preprocess_detection, postprocess_detections_for_tracker, remove_non_players

class ObjectDetector:
    def __init__(self, model_path):
        
        self.predictor = TRTInferenceObject(model_path)
        self.tracker = BYTETracker()
    
    def predict(self, im):
        im, orig_shape = preprocess_detection(im)
        return self.predictor.predict({'images': im, 'orig_target_sizes': orig_shape})
    
    def track_players(self, prediction, frame_id, keypoint_detections):
        # tracker_input = postprocess_detections_for_tracker(trt_output, (0))
        boxes, scores, labels = filter_detections(prediction, 0.6, (0))
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
        boxes, scores, labels = filter_detections(prediction, threshold, (32))
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
    
    