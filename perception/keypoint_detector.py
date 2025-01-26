import cv2
from utils import TRTInferenceKeypoint, preprocess_keypoints

class KeypointDetector:
    def __init__(self, model_path):
        self.predictor = TRTInferenceKeypoint(model_path, 14)
        
    def predict(self, im):
        h, w = im.shape[:2]
        im = preprocess_keypoints(im)
        keypoints = self.predictor.predict(im)
        keypoints[::2] *= w / 224.0 #TODO as postprocess in utils
        keypoints[1::2] *= h / 224.0
        return keypoints
    
    @staticmethod
    def draw_keypoints(im, keypoints):
        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i]), int(keypoints[i+1])
            cv2.circle(im, (x, y), 5, (0, 255, 0), -1)
            # cv2.putText(im, str(i) + str(i+1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return im