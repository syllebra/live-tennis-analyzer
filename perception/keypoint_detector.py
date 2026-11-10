import cv2
import numpy as np

from utils import TRTInferenceKeypoint, preprocess_keypoints


class KeypointDetector:
    """Tennis court keypoint detector (14 points, 28 coordinates).

    Keypoint order from the TennisCourtDetector dataset / model card:

        0, 2   left outer sideline
        1, 3   right outer sideline
        0, 1   far baseline
        2, 3   near baseline
        4, 5   left singles sideline
        6, 7   right singles sideline
        8, 9   far service line
        10, 11 near service line
        12, 13 centre service line
    """

    COURT_SEGMENTS = (
        (0, 2),   # left outer sideline
        (1, 3),   # right outer sideline
        (0, 1),   # far baseline
        (2, 3),   # near baseline
        (4, 5),   # left singles sideline
        (6, 7),   # right singles sideline
        (8, 9),   # far service line
        (10, 11), # near service line
        (12, 13), # centre service line
    )

    def __init__(self, model_path):
        self.predictor = TRTInferenceKeypoint(model_path, 14)

    def predict(self, im):
        h, w = im.shape[:2]
        im = preprocess_keypoints(im)
        keypoints = self.predictor.predict(im)
        keypoints[::2] *= w / 224.0  # TODO as postprocess in utils
        keypoints[1::2] *= h / 224.0
        return keypoints

    @staticmethod
    def draw_keypoints(
        im,
        keypoints,
        draw_court: bool = True,
        labels: bool = True,
        court_color=(0, 255, 255),
    ):
        """Draw the 14 court keypoints and the court skeleton on an image.

        This is deliberately explicit so a bad court fit is obvious when
        reviewing an output video.  Set ``draw_court=False`` to show only the
        numbered keypoints.
        """
        points_array = np.asarray(keypoints, dtype=np.float32).reshape(-1)
        if points_array.size < 28 or not np.isfinite(points_array[:28]).all():
            return im

        points = [
            (int(round(points_array[i])), int(round(points_array[i + 1])))
            for i in range(0, 28, 2)
        ]

        if draw_court:
            overlay = im.copy()
            for start, end in KeypointDetector.COURT_SEGMENTS:
                cv2.line(overlay, points[start], points[end], court_color, 2, cv2.LINE_AA)
            im = cv2.addWeighted(overlay, 0.55, im, 0.45, 0)

        # BGR colours: outer corners green, singles lines orange, service lines magenta.
        groups = ((0, 4, (0, 255, 0)), (4, 8, (0, 128, 255)), (8, 14, (255, 0, 255)))

        for index, (x, y) in enumerate(points):
            color = next(group_color for start, end, group_color in groups if start <= index < end)
            cv2.circle(im, (x, y), 6, color, -1, cv2.LINE_AA)
            cv2.circle(im, (x, y), 6, (0, 0, 0), 1, cv2.LINE_AA)
            if labels:
                label = str(index)
                org = (x + 8, y - 8)
                # Black outline makes the number readable on any court colour.
                cv2.putText(im, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(im, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return im
