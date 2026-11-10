import numpy as np
import cv2

from utils import filter_detections

class MiniMap:
    court_length = 23.77  # Full length
    court_width = 10.97    # Full width
    service_line_dist = 6.4  # Distance from net to service line
    alley_width = 1.37  # Alley width (doubles minus singles)
    
    def __init__(self, position: tuple[int, int], width: int, alpha: float):
        self.alpha = alpha
        self.position = position
        self.width = width
        self.scale = self.width / self.court_width
        self.height = int(self.court_length * self.scale)
        
        self.keypoints = self.make_keypoint_dict()
        self.minimap = self.make_minimap()
        
    def to_pixels(self, x: float, y: float) -> tuple[int, int]:
        """Convert meters to pixels"""
        return int(x * self.scale), int(y * self.scale)
    
    def make_keypoint_dict(self) -> dict:
        kps = {}
        net_y = self.court_length / 2
        
        # general
        kps['top_left'] = self.to_pixels(0, 0)
        kps['bottom_left'] = self.to_pixels(0, self.court_length)
        kps['top_right'] = self.to_pixels(self.court_width, 0)
        kps['bottom_right'] = self.to_pixels(self.court_width, self.court_length)
        kps['service_top'] = self.to_pixels(self.court_width / 2, net_y - self.service_line_dist)
        kps['service_bottom'] = self.to_pixels(self.court_width / 2, net_y + self.service_line_dist)
        kps['base_top'] = self.to_pixels(self.court_width / 2, 0)
        kps['base_bottom'] = self.to_pixels(self.court_width / 2, self.court_length)
        
        # for drawing
        kps['net_start'] = self.to_pixels(0, net_y)
        kps['net_end'] = self.to_pixels(self.court_width, net_y)
        kps['service_top_left'] = self.to_pixels(self.alley_width, net_y - self.service_line_dist)
        kps['service_top_right'] = self.to_pixels(self.court_width - self.alley_width, net_y - self.service_line_dist)
        kps['service_bottom_left'] = self.to_pixels(self.alley_width, net_y + self.service_line_dist)
        kps['service_bottom_right'] = self.to_pixels(self.court_width - self.alley_width, net_y + self.service_line_dist)
        kps['side_top_left'] = self.to_pixels(self.alley_width, 0)
        kps['side_top_right'] = self.to_pixels(self.court_width - self.alley_width, 0)
        kps['side_bottom_left'] = self.to_pixels(self.alley_width, self.court_length)
        kps['side_bottom_right'] = self.to_pixels(self.court_width - self.alley_width, self.court_length)
        
        return kps
        
    def make_minimap(self, line_thickness: int = 2) -> np.ndarray:
        white = (255, 255, 255)
        kps = self.keypoints
        t = line_thickness
        
        minimap = np.zeros((self.height, self.width, 3), dtype="uint8")
        
        # court boundary
        cv2.rectangle(minimap, kps['top_left'], kps['bottom_right'], white, t)

        # net
        cv2.line(minimap, kps['net_start'], kps['net_end'], white, t)
        
        # service lines
        cv2.line(minimap, kps['service_top_left'], kps['service_top_right'], white, t)
        cv2.line(minimap, kps['service_bottom_left'], kps['service_bottom_right'], white, t)
        
        # center service line
        cv2.line(minimap, kps['service_top'], kps['service_bottom'], white, t)
        
        # side lines
        cv2.line(minimap, kps['side_top_left'], kps['side_bottom_left'], white, t)
        cv2.line(minimap, kps['side_top_right'], kps['side_bottom_right'], white, t)
        
        return minimap
    
    def draw_minimap(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        x, y = self.position
        
        if y + self.height > frame.shape[0] or x + self.width > frame.shape[1]:
            raise ValueError("The minimap exceeds the frame boundaries at the given position.")
        
        roi = frame[y:y + self.height, x:x + self.width]
        blended = cv2.addWeighted(self.minimap, self.alpha, roi, 1 - self.alpha, 0)
        overlay[y:y + self.height, x:x + self.width] = blended
        
        return overlay
    
    def find_homography(self, image_kps: np.ndarray) -> np.ndarray:
        """Find homography matrix to map keypoints from image to minimap"""
        image_kps = image_kps.reshape(-1, 2)
        image_kps = image_kps[[0, 1, 3, 2, 12, 13]].astype(np.float32)
        
        kps = self.keypoints
        minimap_kps = np.array([
            kps['top_left'],
            kps['top_right'],
            kps['bottom_right'],
            kps['bottom_left'],
            kps['service_top'],
            kps['service_bottom']
        ])
        minimap_kps = minimap_kps + np.array(self.position)
        
        H, _ = cv2.findHomography(image_kps, minimap_kps)
        return H
    
    def draw_players(self, frame: np.ndarray, detections: dict, image_kps: np.ndarray, color_top: tuple, color_bottom: tuple) -> np.ndarray:
        """Draw player positions on the minimap"""
        frame = frame.copy()
        H = self.find_homography(image_kps)
        if H is None:
            return frame
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return frame
        
        # find position of all people on minimap
        minimap_person_positions = []
        player_boxes, _, _ = filter_detections(detections, 0.6, [0])
        for box in player_boxes:
            point = (int((box[0] + box[2]) / 2), int(box[3]))
            point = np.array([[point]], dtype=np.float32)  # Shape (1, 1, 2)
            transformed_point = cv2.perspectiveTransform(point, H)
            minimap_person_positions.append((int(transformed_point[0][0][0]), int(transformed_point[0][0][1])))
            
        # filter for players and draw on minimap
        ref_point_top = np.array(self.keypoints['base_top']) + np.array(self.position)
        ref_point_bottom = np.array(self.keypoints['base_bottom']) + np.array(self.position)
        ind_top, ind_bottom = None, None
        player_top, player_bottom = None, None
        if minimap_person_positions:
            dist_top = []
            dist_bottom = []    
            for player in minimap_person_positions:
                dist_top.append(np.linalg.norm(np.array(player) - ref_point_top))
                dist_bottom.append(np.linalg.norm(np.array(player) - ref_point_bottom))
            
            if dist_top:
                ind_top = np.argmin(dist_top)
                player_top = minimap_person_positions[ind_top]
                cv2.circle(frame, player_top, radius=20, color=color_top, thickness=-1)
                dist_bottom[ind_top] = float('inf')
            if dist_bottom:
                ind_bottom = np.argmin(dist_bottom)
                player_bottom = minimap_person_positions[ind_bottom]
                cv2.circle(frame, player_bottom, radius=20, color=color_bottom, thickness=-1)
                
        if player_top is not None:
            frame = self.draw_player_circle(frame, player_top, player_boxes, ind_top, H_inv, color_top)
        if player_bottom is not None:
            frame = self.draw_player_circle(frame, player_bottom, player_boxes, ind_bottom, H_inv, color_bottom)
                          
        return frame
    
    @staticmethod
    def draw_player_circle(frame, player_xy_minimap, player_boxes, ind, H_inv, color):
        overlay = frame.copy()
        center = np.array([[(player_xy_minimap[0], player_xy_minimap[1]-10)]], dtype=np.float32)
        circle_points = []
        x1, _, x2, _ = player_boxes[ind]
        radius = min(int((x2 - x1)/2) - 10, 20)

        # Create circle points
        for angle in range(0, 360, 10):
            x = center[0][0][0] + radius * np.cos(np.radians(angle))
            y = center[0][0][1] + radius * np.sin(np.radians(angle))
            circle_points.append([[x, y]])
            
        circle_points = np.array(circle_points, dtype=np.float32)
        transformed_circle = cv2.perspectiveTransform(circle_points, H_inv)

        # Draw filled polygon
        pts = transformed_circle.reshape((-1,1,2)).astype(np.int32)
        cv2.fillPoly(overlay, [pts], color)

        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame
    
    def draw_ball(self, frame: np.ndarray, detections: dict, image_kps: np.ndarray, color: tuple) -> np.ndarray:
        """Draw ball position on the minimap"""
        frame = frame.copy()
        H = self.find_homography(image_kps)

        if not hasattr(self, 'ball_tracker'):
            self.ball_tracker = BallTracker()

        # Get ball detection
        ball_detected = False
        x, y = np.nan, np.nan
        ball_boxes, _, _ = filter_detections(detections, 0.4, [32])
        if H is not None and len(ball_boxes) > 0:
            ball = ball_boxes[0]
            point = (int((ball[0] + ball[2]) / 2), int((ball[3] + ball[1]) / 2))
            point = np.array([[point]], dtype=np.float32)
            transformed_point = cv2.perspectiveTransform(point, H)
            x, y = int(transformed_point[0][0][0]), int(transformed_point[0][0][1])
            ball_detected = True
            # cv2.circle(frame, (int(x), int(y)), radius=5, color=color, thickness=-1)

        # Update tracker and get prediction
        self.ball_tracker.update(x, y, detected=ball_detected)
        if np.isnan(x) or np.isnan(y):
            x, y = self.ball_tracker.predict_next_position()
        
        if x is not None and y is not None:
            cv2.circle(frame, (int(x), int(y)), radius=5, color=color, thickness=-1)
        
        return frame
    
    
class BallTracker:
    def __init__(self, max_frames_to_interpolate=5):
        self.positions = []
        self.velocities = []
        self.accelerations = []
        self.confidence = 1.0
        self.confidence_threshold = 0.2
        self.last_detection_frame = 0
        self.current_frame = 0
        self.max_frames_to_interpolate = max_frames_to_interpolate
        self.total_preds = 0
        
        
    def calculate_weighted_velocity(self):
        if len(self.velocities) < 2:
            return self.velocities[-1] if self.velocities else np.array([0, 0])
            
        weights = np.linspace(0.5, 1.0, len(self.velocities)) * self.confidence
        weighted_vel = np.average(self.velocities, weights=weights, axis=0)
        return weighted_vel
        
    def update_acceleration(self):
        if len(self.velocities) >= 2:
            acc = self.velocities[-1] - self.velocities[-2]
            self.accelerations.append(acc)
            if len(self.accelerations) > 5:
                self.accelerations.pop(0)
    
    def update(self, x, y, detected=True):
        self.current_frame += 1
        self.confidence = 1.0 if detected else max(0.1, self.confidence * 0.9)
        
        if not (np.isnan(x) or np.isnan(y)):
            pos = np.array([x, y])
            if self.positions:
                # Smooth position using weighted average with previous
                alpha = 0.9 if detected else 0.1
                pos = alpha * pos + (1-alpha) * self.positions[-1]
            
            self.positions.append(pos)
            if detected:
                self.last_detection_frame = self.current_frame
                
            if len(self.positions) >= 2:
                velocity = (self.positions[-1] - self.positions[-2])
                self.velocities.append(velocity)
                self.update_acceleration()
                
            if len(self.positions) > 5:
                self.positions.pop(0)
            if len(self.velocities) > 5:
                self.velocities.pop(0)
    
    def predict_next_position(self):
        frames_since_detection = self.current_frame - self.last_detection_frame
        
        # Stop predicting if confidence too low
        if self.confidence < self.confidence_threshold:
            return None, None

        if not self.positions or frames_since_detection > self.max_frames_to_interpolate:
            return None, None
            
        weighted_vel = self.calculate_weighted_velocity()
        avg_acc = np.mean(self.accelerations, axis=0) if self.accelerations else np.array([0, 0])
        
        last_pos = self.positions[-1]
        decay = (0.95 ** frames_since_detection) * self.confidence
        
        predicted_pos = last_pos + (weighted_vel * decay) + (0.5 * avg_acc * decay**2)
        self.total_preds += 1
        
        return predicted_pos[0], predicted_pos[1]