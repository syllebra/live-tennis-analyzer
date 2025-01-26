import torch
import torch.nn as nn 
import torchvision.transforms.functional as F
import numpy as np

def filter_detections(trt_output, score_threshold, class_ids):
    boxes, scores, labels = trt_output['boxes'].squeeze(), trt_output['scores'].squeeze(), trt_output['labels'].squeeze()
        
    valid_indices = scores > score_threshold 
    boxes = torch.tensor(boxes[valid_indices])
    scores = torch.tensor(scores[valid_indices])
    labels = torch.tensor(labels[valid_indices])
    
    valid_indices = torch.isin(labels, torch.tensor(class_ids).to(labels.device))
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    labels = labels[valid_indices]
    
    return boxes, scores, labels

def preprocess_detection(im): #TODO: in numpy?
    im_array = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
    im_array = F.resize(im_array, (640, 640))[None].numpy()
    h, w, _ = im.shape
    orig_shape = torch.tensor([w, h])[None].numpy()
    return im_array, orig_shape
    
def postprocess_detections_for_tracker(prediction, classes_to_track, threshold=0.6):
    boxes, scores, labels = filter_detections(prediction, threshold, classes_to_track)
    tracker_input = torch.cat((boxes, scores.unsqueeze(1), labels.unsqueeze(1)), dim=1)
    return tracker_input.to('cpu')

# def remove_non_players(boxes, scores, labels, keypoints):
#     closest_distances = []
    
#     for idx, box in enumerate(boxes):
#         mid_x = (box[0] + box[2]) / 2
#         mid_y = (box[1] + box[3]) / 2
        
#         # find closest keypoint
#         min_distance = float('inf')
#         for i in range(0, len(keypoints), 2):
#             x, y = int(keypoints[i]), int(keypoints[i+1])
#             distance = np.sqrt((mid_x - x) ** 2 + (mid_y - y) ** 2)
#             if distance < min_distance:
#                 min_distance = distance
    
#         closest_distances.append((min_distance, idx))
        
#     closest_distances.sort(key=lambda x: x[0])
#     to_keep = torch.tensor([closest_distances[0][1], closest_distances[1][1]])
#     # print(to_keep)
#     # print(scores)
#     boxes = boxes[to_keep]
#     scores = scores[to_keep]
#     labels = labels[to_keep]
#     # boxes = torch.tensor([boxes[i] for i in to_keep])
#     # scores = torch.tensor([scores[i] for i in to_keep])
#     # labels = torch.tensor([labels[i] for i in to_keep])
#     print(boxes.shape, scores.shape, labels.shape)
    
#     return boxes, scores, labels

def remove_non_players(boxes, scores, labels, keypoints):
    to_keep = []
    for kp in ((24, 25), (26, 27)):
        distances = []
        for idx, box in enumerate(boxes):
            mid_x = (box[0] + box[2]) / 2
            mid_y = (box[1] + box[3]) / 2
            x, y = int(keypoints[kp[0]]), int(keypoints[kp[1]])
            distance = np.sqrt((mid_x - x) ** 2 + (mid_y - y) ** 2)
            distances.append(distance)
        if not to_keep:
            to_keep.append(np.argmin(distances))
        else:
            to_keep.append(np.argmin([dist for i, dist in enumerate(distances) if i not in to_keep]))
    to_keep = torch.tensor(to_keep)
    boxes = boxes[to_keep]
    scores = scores[to_keep]
    labels = labels[to_keep]
    return boxes, scores, labels   
    

# for pytorch inference
def load_model(config: str, weights: str, device: str) -> nn.Module:
    cfg = YAMLConfig(config, resume=weights) # import from RT-DETR repository
    
    checkpoint = torch.load(weights, map_location='cpu') 
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
        
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    return Model().to(device)



   