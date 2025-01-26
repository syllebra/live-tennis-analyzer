import numpy as np
import cv2
import torch
from torchvision.transforms import v2


def preprocess_keypoints(img):
    transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms(img)
    return img.numpy()
        
        
# def create_gaussian_heatmap(size, center, sigma=2):
#     x, y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
#     d = (x - center[0])**2 + (y - center[1])**2
#     return np.exp(-d / (2 * sigma**2))