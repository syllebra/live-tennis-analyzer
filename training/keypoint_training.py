import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models import mobilenet_v3_large, resnet50
from torchvision.transforms import v2
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class KeypointModel(nn.Module):
    def __init__(self, backbone, num_keypoints):
        super(KeypointModel, self).__init__()
        if isinstance(backbone, torchvision.models.resnet.ResNet):
            output_size = list(backbone.children())[-1].in_features
        else:    # mobilenet_v3_large
            output_size = list(backbone.children())[-1][0].in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1]) 
        # self.keypoint_head = nn.Sequential(
        #     nn.Linear(output_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_keypoints * 2)  
        # )
        self.keypoint_head = nn.Sequential(
            nn.Linear(output_size, num_keypoints * 2)
        )
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        keypoints = self.keypoint_head(features)
        return keypoints
    
    
class KeypointsDataset(Dataset):
    def __init__(self, img_dir, annos_json):
        self.img_dir = img_dir
        with open(annos_json, "r") as f:
            self.data = json.load(f)
        
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img = cv2.imread(f"{self.img_dir}/{item['id']}.png")
        h,w = img.shape[:2]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img)
        
        kps = np.array(item['kps']).flatten()
        kps = kps.astype(np.float32)
        
        kps[::2] *= 224.0 / w 
        kps[1::2] *= 224.0 / h 

        return img, kps

if __name__ == '__main__':
    
    
    backbone = mobilenet_v3_large(pretrained=True)
    # backbone = resnet50(weights='DEFAULT')
    num_keypoints = 14
    batch_size = 16
    epochs = 20
    device = 'cuda'

    model = KeypointModel(backbone, num_keypoints).to(device)

    train_data = KeypointsDataset('... ', '... ')
    val_data = KeypointsDataset('... ', '... ')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    writer = SummaryWriter(log_dir="...")
    
    for epoch in tqdm(range(epochs)):
        running_loss_train = 0.0
        running_loss_val = 0.0
        
        model.train()
        for imgs, kps in train_loader:
            imgs, kps = imgs.to(device), kps.to(device)
            preds = model(imgs)
            loss = criterion(preds, kps)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss_train += loss.item()
        
        model.eval()
        with torch.no_grad():
            for imgs, kps in val_loader:
                imgs, kps = imgs.to(device), kps.to(device)
                preds = model(imgs)
                loss = criterion(preds, kps)
                running_loss_val += loss.item()
                
        writer.add_scalar("Loss/epoch_train", running_loss_train / len(train_loader), epoch)
        writer.add_scalar("Loss/epoch_val", running_loss_val / len(val_loader), epoch)
            
    writer.close()
    torch.save(model.state_dict(), '...')
