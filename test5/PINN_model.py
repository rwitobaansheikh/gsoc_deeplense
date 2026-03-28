import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

class GravitationalLensingLayer(nn.Module):
    def __init__(self, in_channels=3, img_size=64, hidden_dim=16, einstein_radius_init=0.3):
        super().__init__()
        self.img_size = img_size
        self.log_einstein_radius = nn.Parameter(
            torch.tensor(float(torch.tensor(einstein_radius_init).log()))
        )
        self.deflection_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
            nn.Tanh()
        )
        self.register_buffer('theta_grid', self._make_theta_grid(img_size))

    @staticmethod
    def _make_theta_grid(size):
        lin = torch.linspace(-1.0, 1.0, size)
        gy, gx = torch.meshgrid(lin, lin, indexing='ij')
        return torch.stack([gx, gy], dim=-1).unsqueeze(0)

    def _sis_deflection(self, theta):
        einstein_r = self.log_einstein_radius.exp()
        norm = theta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return einstein_r * theta / norm

    def forward(self, x):
        B = x.size(0)
        theta = self.theta_grid.expand(B, -1, -1, -1)
        alpha_sis = self._sis_deflection(theta)
        alpha_nn = self.deflection_net(x).permute(0, 2, 3, 1) * 0.1
        beta = theta - (alpha_sis + alpha_nn)
        return F.grid_sample(x, beta, mode='bilinear', padding_mode='border', align_corners=True)


class PINNLensingClassifier(nn.Module):
    def __init__(self, num_classes=2, img_size=64, dropout=0.5):
        super().__init__()
        self.lensing_layer = GravitationalLensingLayer(
            in_channels=3, img_size=img_size, hidden_dim=16, einstein_radius_init=0.3
        )
        backbone = models.resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.lensing_layer(x)
        features = self.backbone(x)
        return self.classifier(features)

    def get_einstein_radius(self):
        return self.lensing_layer.log_einstein_radius.exp().item()
    
class LensingNpyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir=root_dir
        self.transform=transform
        self.data=[]
        self.classes=sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for idx, cl in enumerate(self.classes):
            class_path =os.path.join(root_dir, cl)
            if not os.path.isdir(class_path):
                continue
            for f in os.listdir(class_path):
                if f.endswith('.npy'):
                    self.data.append((os.path.join(class_path,f), idx))
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = np.load(img_path).astype(np.float32)
        
        # Ensure (C, H, W) format
        if image.ndim == 2:
            image = image[np.newaxis, ...]       # (H, W) → (1, H, W)
        elif image.ndim == 3 and image.shape[-1] in (1, 3, 4):
            image = np.transpose(image, (2, 0, 1))  # (H, W, C) → (C, H, W)
        
        image = torch.from_numpy(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
