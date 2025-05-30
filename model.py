import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FaceClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),  # Aumentato la capacità
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.6),  # Aumentato dropout
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)