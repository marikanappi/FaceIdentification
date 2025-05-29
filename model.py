import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock35(nn.Module):
    def __init__(self, input_dim, scale=1.0):
        super().__init__()
        self.scale = scale
        self.branch0 = nn.Linear(input_dim, 32)
        self.branch1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32)
        )
        self.branch2 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32)
        )
        self.conv = nn.Linear(32 * 3, input_dim)

    def forward(self, x):
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        mixed = torch.cat([b0, b1, b2], dim=1)
        up = self.conv(mixed)
        return F.relu(x + self.scale * up)

class ReductionA(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.branch0 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True)
        )
        self.branch1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, output_dim)
        )
        self.branch2 = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        return b0 + b1 + b2  

class ResidualBlock17(nn.Module):
    def __init__(self, input_dim, scale=1.0):
        super().__init__()
        self.scale = scale
        self.branch0 = nn.Linear(input_dim, 64)
        self.branch1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64)
        )
        self.conv = nn.Linear(128, input_dim)

    def forward(self, x):
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        mixed = torch.cat([b0, b1], dim=1)
        up = self.conv(mixed)
        return F.relu(x + self.scale * up)

class FaceClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.5):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5)
        )

        self.block35 = ResidualBlock35(1024, scale=0.17)
        self.reduction_a = ReductionA(1024, 512)
        self.block17 = ResidualBlock17(512, scale=0.10)

        self.transition = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7)
        )

        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.block35(x)
        x = self.reduction_a(x)
        x = self.block17(x)
        x = self.transition(x)
        attention = self.attention(x)
        x = x * attention
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
