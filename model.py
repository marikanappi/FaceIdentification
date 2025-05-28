import torch.nn as nn
import torch.nn.functional as F

class FaceClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.5, use_batch_norm=True):
        super(FaceClassifier, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        
        # Architettura più profonda con batch normalization
        layers = []
        
        # Layer 1
        layers.append(nn.Linear(input_dim, 512))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(512))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Layer 2
        layers.append(nn.Linear(512, 256))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate * 0.8))  # Dropout decrescente
        
        # Layer 3
        layers.append(nn.Linear(256, 128))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate * 0.6))
        
        # Layer 4
        layers.append(nn.Linear(128, 64))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(64))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate * 0.4))
        
        # Output layer
        layers.append(nn.Linear(64, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Inizializzazione dei pesi
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.classifier(x)