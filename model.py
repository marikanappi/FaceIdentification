import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceClassifier(nn.Module):
    """
    Modello ottimizzato per face identification con architettura profonda,
    residual connections, attention mechanism e tecniche avanzate di regolarizzazione
    """
    
    def __init__(self, input_dim, num_classes, dropout_rate=0.5):
        super(FaceClassifier, self).__init__()
        
        # Feature extraction iniziale più ampia
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Primo blocco residuale
        self.res_block1 = self._make_residual_block(1024, 1024, dropout_rate * 0.6)
        
        # Secondo blocco con riduzione dimensionale
        self.transition1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7)
        )
        
        # Terzo blocco residuale
        self.res_block2 = self._make_residual_block(512, 512, dropout_rate * 0.8)
        
        # Quarto blocco con ulteriore riduzione
        self.transition2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.9)
        )
        
        # Attention mechanism per pesare le features importanti
        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.Sigmoid()
        )
        
        # Blocco finale di classificazione
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            
            # Output layer senza attivazione (CrossEntropy la applicherà)
            nn.Linear(64, num_classes)
        )
        
        # Inizializzazione ottimale dei pesi
        self._initialize_weights()
    
    def _make_residual_block(self, in_features, out_features, dropout_rate):
        """Crea un blocco residuale"""
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
    
    def _initialize_weights(self):
        """Inizializzazione ottimale dei pesi"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He initialization per ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction iniziale
        x = self.input_layer(x)
        
        # Primo blocco residuale con skip connection
        identity1 = x
        residual1 = self.res_block1(x)
        x = F.relu(residual1 + identity1)
        
        # Transizione con riduzione dimensionale
        x = self.transition1(x)
        
        # Secondo blocco residuale con skip connection
        identity2 = x
        residual2 = self.res_block2(x)
        x = F.relu(residual2 + identity2)
        
        # Seconda transizione
        x = self.transition2(x)
        
        # Attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights  # Element-wise multiplication
        
        # Classificazione finale
        output = self.classifier(x)
        
        return output