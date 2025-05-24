import torch.nn as nn
import torch.nn.functional as F
# Model definition
class FaceClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FaceClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)  # logits (no softmax)
