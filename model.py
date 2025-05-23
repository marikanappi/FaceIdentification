import torch.nn as nn

class LandmarkClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LandmarkClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
