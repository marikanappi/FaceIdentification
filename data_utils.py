import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler

class FaceDataset(Dataset):
    def __init__(self, features, labels):
        if isinstance(features, torch.Tensor):
            self.features = features
        else:
            self.features = torch.tensor(features, dtype=torch.float32)
        if isinstance(labels, torch.Tensor):
            self.labels = labels
        else:
            self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_data(csv_path, split=0.8):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Extract labels and features
    identities = df['identity'].values
    features = df.drop(columns=['identity', 'expression', 'filename']).values.astype('float32')

    # Label encoding
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(identities)

    # Feature scaling
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Shuffle and split
    total = len(labels)
    indices = torch.randperm(total)
    train_size = int(split * total)

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # Create datasets
    train_dataset = FaceDataset(features[train_idx], labels[train_idx])
    test_dataset = FaceDataset(features[test_idx], labels[test_idx])

    return train_dataset, test_dataset, label_encoder
