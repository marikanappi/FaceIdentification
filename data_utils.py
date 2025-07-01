import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import time

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

def load_data(csv_path, train_split=0.7, val_split=0.15, test_split=0.15, seed=None):
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    df = pd.read_csv(csv_path)
    identities = df['identity'].values
    features = df.drop(columns=['identity', 'expression', 'filename']).values.astype('float32')

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(identities)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    joblib.dump(scaler, "scaler.pkl")

    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = FaceDataset(features, labels)

    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    current_seed = seed if seed is not None else int(time.time())
    generator = torch.Generator().manual_seed(current_seed)
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    return train_dataset, val_dataset, test_dataset, label_encoder