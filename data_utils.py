import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

def load_and_preprocess_data(csv_path, test_size=0.2, val_size=0.1, batch_size=32):
    df = pd.read_csv(csv_path)
    df = df.dropna()

    le = LabelEncoder()
    df['identity'] = le.fit_transform(df['identity'])
    num_classes = len(le.classes_)

    X = df.drop(columns=['identity', 'expression']).values
    y = df['identity'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Prima split train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y)

    # Poi split train vs val
    val_relative_size = val_size / (1 - test_size)  # calcolo proporzione rispetto a train+val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_relative_size, random_state=42, stratify=y_trainval)

    # Conversione in tensori
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Dataset e DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes, le
