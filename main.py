import torch
from data_utils import load_and_preprocess_data
from train import train_model
from test import test_model
from model import FaceClassifier


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader, num_classes, le = load_and_preprocess_data("Dataset/Dataset_dist.csv")

    for X_batch, _ in train_loader:
        input_dim = X_batch.shape[1]
        break

    model = FaceClassifier(input_dim=input_dim, num_classes=num_classes)

    train_model(model, train_loader, val_loader, device=device, epochs=200)  # Passa anche val_loader

    test_model(model, test_loader, device=device, le=le)

if __name__ == "__main__":
    main()