import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import numpy as np
from data_utils import load_data
from model import FaceClassifier
from train import train
from test import evaluate
from sklearn.metrics import confusion_matrix
import seaborn as sns

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    # Config
    csv_path = 'Dataset2/Dataset_facial_features_standard.csv'
    batch_size = 32
    lr = 1e-3
    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_dataset, test_dataset, label_encoder = load_data(csv_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Input/output sizes
    input_dim = train_dataset[0][0].shape[0]
    num_classes = len(label_encoder.classes_)

    # Model, loss, optimizer
    model = FaceClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Metrics to track
    train_losses, train_accuracies = [], []
    test_accuracies = []

    patience = 20
    patience_counter = 0
    best_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Test Acc: {test_acc*100:.2f}%")

        # Early stopping logic
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Ripristino modello migliore
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final test accuracy
    final_test_acc = best_acc
    print(f"Final Test Accuracy (best): {final_test_acc * 100:.2f}%")

    # Plot
    plt.figure(figsize=(10, 4))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(test_accuracies, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("📊 Training curves saved to training_curves.png")

if __name__ == "__main__":
    main()
