import torch
from torch.utils.data import DataLoader
from torch import nn

from data_utils import load_data
from model import FaceClassifier

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def run_test():
    csv_path = 'Dataset_facial_features_standard.csv'
    batch_size = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    _, test_dataset, label_encoder = load_data(csv_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = test_dataset[0][0].shape[0]
    num_classes = len(label_encoder.classes_)

    # Load best model
    model = FaceClassifier(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"🧪 Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")
