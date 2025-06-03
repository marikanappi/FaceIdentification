import torch
from torch.utils.data import DataLoader
from torch import nn

from data_utils import load_data
from model import FaceClassifier

def predict_with_unknown(model, X, device, threshold=0.5, label_encoder=None):
    model.eval()
    X = X.to(device)

    with torch.no_grad():
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)
        max_probs, preds = torch.max(probs, dim=1)

        results = []
        for prob, pred in zip(max_probs, preds):
            if prob.item() < threshold:
                results.append('unknown')
            else:
                if label_encoder is not None:
                    results.append(label_encoder.inverse_transform([pred.item()])[0])
                else:
                    results.append(str(pred.item()))

    return results

def evaluate(model, dataloader, criterion, device, threshold=0.5):
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

            # Calcolo probabilità
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            # Applico soglia
            unknown_mask = max_probs < threshold
            preds[unknown_mask] = -1  # Etichetta speciale per "unknown"

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

    # Test predizioni con unknown su un batch
    test_batch = next(iter(test_loader))
    X_batch, y_batch = test_batch
    y_batch = y_batch.cpu().numpy()

    # Predizioni con unknown
    threshold = 0.6
    preds_with_unknown = predict_with_unknown(model, X_batch, device, threshold=threshold, label_encoder=label_encoder)

    # Etichette vere (stringhe)
    true_labels = label_encoder.inverse_transform(y_batch)

    print("\nEsempio predizioni (con unknown):")
    for true_label, pred in zip(true_labels, preds_with_unknown):
        print(f"Vero: {true_label} | Predetto: {pred}")
