import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns

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

def evaluate(model, dataloader, criterion, device, threshold=0.5, return_predictions=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    # Timing inference
    inference_times = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(X)
            end_time = time.time()
            
            # Store per-sample inference time
            batch_inference_time = (end_time - start_time) / X.size(0)
            inference_times.append(batch_inference_time)
            
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
            
            # Store predictions and labels for metrics calculation
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    avg_inference_time = np.mean(inference_times)
    
    # Calculate Precision, Recall, F1
    # Remove unknown predictions (-1) for proper metric calculation
    valid_mask = np.array(all_predictions) != -1
    valid_predictions = np.array(all_predictions)[valid_mask]
    valid_labels = np.array(all_labels)[valid_mask]
    
    if len(valid_predictions) > 0:
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_labels, valid_predictions, average='macro', zero_division=0
        )
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0
    
    if return_predictions:
        return avg_loss, accuracy, precision, recall, f1, avg_inference_time, all_labels, all_predictions
    else:
        return avg_loss, accuracy, precision, recall, f1, avg_inference_time

def run_test():
    csv_path = 'dataset_features_final.csv'
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

    # Comprehensive evaluation
    test_loss, test_acc, test_precision, test_recall, test_f1, test_inf_time, y_true, y_pred = evaluate(
        model, test_loader, criterion, device, return_predictions=True
    )
    
    print("="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"🧪 Test Loss: {test_loss:.4f}")
    print(f"🧪 Test Accuracy: {test_acc*100:.2f}%")
    print(f"🧪 Test Precision: {test_precision:.3f}")
    print(f"🧪 Test Recall: {test_recall:.3f}")
    print(f"🧪 Test F1-Score: {test_f1:.3f}")
    print(f"🧪 Average Inference Time: {test_inf_time*1000:.2f}ms")
    print("="*60)

    # Create test confusion matrix
    valid_mask = np.array(y_pred) != -1
    valid_y_true = np.array(y_true)[valid_mask]
    valid_y_pred = np.array(y_pred)[valid_mask]
    
    if len(valid_y_true) > 0:
        cm = confusion_matrix(valid_y_true, valid_y_pred)
        class_names = label_encoder.classes_
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Test Set - Confusion Matrix (Interpretability)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("test_confusion_matrix.png", dpi=300, bbox_inches='tight')
        print("📊 Test confusion matrix saved to test_confusion_matrix.png")

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