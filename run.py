import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import random
import numpy as np
import joblib
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import time

from data_utils import load_data
from model import FaceClassifier
from train import train
from test import evaluate

def run_training():
    # === Config ===
    csv_path = 'dataset_features_final.csv'
    batch_size = 32
    lr = 1e-3
    num_epochs = 200
    patience = 20
    seed = 42
    val_split = 0.2

    '''# === Seed per riproducibilità ===
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False'''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Load data ===
    full_dataset, _, label_encoder = load_data(csv_path)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_dataset[0][0].shape[0]
    num_classes = len(label_encoder.classes_)
    print(f"Input dimension: {input_dim}, Number of classes: {num_classes}")
    model = FaceClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accuracies = [], []
    val_accuracies, val_precisions, val_recalls, val_f1s = [], [], [], []
    val_inference_times = []
    best_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_precision, val_recall, val_f1, val_inf_time = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        val_inference_times.append(val_inf_time)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.3f} | "
              f"Val Inf Time: {val_inf_time*1000:.2f}ms")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"✅ Best Validation Accuracy: {best_acc * 100:.2f}%")
    print(f"📊 Average Inference Time: {np.mean(val_inference_times)*1000:.2f}ms")

    # === Final evaluation with confusion matrix ===
    final_val_loss, final_val_acc, final_val_precision, final_val_recall, final_val_f1, final_inf_time, y_true, y_pred = evaluate(model, val_loader, criterion, device, return_predictions=True)
    
    # Confusion Matrix / Interpretability Matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = label_encoder.classes_
    
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Training curves
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Plot 2: Precision, Recall, F1
    plt.subplot(2, 3, 3)
    plt.plot(val_precisions, label='Precision', alpha=0.7)
    plt.plot(val_recalls, label='Recall', alpha=0.7)
    plt.plot(val_f1s, label='F1-Score', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision, Recall, F1-Score')
    plt.legend()

    # Plot 3: Inference Time
    plt.subplot(2, 3, 4)
    plt.plot([t*1000 for t in val_inference_times], label='Inference Time', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Time (ms)')
    plt.title('Average Inference Time')
    plt.legend()

    # Plot 4: Confusion Matrix / Interpretability Matrix
    plt.subplot(2, 3, (5, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Interpretability)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=300, bbox_inches='tight')
    print("📊 Training metrics saved to training_metrics.png")

    # Print final metrics summary
    print("\n" + "="*50)
    print("FINAL METRICS SUMMARY")
    print("="*50)
    print(f"Final Validation Accuracy: {final_val_acc*100:.2f}%")
    print(f"Final Validation Precision: {final_val_precision:.3f}")
    print(f"Final Validation Recall: {final_val_recall:.3f}")
    print(f"Final Validation F1-Score: {final_val_f1:.3f}")
    print(f"Average Inference Time: {final_inf_time*1000:.2f}ms")
    print("="*50)

    torch.save(model.state_dict(), "best_model.pth")
    print("💾 Best model saved as best_model.pth")

    joblib.dump(label_encoder, "label_encoder.pkl")
    print("💾 Label encoder saved as label_encoder.pkl")