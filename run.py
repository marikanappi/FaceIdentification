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

def run_training(seed=None):
    # === Config ===
    csv_path = 'balanced_dataset.csv'
    batch_size = 128
    lr = 1e-3
    num_epochs = 150
    patience = 20
    
    # Usa seed solo per l'inizializzazione del modello se specificato
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Load data ===
    train_dataset, val_dataset, test_dataset, label_encoder = load_data(csv_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    input_dim = train_dataset[0][0].shape[0]
    num_classes = len(label_encoder.classes_)
    print(f"Input dimension: {input_dim}, Number of classes: {num_classes}")
    model = FaceClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accuracies = [], []
    val_accuracies, val_precisions, val_recalls, val_f1s = [], [], [], []
    val_inference_times = []
    best_val_loss = float('inf')
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
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.3f} | "
            f"Val Inf Time: {val_inf_time*1000:.2f}ms")

        if val_loss < best_val_loss - 1e-4:  
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f" Early stopping at epoch {epoch+1}")
                break


    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Save the best model to file
        torch.save(best_model_state, "best_model.pth")
        print("✓ Best model saved to best_model.pth")

    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Average Inference Time: {np.mean(val_inference_times)*1000:.2f}ms")

    # === Final evaluation with confusion matrix ===
    final_val_loss, final_val_acc, final_val_precision, final_val_recall, final_val_f1, final_inf_time, y_true, y_pred = evaluate(model, val_loader, criterion, device, return_predictions=True)
    class_names = label_encoder.classes_
    
        # === Plot 1: Training Loss ===
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("train_loss.png", dpi=300)
    plt.close()
    print(" Training loss plot salvato in train_loss.png")

    # === Plot 2: Accuracy ===
    plt.figure(figsize=(8,6))
    plt.plot(train_accuracies, label='Train Accuracy', color='green')
    plt.plot(val_accuracies, label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy.png", dpi=300)
    plt.close()
    print(" Accuracy plot salvato in accuracy.png")

    # === Plot 3a: Precision ===
    plt.figure(figsize=(8,6))
    plt.plot(val_precisions, label='Precision', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Validation Precision')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("precision.png", dpi=300)
    plt.close()
    print(" Precision plot salvato in precision.png")

    # === Plot 3b: Recall ===
    plt.figure(figsize=(8,6))
    plt.plot(val_recalls, label='Recall', color='brown')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Validation Recall')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("recall.png", dpi=300)
    plt.close()
    print(" Recall plot salvato in recall.png")

    # === Plot 3c: F1-Score ===
    plt.figure(figsize=(8,6))
    plt.plot(val_f1s, label='F1-Score', color='darkcyan')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title('Validation F1-Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("f1_score.png", dpi=300)
    plt.close()
    print(" F1-Score plot salvato in f1_score.png")


    # === Plot 4: Average Inference Time ===
    plt.figure(figsize=(8,6))
    plt.plot([t*1000 for t in val_inference_times], label='Inference Time (ms)', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Time (ms)')
    plt.title('Average Inference Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("inference_time.png", dpi=300)
    plt.close()
    print(" Inference time plot salvato in inference_time.png")
