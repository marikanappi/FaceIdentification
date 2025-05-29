import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score
import numpy as np

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o', markersize=3)
    plt.plot(val_losses, label='Validation Loss', marker='o', markersize=3)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', marker='o', markersize=3)
    plt.plot(val_losses, label='Validation Loss', marker='o', markersize=3)
    plt.title('Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

class FocalLoss(nn.Module):
    """Focal Loss per gestire classi sbilanciate"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        pred = self.log_softmax(x)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def train_model_no_val(model, train_loader, num_classes, class_weights, 
                       epochs, learning_rate, weight_decay, 
                       loss_type='weighted_ce', device='cpu'):
    """
    Addestra il modello solo sul training set (nessuna validazione).
    """
    
    model.to(device)
    
    # Scegli la loss function
    if loss_type == 'weighted_ce' and class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Usando CrossEntropy Loss pesata")
    elif loss_type == 'focal':
        criterion = FocalLoss(alpha=1, gamma=2)
        print("Usando Focal Loss")
    elif loss_type == 'label_smoothing':
        criterion = LabelSmoothingLoss(classes=num_classes, smoothing=0.1)
        print("Usando Label Smoothing Loss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Usando CrossEntropy Loss standard")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_losses, train_accuracies, train_balanced_accs = [], [], []

    print(f"\nInizio training (solo su training set) su {device}")
    print(f"Parametri: lr={learning_rate}, weight_decay={weight_decay}, epochs={epochs}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        train_balanced_accs.append(train_balanced_acc)

        if epoch <= 5 or epoch % 10 == 0 or epoch >= epochs - 5:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Train Acc: {train_accuracy:5.2f}% | "
                  f"Train Balanced Acc: {train_balanced_acc:5.2f}%")

    # Plot finale
    plot_train_metrics(train_losses, train_accuracies, train_balanced_accs)

    return model

def plot_train_metrics(train_losses, train_accs, train_bal_accs):
    """Plot delle metriche di training"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(train_losses, label='Train Loss', marker='o', markersize=2)
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_accs, label='Train Accuracy', marker='o', markersize=2)
    axes[1].set_title('Train Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(train_bal_accs, label='Balanced Accuracy', marker='o', markersize=2)
    axes[2].set_title('Train Balanced Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Balanced Acc (%)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
