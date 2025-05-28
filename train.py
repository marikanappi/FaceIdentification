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

def train_model(model, train_loader, val_loader, num_classes, class_weights=None, 
                epochs=200, learning_rate=0.001, weight_decay=5e-4, 
                loss_type='weighted_ce', device='cpu'):
    """
    Addestra il modello con supporto per classi sbilanciate.
    
    Args:
        model: modello da addestrare
        train_loader, val_loader: data loaders
        num_classes: numero di classi
        class_weights: pesi delle classi per la loss function
        epochs: numero di epoche
        learning_rate, weight_decay: parametri ottimizzatore
        loss_type: tipo di loss function
            - 'weighted_ce': CrossEntropy pesata
            - 'focal': Focal Loss
            - 'label_smoothing': Label Smoothing
            - 'ce': CrossEntropy standard
        device: dispositivo di calcolo
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
    
    # Ottimizzatore con weight decay per regolarizzazione
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Scheduler per ridurre il learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_balanced_accs, val_balanced_accs = [], []
    
    best_val_loss = float('inf')
    best_balanced_acc = 0
    patience = 20
    counter = 0
    best_model_state = None

    print(f"\nInizio training su {device}")
    print(f"Parametri: lr={learning_rate}, weight_decay={weight_decay}, epochs={epochs}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping per stabilità
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

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        all_val_preds, all_val_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_balanced_acc = balanced_accuracy_score(all_val_labels, all_val_preds) * 100
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted') * 100

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_balanced_accs.append(train_balanced_acc)
        val_balanced_accs.append(val_balanced_acc)

        # Update scheduler
        scheduler.step(avg_val_loss)

        # Print ogni 10 epoche o le prime/ultime 5
        if epoch <= 5 or epoch % 10 == 0 or epoch >= epochs - 5:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"Train Acc: {train_accuracy:5.2f}% | Val Acc: {val_accuracy:5.2f}% | "
                  f"Val Balanced Acc: {val_balanced_acc:5.2f}% | Val F1: {val_f1:5.2f}%")

        # Early stopping basato su balanced accuracy (migliore per classi sbilanciate)
        if val_balanced_acc > best_balanced_acc:
            best_balanced_acc = val_balanced_acc
            best_val_loss = avg_val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f"\nEarly stopping all'epoca {epoch}")
                print(f"Migliore Balanced Accuracy: {best_balanced_acc:.2f}%")
                break

    # Carica il modello migliore
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\nCaricato il modello con Balanced Accuracy: {best_balanced_acc:.2f}%")

    # Plot delle metriche
    plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies, 
                         train_balanced_accs, val_balanced_accs)

    return model

def plot_training_metrics(train_losses, val_losses, train_accs, val_accs, 
                         train_bal_accs, val_bal_accs):
    """Plot completo delle metriche di training"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(train_losses, label='Train Loss', marker='o', markersize=2)
    axes[0, 0].plot(val_losses, label='Validation Loss', marker='o', markersize=2)
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(train_accs, label='Train Accuracy', marker='o', markersize=2)
    axes[0, 1].plot(val_accs, label='Validation Accuracy', marker='o', markersize=2)
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Balanced Accuracy
    axes[1, 0].plot(train_bal_accs, label='Train Balanced Acc', marker='o', markersize=2)
    axes[1, 0].plot(val_bal_accs, label='Val Balanced Acc', marker='o', markersize=2)
    axes[1, 0].set_title('Balanced Accuracy (Important for Imbalanced Classes)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Balanced Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss (log scale)
    axes[1, 1].plot(train_losses, label='Train Loss', marker='o', markersize=2)
    axes[1, 1].plot(val_losses, label='Validation Loss', marker='o', markersize=2)
    axes[1, 1].set_title('Loss (Log Scale)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss (log)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()