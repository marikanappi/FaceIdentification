from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score, f1_score
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def test_model(model, test_loader, device='cpu', le=None, show_confusion_matrix=True):
    model.to(device)
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            
            # Predizioni e probabilità
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probabilities.cpu().numpy())

    # Calcola metriche
    accuracy = accuracy_score(y_true, y_pred) * 100
    balanced_acc = balanced_accuracy_score(y_true, y_pred) * 100
    f1_weighted = f1_score(y_true, y_pred, average='weighted') * 100
    f1_macro = f1_score(y_true, y_pred, average='macro') * 100

    print("\n🧪 TEST RESULTS:")
    print(f"{'='*50}")
    print(f"Accuracy:           {accuracy:.2f}%")
    print(f"Balanced Accuracy:  {balanced_acc:.2f}%")
    print(f"F1-Score (Weighted): {f1_weighted:.2f}%")
    print(f"F1-Score (Macro):   {f1_macro:.2f}%")
    print(f"{'='*50}")

    # Classification Report dettagliato
    print("\n📊 Classification Report:")
    if le:
        target_names = le.classes_
        print(classification_report(y_true, y_pred, target_names=target_names, digits=3))
    else:
        print(classification_report(y_true, y_pred, digits=3))

    # Analisi per classe
    if le:
        print("\n🔍 Analisi per classe:")
        print(f"{'Classe':<10} {'Accuracy':<10} {'Support':<10} {'F1-Score':<10}")
        print("-" * 45)
        
        # Calcola metriche per classe
        unique_classes = np.unique(y_true)
        for cls in unique_classes:
            cls_mask = np.array(y_true) == cls
            cls_accuracy = np.mean(np.array(y_pred)[cls_mask] == cls) * 100
            cls_support = np.sum(cls_mask)
            cls_f1 = f1_score(y_true, y_pred, labels=[cls], average='macro') * 100
            
            cls_name = le.inverse_transform([cls])[0]
            print(f"{cls_name:<10} {cls_accuracy:<10.1f} {cls_support:<10} {cls_f1:<10.1f}")

    # Analisi degli errori più comuni
    analyze_common_errors(y_true, y_pred, le if le else None, top_k=10)
    
    # Analisi della confidenza delle predizioni
    analyze_prediction_confidence(y_probs, y_true, y_pred, le if le else None)

    return accuracy

def analyze_common_errors(y_true, y_pred, le=None, top_k=10):
    """Analizza gli errori più comuni"""
    print(f"\n❌ Top {top_k} errori più comuni:")
    print("-" * 60)
    
    # Conta gli errori
    error_counts = {}
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label != pred_label:
            error_pair = (true_label, pred_label)
            error_counts[error_pair] = error_counts.get(error_pair, 0) + 1
    
    # Ordina per frequenza
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Reale':<10} {'Predetta':<10} {'Count':<8} {'%':<8}")
    print("-" * 40)
    
    total_errors = sum(error_counts.values())
    for (true_cls, pred_cls), count in sorted_errors[:top_k]:
        percentage = (count / total_errors) * 100
        
        if le:
            true_name = le.inverse_transform([true_cls])[0]
            pred_name = le.inverse_transform([pred_cls])[0]
            print(f"{true_name:<10} {pred_name:<10} {count:<8} {percentage:<8.1f}")
        else:
            print(f"{true_cls:<10} {pred_cls:<10} {count:<8} {percentage:<8.1f}")

def analyze_prediction_confidence(y_probs, y_true, y_pred, le=None):
    """Analizza la confidenza delle predizioni"""
    y_probs = np.array(y_probs)
    max_probs = np.max(y_probs, axis=1)
    
    # Confidenza media per predizioni corrette vs sbagliate
    correct_mask = np.array(y_true) == np.array(y_pred)
    correct_confidence = np.mean(max_probs[correct_mask])
    incorrect_confidence = np.mean(max_probs[~correct_mask])
    
    print(f"\n🎯 Analisi Confidenza:")
    print("-" * 30)
    print(f"Confidenza media (corrette):   {correct_confidence:.3f}")
    print(f"Confidenza media (sbagliate):  {incorrect_confidence:.3f}")
    print(f"Differenza:                    {correct_confidence - incorrect_confidence:.3f}")
    
    # Distribuzione delle confidenze
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(max_probs[correct_mask], bins=20, alpha=0.7, label='Corrette', color='green')
    plt.hist(max_probs[~correct_mask], bins=20, alpha=0.7, label='Sbagliate', color='red')
    plt.xlabel('Confidenza Massima')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione Confidenza')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    confidence_thresholds = np.linspace(0.1, 1.0, 10)
    accuracies_by_confidence = []
    
    for threshold in confidence_thresholds:
        high_conf_mask = max_probs >= threshold
        if np.sum(high_conf_mask) > 0:
            acc = np.mean(np.array(y_true)[high_conf_mask] == np.array(y_pred)[high_conf_mask])
            accuracies_by_confidence.append(acc)
        else:
            accuracies_by_confidence.append(0)
    
    plt.plot(confidence_thresholds, accuracies_by_confidence, 'o-')
    plt