from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch

def test_model(model, test_loader, device='cpu', le=None):
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calcola e stampa metriche
    print("\n🧪 Test Results:")
    print("Accuracy: {:.2f}%".format(accuracy_score(y_true, y_pred) * 100))

    if le:
        target_names = le.classes_
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=target_names))

        # Stampa identity reale e predetta
        print("\n📋 Identity reale vs predetta:")
        for real, pred in zip(y_true, y_pred):
            real_label = le.inverse_transform([real])[0]
            pred_label = le.inverse_transform([pred])[0]
            print(f"Reale: {real_label:<10} | Predetta: {pred_label:<10}")
    else:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))