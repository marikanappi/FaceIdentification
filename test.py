from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch

def test_model(model, test_loader, device='cpu', le=None):
    model.to(device)
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("\n\U0001f9ea Test Results:")
    print("Accuracy: {:.2f}%".format(accuracy_score(y_true, y_pred) * 100))

    print("\nClassification Report:")
    if le:
        print(classification_report(y_true, y_pred, target_names=le.classes_))
        print("\n\U0001f4cb Identity reale vs predetta:")
        for real, pred in zip(y_true, y_pred):
            print(f"Reale: {le.inverse_transform([real])[0]:<10} | Predetta: {le.inverse_transform([pred])[0]:<10}")
    else:
        print(classification_report(y_true, y_pred))
