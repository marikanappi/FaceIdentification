import torch
from torch.nn.functional import softmax
import numpy as np

def evaluate_model(model, X_test, y_test, label_encoder, device, threshold=0.5):
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probs = softmax(outputs, dim=1).cpu().numpy()
    
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)

    preds_with_unknown = [
        "unknown" if p < threshold else label_encoder.inverse_transform([c])[0]
        for p, c in zip(max_probs, preds)
    ]
    true_labels = label_encoder.inverse_transform(y_test)

    print("Esempi di classificazione (vero, predetto):")
    for t, p in zip(true_labels[:10], preds_with_unknown[:10]):
        print(f"{t} → {p}")
