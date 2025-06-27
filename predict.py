import torch
import joblib
from model import FaceClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Caricamento modello e risorse (caricati una volta all'importazione)
model = FaceClassifier(input_dim=40, num_classes=111)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

def predict_identity(features, threshold=0.5):
    # Preprocessing
    features_scaled = scaler.transform([features])
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        max_prob, pred = torch.max(probs, dim=1)

    if max_prob.item() < threshold:
        return "unknown", max_prob.item()
    else:
        label = label_encoder.inverse_transform([pred.item()])[0]
        return label, max_prob.item()
