import torch
import joblib
from model import LandmarkClassifier
from torch.nn.functional import softmax
import numpy as np

def save_artifacts(model, scaler, label_encoder, model_path="mlp_landmark.pt", scaler_path="scaler.pkl", encoder_path="label_encoder.pkl"):
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    print("Modello e oggetti salvati.")

def load_artifacts(model_path, scaler_path, encoder_path, input_dim, hidden_dim, output_dim, device):
    model = LandmarkClassifier(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    return model, scaler, label_encoder

def predict_identity(model, X_new, scaler, label_encoder, device, threshold=0.5):
    X_scaled = scaler.transform([X_new])
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(X_tensor)
        probs = softmax(output, dim=1).cpu().numpy()[0]
    if np.max(probs) < threshold:
        return "unknown"
    return label_encoder.inverse_transform([np.argmax(probs)])[0]
