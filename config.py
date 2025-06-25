from model import FaceClassifier
import torch
import joblib

def config():
    MODEL_PATH = "best_model.pth"
    ENCODER_PATH = "label_encoder.pkl" 
    INPUT_DIM = 22  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Caricamento modello e label encoder ===
    label_encoder = joblib.load(ENCODER_PATH)
    num_classes = len(label_encoder.classes_)

    model = FaceClassifier(INPUT_DIM, num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()