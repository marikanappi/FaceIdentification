import torch
from data_utils import load_and_preprocess_data, create_dataloaders
from model import LandmarkClassifier
from train import train_model
from test import evaluate_model
from utils import save_artifacts

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = "Dataset/Dataset_dist.csv"

    print(">> Caricamento e preprocessing dati...")
    X_train, X_test, y_train, y_test, scaler, label_encoder = load_and_preprocess_data(csv_path)
    train_loader = create_dataloaders(X_train, y_train)

    input_dim = X_train.shape[1]
    hidden_dim = 64
    output_dim = len(label_encoder.classes_)

    print(">> Inizializzazione e training del modello...")
    model = LandmarkClassifier(input_dim, hidden_dim, output_dim)
    train_model(model, train_loader, device)

    print(">> Valutazione del modello...")
    evaluate_model(model, X_test, y_test, label_encoder, device, threshold=0.5)

    print(">> Salvataggio del modello e oggetti...")
    save_artifacts(model, scaler, label_encoder)

if __name__ == "__main__":
    main()
