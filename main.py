import torch
from data_utils import load_and_preprocess_data, analyze_class_distribution
from train import train_model
from test import test_model
from model import FaceClassifier
import pandas as pd

def main():
    # Configura il dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    print("=" * 60)
    analyze_class_distribution("Dataset/Dataset_dist.csv")
    print("=" * 60)

    # Configurazione unica
    config = {
        'name': 'Baseline_Improved_NoVal',
        'model_class': FaceClassifier,
        'model_params': {'dropout_rate': 0.5},
        'balance_method': 'weighted_sampling',
        'loss_type': 'weighted_ce',
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'batch_size': 64,
        'epochs': 200
    }

    print(f"\n{'='*20} AVVIO {config['name'].upper()} {'='*20}")

    train_loader, test_loader, num_classes, le, class_weights = load_and_preprocess_data(
        "Dataset/Dataset_dist.csv",
        balance_method=config['balance_method'],
        batch_size=config['batch_size']
    )

    # Ottieni input_dim
    for X_batch, _ in train_loader:
        input_dim = X_batch.shape[1]
        break

    # Crea il modello
    model = config['model_class'](
        input_dim=input_dim,
        num_classes=num_classes,
        **config['model_params']
    )

    # Addestra il modello (senza validazione)
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        num_classes=num_classes,
        class_weights=class_weights if config['loss_type'] == 'weighted_ce' else None,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        loss_type=config['loss_type'],
        device=device
    )

    # Testa il modello
    print(f"\n{'-'*20} RISULTATI TEST {config['name'].upper()} {'-'*20}")
    test_accuracy = test_model(trained_model, test_loader, device=device, le=le)

    print(f"Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
