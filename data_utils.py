import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

def load_and_preprocess_data(file_path, balance_method='weighted_sampling', test_size=0.2, batch_size=32):
    df = pd.read_csv(file_path)
    
    print("Informazioni sul dataset:")
    print(f"Shape: {df.shape}")
    print(f"Colonne: {list(df.columns)}")
    
    target_column = 'identity'
    categorical_features = ['expression']
    all_columns = df.columns.tolist()
    numeric_features = [col for col in all_columns if col not in [target_column] + categorical_features]
    
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if df[numeric_features].isnull().any().any():
        df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
    
    X_parts = []
    if numeric_features:
        X_numeric = df[numeric_features].values
        X_parts.append(X_numeric)
    
    if categorical_features:
        for col in categorical_features:
            le_cat = LabelEncoder()
            X_cat = le_cat.fit_transform(df[col].astype(str)).reshape(-1, 1)
            X_parts.append(X_cat)
    
    X = np.hstack(X_parts) if len(X_parts) > 1 else X_parts[0]
    y = df[target_column].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified test set: 1 sample per class
    test_indices = []
    remaining_indices = []
    np.random.seed(42)
    for cls in np.unique(y_encoded):
        class_indices = np.where(y_encoded == cls)[0]
        np.random.shuffle(class_indices)
        test_indices.append(class_indices[0])
        remaining_indices.extend(class_indices[1:])
    
    test_indices = np.array(test_indices)
    remaining_indices = np.array(remaining_indices)

    X_test = X_scaled[test_indices]
    y_test = y_encoded[test_indices]
    X_train = X_scaled[remaining_indices]
    y_train = y_encoded[remaining_indices]

    print(f"\nDimensioni split:")
    print(f"Train: {X_train.shape[0]}, Test (1 per classe): {X_test.shape[0]}")
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.FloatTensor(class_weights)
    
    if balance_method == 'smote':
        smote = SMOTE(random_state=42, k_neighbors=min(5, min(np.bincount(y_train)) - 1))
        X_train, y_train = smote.fit_resample(X_train, y_train)
    elif balance_method == 'random_oversample':
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
    elif balance_method == 'random_undersample':
        rus = RandomUnderSampler(random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
    elif balance_method == 'smoteenn':
        min_samples = min(np.bincount(y_train))
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
        smoteenn = SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=k_neighbors))
        X_train, y_train = smoteenn.fit_resample(X_train, y_train)

    if balance_method != 'none' and balance_method != 'weighted_sampling':
        print(f"\nNuova distribuzione dopo {balance_method}:")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"Classe {le.inverse_transform([cls])[0]}: {count}")
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    if balance_method == 'weighted_sampling':
        print("\nUsando Weighted Random Sampler...")
        sample_weights = [class_weights[label] for label in y_train]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    num_classes = len(np.unique(y_encoded))

    return train_loader, test_loader, num_classes, le, class_weights

def analyze_class_distribution(file_path):
    """Analizza la distribuzione delle classi nel dataset"""
    df = pd.read_csv(file_path)
    
    print("=== ANALISI DISTRIBUZIONE CLASSI ===")
    counts = df['identity'].value_counts()
    print(f"\nNumero totale di campioni: {len(df)}")
    print(f"Numero di classi: {df['identity'].nunique()}")
    print(f"Campioni per classe - Min: {counts.min()}, Max: {counts.max()}, Media: {counts.mean():.1f}")
    
    # Calcola statistiche di sbilanciamento
    imbalance_ratio = counts.max() / counts.min()
    print(f"Ratio di sbilanciamento: {imbalance_ratio:.2f}")
    
    # Mostra distribuzione
    print(f"\nDistribuzione completa:")
    print(counts)
    
    return counts