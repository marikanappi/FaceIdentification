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

def load_and_preprocess_data(file_path, balance_method='weighted_sampling', test_size=0.2, val_size=0.2, batch_size=32):
    """
    Carica e preprocessa i dati gestendo lo sbilanciamento delle classi.
    
    Args:
        file_path: percorso del dataset
        balance_method: metodo per bilanciare le classi
            - 'weighted_sampling': campionamento pesato
            - 'smote': SMOTE oversampling
            - 'random_oversample': oversampling casuale
            - 'random_undersample': undersampling casuale
            - 'smoteenn': combinazione SMOTE + Edited Nearest Neighbours
            - 'none': nessun bilanciamento
        test_size: dimensione del test set
        val_size: dimensione del validation set
        batch_size: dimensione del batch
    
    Returns:
        train_loader, val_loader, test_loader, num_classes, le, class_weights
    """
    
    # Carica il dataset
    df = pd.read_csv(file_path)
    
    print("Informazioni sul dataset:")
    print(f"Shape: {df.shape}")
    print(f"Colonne: {list(df.columns)}")
    print(f"\nTipi di dati:")
    print(df.dtypes)
    
    print("\nDistribuzione originale delle classi:")
    print(df['identity'].value_counts())
    print(f"\nNumero totale di campioni: {len(df)}")
    print(f"Numero di classi: {df['identity'].nunique()}")
    
    # Identifica manualmente le colonne in base al contenuto
    # 'identity' è il target, 'expression' è categorica, il resto sono numeriche
    target_column = 'identity'
    categorical_features = ['expression']  # Colonna con valori come 'CAU'
    
    # Tutte le altre colonne (esclusi target e categoriche) sono numeriche
    all_columns = df.columns.tolist()
    numeric_features = [col for col in all_columns if col not in [target_column] + categorical_features]
    
    print(f"\nColonne numeriche: {numeric_features}")
    print(f"Colonne categoriche: {categorical_features}")
    print(f"Target: {target_column}")
    
    # Verifica che le colonne numeriche siano effettivamente convertibili
    for col in numeric_features:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            print(f"Attenzione: problemi nella conversione della colonna {col}")
    
    # Controlla se ci sono valori NaN dopo la conversione
    if df[numeric_features].isnull().any().any():
        print("Attenzione: trovati valori NaN dopo la conversione numerica")
        print("Valori NaN per colonna:")
        print(df[numeric_features].isnull().sum())
        # Riempi i NaN con la media della colonna
        df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
    
    # Prepara le features
    X_parts = []
    
    # Aggiungi features numeriche
    if numeric_features:
        X_numeric = df[numeric_features].values
        print(f"Features numeriche shape: {X_numeric.shape}")
        X_parts.append(X_numeric)
    
    # Gestisci features categoriche con Label Encoding
    if categorical_features:
        print(f"\nProcessando colonne categoriche...")
        for col in categorical_features:
            unique_values = df[col].unique()
            print(f"Colonna '{col}': {len(unique_values)} valori unici - {unique_values}")
            le_cat = LabelEncoder()
            X_cat = le_cat.fit_transform(df[col].astype(str)).reshape(-1, 1)
            X_parts.append(X_cat)
            print(f"Colonna '{col}' codificata: shape {X_cat.shape}")
    
    # Combina tutte le features
    if len(X_parts) > 1:
        X = np.hstack(X_parts)
    else:
        X = X_parts[0]
    
    print(f"\nFeatures finali shape: {X.shape}")
    
    # Target
    y = df[target_column].values
    
    # Encoding delle label target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Normalizzazione delle features
    print("Normalizzando le features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split stratificato per mantenere la distribuzione delle classi
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(sss.split(X_scaled, y_encoded))
    
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    
    # Split train in train e validation
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    train_idx2, val_idx = next(sss_val.split(X_train, y_train))
    
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    X_train = X_train[train_idx2]
    y_train = y_train[train_idx2]
    
    print(f"\nDimensioni dopo split:")
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Calcola i pesi delle classi per la loss function
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.FloatTensor(class_weights)
    
    # Applica il metodo di bilanciamento scelto
    if balance_method == 'smote':
        print("\nApplicando SMOTE...")
        smote = SMOTE(random_state=42, k_neighbors=min(5, min(np.bincount(y_train)) - 1))
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
    elif balance_method == 'random_oversample':
        print("\nApplicando Random Oversampling...")
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        
    elif balance_method == 'random_undersample':
        print("\nApplicando Random Undersampling...")
        rus = RandomUnderSampler(random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        
    elif balance_method == 'smoteenn':
        print("\nApplicando SMOTEENN...")
        # Usa k_neighbors più piccolo per evitare errori con classi piccole
        min_samples = min(np.bincount(y_train))
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
        smoteenn = SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=k_neighbors))
        X_train, y_train = smoteenn.fit_resample(X_train, y_train)
    
    if balance_method != 'none' and balance_method != 'weighted_sampling':
        print(f"\nNuova distribuzione dopo {balance_method}:")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"Classe {le.inverse_transform([cls])[0]}: {count}")
    
    # Converti in tensori PyTorch
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Crea i dataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Crea i DataLoader
    if balance_method == 'weighted_sampling':
        print("\nUsando Weighted Random Sampler...")
        # Calcola i pesi per ogni campione
        sample_weights = []
        for label in y_train:
            sample_weights.append(class_weights[label])
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    num_classes = len(np.unique(y_encoded))
    
    return train_loader, val_loader, test_loader, num_classes, le, class_weights


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