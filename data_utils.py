import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
def load_and_preprocess_data(csv_path):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    df = pd.read_csv(csv_path)

    # Filtra via le identità con meno di 2 campioni
    identity_counts = df['identity'].value_counts()
    df = df[df['identity'].isin(identity_counts[identity_counts >= 2].index)]

    # Seleziona le features (distanze)
    X = df[['ensx_se', 'ensx_exsx', 'se_prn', 'se_alsx',
            'prn_alsx', 'ensx_alsx', 'prn_sn', 'alsx_sn', 'ensx_prn']].values
    y = df['identity'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, label_encoder


def create_dataloaders(X_train, y_train, batch_size=32):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    return DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
