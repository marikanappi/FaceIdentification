import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, device='cpu', epochs= 200):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005,  weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(1, epochs + 1):
        # ----- TRAIN -----
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == y_batch).sum().item()
            total_train += y_batch.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_accuracy)

        # ----- VALIDATION -----
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)

                outputs = model(X_val)
                loss = criterion(outputs, y_val)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == y_val).sum().item()
                total_val += y_val.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val

        val_loss_history.append(avg_val_loss)
        val_acc_history.append(val_accuracy)

        print(f"Epoch {epoch}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% - "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Plot delle loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss during training')
    plt.legend()
    plt.grid(True)

    # Plot delle accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy during training')
    plt.legend()
    plt.grid(True)

    plt.show()
