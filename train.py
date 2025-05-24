import torch
import torch.nn as nn
def train_model(model, train_loader, device='cpu', epochs = 300):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch {epoch}/50 - Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.2f}%")