import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for sequences, labels in train_loader:
            # Move data to the correct device
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )

        # Debugging: Check predictions on validation data
        print("Sample Validation Predictions:")
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                _, preds = torch.max(outputs, 1)
                print(f"Predictions: {preds.tolist()}, Ground Truth: {labels.tolist()}")
                break  # Only check the first batch


def evaluate_model(model, loader, criterion, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    loss = 0.0

    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return loss / len(loader), 100 * correct / total
