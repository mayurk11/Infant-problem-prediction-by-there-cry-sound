import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset_of_pretrain import get_dataloaders
from torchvision import models

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, running_corrects = 0.0, 0

    for inputs, labels in tqdm(dataloader, desc="[Train]"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc.item()

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, running_corrects = 0.0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="[Val]"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc.item()

def main():
    # Paths
    data_dir = r"C:\Users\mayur\Desktop\Infrant Problem\data2"  # Make sure this has 'train' and 'val' folders
    save_path = r"C:\Users\mayur\Desktop\Infrant Problem\models"
    os.makedirs(save_path, exist_ok=True)

    # Training config
    num_classes = 7
    num_epochs = 15
    batch_size = 32
    learning_rate = 0.001
    image_size = 224

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader = get_dataloaders(data_dir, batch_size, image_size)

    # Load pretrained EfficientNet
    model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Model checkpointing
    best_val_acc = 0.0
    checkpoint_path = os.path.join(save_path, 'best_model_pretrain_model.pth')

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"✅ Best model saved at epoch {epoch+1} with val acc: {val_acc:.4f}")

if __name__ == "__main__":
    main()
