import os
import torch
import torch.nn as nn
from torchvision import models
from dataset_of_pretrain import get_dataloaders
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

def load_model(checkpoint_path, num_classes, device):
    model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def evaluate(model, dataloader, device, class_names):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    print("✅ Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    print("✅ Confusion Matrix:\n")
    print(confusion_matrix(all_labels, all_preds))

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Overall Accuracy: {acc:.4f}")

def main():
    data_dir = r"C:\Users\mayur\Desktop\Infrant Problem\data2"
    checkpoint_path = r"C:\Users\mayur\Desktop\Infrant Problem\models\best_model_pretrain_model.pth"
    batch_size = 32
    image_size = 224
    num_classes = 7
    class_names = [f"Class {i}" for i in range(num_classes)]  # Or replace with real class names

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = get_dataloaders(data_dir, batch_size, image_size)
    model = load_model(checkpoint_path, num_classes, device)

    evaluate(model, val_loader, device, class_names)

if __name__ == "__main__":
    main()
