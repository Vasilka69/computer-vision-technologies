import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'augment')

# Трансформации
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Загрузка датасетов
datasets_dict = {
    'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform),
    'val': datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_test_transform),
    'test': datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=val_test_transform),
}

# DataLoaders
dataloaders = {
    x: DataLoader(datasets_dict[x], batch_size=32, shuffle=(x == 'train'))
    for x in ['train', 'val', 'test']
}

# Классы
class_names = datasets_dict['train'].classes
num_classes = len(class_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, mode_label):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    num_epochs = 10
    print(f"\n=== Начинаем обучение в режиме: {mode_label} ===")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(dataloaders['train']):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress = 100 * (batch_idx + 1) / len(dataloaders['train'])
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100 * correct / total
            print(f"\rПрогресс: {progress:.1f}% | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%", end='', flush=True)

        print()
        # Валидация
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in dataloaders['val']:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        elapsed = time.time() - start_time
        print(f"Эпоха {epoch + 1} завершена за {elapsed:.2f} сек | Точность валидации: {val_acc:.2f}%")

    return model

def evaluate_model(model, mode_label):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloaders['test']:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(f"\n=== Classification Report ({mode_label}) ===")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {mode_label}')
    plt.tight_layout()
    plt.show()

# Feature Extraction
model_feature = resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model_feature.parameters():
    param.requires_grad = False
model_feature.fc = nn.Linear(model_feature.fc.in_features, num_classes)
model_feature.fc.requires_grad = True
model_feature = model_feature.to(device)

model_feature = train_model(model_feature, "Feature Extraction")
evaluate_model(model_feature, "Feature Extraction")

# Fine Tuning
model_fine = resnet18(weights=ResNet18_Weights.DEFAULT)
model_fine.fc = nn.Linear(model_fine.fc.in_features, num_classes)
model_fine = model_fine.to(device)

model_fine = train_model(model_fine, "Fine Tuning")
evaluate_model(model_fine, "Fine Tuning")
