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
# DATA_DIR = os.path.join(BASE_DIR, 'split_data') # Нормальные данные
DATA_DIR = os.path.join(BASE_DIR, 'augment') # Аугментированные данные

# Трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Загрузка датасетов
datasets_dict = {
    'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform),
    'val': datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform),
    'test': datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform),
}

# DataLoaders
dataloaders = {
    x: DataLoader(datasets_dict[x], batch_size=32, shuffle=(x == 'train'))
    for x in ['train', 'val', 'test']
}

# Классы
class_names = datasets_dict['train'].classes
num_classes = len(class_names)

# Устройство
device = torch.device("cuda")

# Модель
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Оптимизатор и функция потерь
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Обучение
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"\nЭпоха {epoch + 1}/{num_epochs}")
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

    print()  # переход на новую строку
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

# Тест
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

# Метрики
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Матрица ошибок
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
