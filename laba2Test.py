import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Автоматическое определение пути к split_data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(BASE_DIR, 'split_data') # Нормальные данные
DATA_DIR = os.path.join(BASE_DIR, 'split_data') # Аугментированные данные

# Преобразования
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Загрузка датасета
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка предобученной модели
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Обучение
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Тренировка
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in dataloaders['train']:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {running_loss:.4f}")

# Оценка на тестовой выборке
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in dataloaders['test']:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Метрики
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Матрица ошибок
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
