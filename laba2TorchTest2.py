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

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'augment')

# Трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Датасеты и загрузчики
datasets_dict = {
    'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform),
    'val': datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform),
    'test': datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform),
}

dataloaders = {
    x: DataLoader(datasets_dict[x], batch_size=32, shuffle=(x == 'train'))
    for x in ['train', 'val', 'test']
}

# Классы
class_names = datasets_dict['train'].classes
num_classes = len(class_names)

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Модель
model = resnet18(weights=ResNet18_Weights.DEFAULT)

#  Feature extraction: заморозить все, кроме последнего слоя
for param in model.parameters():
    param.requires_grad = False

# Заменить классификатор под свои классы
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# Обучение: Feature Extraction
def train_model(model, dataloaders, criterion, optimizer, num_epochs=5, phase="Feature Extraction"):
    print(f"\n=== {phase} ===")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(dataloaders['train']):
            images, labels = images.to(device), labels.to(device)
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

        # Валидация
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in dataloaders['val']:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = 100 * val_correct / val_total
        print(f"\nЭпоха {epoch + 1} завершена за {time.time() - start_time:.2f} сек | Валидация: {val_acc:.2f}%")

# Оптимизатор только для fc
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Шаг 1: Feature Extraction
train_model(model, dataloaders, criterion, optimizer, num_epochs=5, phase="Feature Extraction")

#  Шаг 2: Fine-tuning — размораживаем все слои
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Шаг 2: Fine-Tuning
train_model(model, dataloaders, criterion, optimizer, num_epochs=5, phase="Fine Tuning")

# Оценка на тесте
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in dataloaders['test']:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Отчёт
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# === Матрица ошибок ===
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()


# Сохранение модели
save_prompt = input("\nСохранить обученную модель? (y/n): ").strip().lower()
if save_prompt == 'y':
    os.makedirs(os.path.join(BASE_DIR, 'pth'), exist_ok=True)
    file_name = input("Введите имя файла модели (без .pth): ").strip()
    model_path = os.path.join(BASE_DIR, f'pth/{file_name}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Модель успешно сохранена в файл: {model_path}")
else:
    print("Модель не сохранена.")