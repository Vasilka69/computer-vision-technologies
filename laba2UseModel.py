import os
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Путь к модели
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_name = input("Введите имя модели (без .pth): ").strip()
model_path = os.path.join(BASE_DIR, "pth", f"{model_name}.pth")

# Папка с картинками
image_folder = input("Введите путь к папке с изображениями по классам: ").strip()

# Трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Датасет и загрузка
test_dataset = datasets.ImageFolder(image_folder, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
class_names = test_dataset.classes
num_classes = len(class_names)

# Загрузка модели
model_loaded = resnet18(weights=ResNet18_Weights.DEFAULT)
model_loaded.fc = torch.nn.Linear(model_loaded.fc.in_features, num_classes)
model_loaded.load_state_dict(torch.load(model_path, map_location=device))
model_loaded = model_loaded.to(device)
model_loaded.eval()

# Предсказания
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_loaded(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Отчёт
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Матрица
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
