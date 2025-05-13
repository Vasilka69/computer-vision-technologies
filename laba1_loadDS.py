import os
from datasets import load_dataset
from PIL import Image
import numpy as np

# Путь к папке с датасетом
dataset_path = r"C:\Users\slmax\OneDrive\Рабочий стол\ДЗ\Магистратура\2м семак ТКЗ\laba1\HOXSEC___csgo-maps"

# Загрузка датасета
dataset = load_dataset("HOXSEC/csgo-maps", cache_dir=dataset_path)

# Папка для сохранения изображений по классам
output_dir = r"C:\Users\slmax\OneDrive\Рабочий стол\ДЗ\Магистратура\2м семак ТКЗ\laba1\data"
os.makedirs(output_dir, exist_ok=True)

# Список классов
class_names = [
    "de_dust2", "de_inferno", "de_mirage", "de_train", "de_vertigo"
]

# Создаем подкаталоги для каждого класса
for class_name in class_names:
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)


# Функция для сохранения изображения в соответствующую папку
def save_image(image_data, image_path):
    if isinstance(image_data, np.ndarray):
        image = Image.fromarray(image_data)
    elif isinstance(image_data, Image.Image):
        image = image_data
    else:
        raise TypeError("Неподдерживаемый формат данных изображения")

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image.save(image_path, 'JPEG')


# Получаем тренировочные данные
train_data = dataset['train']

# Разделяем изображения по классам
total_images = len(train_data)
for i, example in enumerate(train_data):
    if i >= 2000:  # Ограничиваем до 2000 изображений
        break

    # Получаем изображение и метку
    image_data = example['image']
    label = example['label']  # Индекс класса
    class_name = class_names[label]

    # Преобразуем изображение в numpy, если нужно
    if isinstance(image_data, Image.Image):
        image_data = np.array(image_data)

    # Определяем путь для сохранения изображения
    image_path = os.path.join(output_dir, class_name, f"image_{i}.jpg")

    # Сохраняем изображение в папку соответствующего класса
    save_image(image_data, image_path)

    # Выводим процесс
    print(f"Изображение {i + 1}/{total_images} сохранено в классе {class_name}: {image_path}")

print("Процесс завершен.")
