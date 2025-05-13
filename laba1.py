import os
import shutil
import random
import logging
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import tempfile

# Use matplotlib's TkAgg backend to fix PyCharm visualization issues
matplotlib.use('TkAgg')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline_log.txt", mode='w'),
        logging.StreamHandler()
    ]
)

def split_data(source_dir, base_dest_dir, val_size=0.1, test_size=0.1, img_extensions=('.jpg', '.png')):
    # Создание папки split_data внутри указанного пользователем пути
    output_dir = os.path.join(base_dest_dir, 'split_data')
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Создана выходная папка для split: {output_dir}")

    logging.info("Запущена сортировка данных...")

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in tqdm(os.listdir(source_dir), desc="Сортировка классов"):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [img for img in os.listdir(class_path) if img.endswith(img_extensions)]
        logging.info(f"Обнаружено {len(images)} изображений в классе '{class_name}'.")

        if not images:
            logging.warning(f"Папка {class_name} пуста, пропускаем.")
            continue

        random.shuffle(images)
        total = len(images)
        val_count = int(total * val_size)
        test_count = int(total * test_size)

        sets = {
            train_dir: images[val_count + test_count:],
            val_dir: images[:val_count],
            test_dir: images[val_count:val_count + test_count]
        }

        for dest, img_list in sets.items():
            class_dest = os.path.join(dest, class_name)
            os.makedirs(class_dest, exist_ok=True)
            for img in img_list:
                shutil.copy(os.path.join(class_path, img), os.path.join(class_dest, img))
            logging.info(f"Сохранено {len(img_list)} изображений в {class_dest}")

    logging.info("Сортировка завершена.")
    return train_dir, val_dir, test_dir


def augment_image(image):
    augmentations = [
        lambda img: img.rotate(random.uniform(0, 360)),
        lambda img: img.transform(img.size, Image.AFFINE, (1, 0, random.randint(-10, 10), 0, 1, random.randint(-10, 10))),
        lambda img: img.transform(img.size, Image.AFFINE, (1, random.uniform(-0.2, 0.2), 0, random.uniform(-0.2, 0.2), 1, 0)),
        lambda img: img.transpose(Image.FLIP_LEFT_RIGHT if random.choice([True, False]) else Image.FLIP_TOP_BOTTOM),
        lambda img: img.crop((random.randint(0, img.width // 4), random.randint(0, img.height // 4), img.width, img.height)).resize(img.size),
        lambda img: Image.fromarray(np.clip(np.array(img) + np.random.normal(0, 25, np.array(img).shape), 0, 255).astype(np.uint8)),
        lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5)),
        lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 3))),
        lambda img: random_erasing(img),
        lambda img: hide_and_seek(img)
    ]
    chosen = random.choice(augmentations)
    return chosen(image)

def random_erasing(image, max_size=0.3):
    img_np = np.array(image)
    h, w, _ = img_np.shape
    erase_w = random.randint(int(w * 0.1), int(w * max_size))
    erase_h = random.randint(int(h * 0.1), int(h * max_size))
    x, y = random.randint(0, w - erase_w), random.randint(0, h - erase_h)
    img_np[y:y + erase_h, x:x + erase_w] = random.randint(0, 255)
    return Image.fromarray(img_np)

def hide_and_seek(image, grid_size=4):
    img_np = np.array(image)
    h, w, _ = img_np.shape
    for i in range(grid_size):
        for j in range(grid_size):
            if random.random() < 0.25:
                y1, y2 = i * h // grid_size, (i + 1) * h // grid_size
                x1, x2 = j * w // grid_size, (j + 1) * w // grid_size
                img_np[y1:y2, x1:x2] = 0
    return Image.fromarray(img_np)

def augment_dataset(input_root, output_root=None, N=2):
    if output_root is None or os.path.abspath(input_root) == os.path.abspath(output_root):
        # Создаем временную директорию для аугментации
        temp_dir = tempfile.mkdtemp(prefix="aug_temp_")
        logging.info(f"Аугментация будет производиться во временной папке: {temp_dir}")
        augment_output = temp_dir
        inplace_mode = True
    else:
        augment_output = output_root
        inplace_mode = False
        os.makedirs(augment_output, exist_ok=True)

    logging.info("Запущена аугментация данных...")

    for class_name in tqdm(os.listdir(input_root), desc="Аугментация классов"):
        class_path = os.path.join(input_root, class_name)
        output_class_path = os.path.join(augment_output, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        image_files = [f for f in os.listdir(class_path)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        num_to_augment = max(1, len(image_files) // 2)
        selected_images = random.sample(image_files, num_to_augment)

        logging.info(f"[{class_name}] Будет аугментировано {len(selected_images)} из {len(image_files)} изображений.")

        for file_name in selected_images:
            try:
                img_path = os.path.join(class_path, file_name)
                image = Image.open(img_path).convert("RGB")

                for i in range(N):
                    aug = augment_image(image)
                    base, ext = os.path.splitext(file_name)
                    aug_file = f"{base}_aug_{i + 1}{ext}"
                    aug.save(os.path.join(output_class_path, aug_file))
                    logging.info(f"Аугментировано: {aug_file}")
            except Exception as e:
                logging.error(f"Ошибка с {file_name}: {e}")

    # Если работаем в режиме in-place, переносим аугментированные изображения в train
    if inplace_mode:
        logging.info("Перемещаем аугментированные изображения обратно в train...")
        for class_name in os.listdir(augment_output):
            class_aug_dir = os.path.join(augment_output, class_name)
            class_train_dir = os.path.join(input_root, class_name)
            os.makedirs(class_train_dir, exist_ok=True)
            for f in os.listdir(class_aug_dir):
                shutil.move(os.path.join(class_aug_dir, f), os.path.join(class_train_dir, f))
        shutil.rmtree(augment_output)
        logging.info("Аугментация завершена и файлы перемещены.")

    else:
        logging.info("Аугментация завершена в отдельной папке.")


def get_random_images_with_labels(data_dir, n_images=16, per_class=False):
    image_paths, labels = [], []

    if per_class:
        classes = [cls for cls in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cls))]
        for label in classes:
            class_dir = os.path.join(data_dir, label)
            images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)
                      if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                selected = random.sample(images, 1)
                image_paths.extend(selected)
                labels.extend([label])
    else:
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_path, img_name))
                    labels.append(class_name)

        total = len(image_paths)
        n = min(n_images, total)
        selected_indices = random.sample(range(total), n)
        image_paths = [image_paths[i] for i in selected_indices]
        labels = [labels[i] for i in selected_indices]

    return image_paths, labels


def visualize_batch(image_paths, labels, n_cols=4):
    n_images = len(image_paths)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))  # Адаптивный размер фигуры
    axes = axes.flatten()

    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        image = Image.open(img_path).convert('RGB')
        axes[i].imshow(image)
        axes[i].set_title(label, fontsize=10)
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.3)
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state('zoomed')  # Для Windows: разворачивает окно на весь экран
    except:
        pass
    plt.show()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))  # путь к директории скрипта

    while True:
        print("\nВыберите действие:")
        print("1. Сортировка.")
        print("2. Аугментация.")
        print("3. Визуализация батча.")
        print("0. Выход.")

        choice = input("Введите номер: ").strip()

        if choice == "1":
            source = os.path.join(script_dir, input("Введите путь к исходной папке с изображениями (относительно скрипта): ").strip())
            dest = os.path.join(script_dir, input("Введите путь, куда сохранить split (относительно скрипта): ").strip())
            split_data(source, dest)

        elif choice == "2":
            inp = os.path.join(script_dir, input("Введите путь к train-папке (относительно скрипта): ").strip())
            out_input = input("Введите путь для сохранения аугментированных данных (Enter — в той же папке): ").strip()
            out = os.path.join(script_dir, out_input) if out_input else None
            augment_dataset(inp, out)

        elif choice == "3":
            vis_path = os.path.join(script_dir, input("Введите путь к train-папке для визуализации (относительно скрипта): ").strip())
            if os.path.exists(vis_path):
                mode = input("Показать по 1 изображению на класс? (y/n): ").strip().lower()
                per_class = mode == 'y'
                paths, labels = get_random_images_with_labels(vis_path, n_images=16, per_class=per_class)
                visualize_batch(paths, labels)
            else:
                print("Путь не найден.")

        elif choice == "0":
            print("Выход...")
            break
        else:
            print("Неверный ввод!")


if __name__ == "__main__":
    main()

