import os
import zipfile
import random
import glob
import shutil
from ultralytics import YOLO

# Split images into train/valid (80/20) and write data.yaml for YOLO
extract_path = "./datasets/kyrgyz-plates"
zip_path = "./datasets/kyrgyz-car-license-plates.zip"

# If archive exists and folder missing, extract it
if os.path.exists(zip_path) and not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_path)

images_src = os.path.join(extract_path, "images")
train_images = os.path.join(extract_path, "train", "images")
train_labels = os.path.join(extract_path, "train", "labels")
val_images = os.path.join(extract_path, "valid", "images")
val_labels = os.path.join(extract_path, "valid", "labels")

for p in (train_images, train_labels, val_images, val_labels):
    os.makedirs(p, exist_ok=True)

img_exts = (".jpg", ".jpeg", ".png", ".bmp")
images = [p for p in glob.glob(os.path.join(images_src, "*")) if os.path.splitext(p)[1].lower() in img_exts]
if len(images) == 0:
    print(f"[ERROR] Не найдены изображения в {images_src}")
else:
    random.seed(42)
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_list = images[:split_idx]
    val_list = images[split_idx:]

    def copy_items(lst, dest_img_dir, dest_lbl_dir):
        for img_path in lst:
            base = os.path.basename(img_path)
            dst_img = os.path.join(dest_img_dir, base)
            shutil.copy2(img_path, dst_img)
            lbl_src = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(lbl_src):
                shutil.copy2(lbl_src, os.path.join(dest_lbl_dir, os.path.basename(lbl_src)))

    copy_items(train_list, train_images, train_labels)
    copy_items(val_list, val_images, val_labels)

    print(f"[INFO] Скопировано {len(train_list)} изображений в train, {len(val_list)} в valid")

# Ensure pretrained model is available in project root; download if missing
MODEL_NAME = "yolov8n.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
if not os.path.exists(MODEL_NAME):
    try:
        print(f"[INFO] Скачиваем модель {MODEL_NAME}...")
        import urllib.request
        urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)
        print(f"[INFO] Модель {MODEL_NAME} скачана.")
    except Exception as e:
        print(f"[WARN] Не удалось скачать модель {MODEL_NAME}: {e}")
else:
    print(f"[INFO] Модель {MODEL_NAME} уже есть.")

# Write data.yaml
yaml_path = os.path.join(extract_path, "data.yaml")
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(f"""
path: {extract_path}
train: train/images
val: valid/images

names:
  0: plate
""")

print("[INFO] data.yaml создан/обновлён.")

# === 4. Обучение YOLO ===
print("[INFO] Старт обучения YOLOv8...")
# используем скачанную модель если есть
pretrained = MODEL_NAME if 'MODEL_NAME' in globals() and os.path.exists(MODEL_NAME) else 'yolov8n.pt'
model = YOLO(pretrained)
model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    name="yolo-plates-kg"
)
print("[INFO] Обучение завершено.")
best_weights = os.path.join("runs", "detect", "yolo-plates-kg", "weights", "best.pt")
print("Лучшие веса:", best_weights)
