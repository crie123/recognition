# Распознавание кыргызских автомобильных номеров

Небольшой репозиторий со скриптами для детекции и OCR кыргызских автомобильных номеров.

Содержимое
- create_dets_from_ocr.py  — конвертация CSV результатов OCR в CSV с детекциями
- detect_kyrgyz_plates.py  — запуск детектора (на базе YOLOv8)
- draw_boxes.py            — отрисовка рамок детекций на изображениях
- match_ocr_with_dets.py   — сопоставление результатов OCR с детекциями
- train_kyrgyz_plates.py   — помощник для обучения детектора номеров
- datasets/                — архивы и изображения датасета

Требования
- Python 3.8+
- pandas
- ultralytics / yolov8 (опционально, для детекции/обучения)
- OpenCV (опционально, для отрисовки и работы с изображениями)

Установка

pip install pandas opencv-python
# если планируете запускать детекцию/обучение
pip install ultralytics

Примеры использования

# Конвертация CSV от OCR в CSV детекций
python create_dets_from_ocr.py --ocr-csv path/to/ocr.csv --out detections.csv

# Запуск детектора (пример; зависит от реализации detect_kyrgyz_plates.py)
python detect_kyrgyz_plates.py --source images/ --weights yolov8n.pt --out runs/detect

Примечания
- Репозиторий содержит утилитные скрипты, а не готовое приложение. Просмотрите скрипты и настройте параметры под свою среду.
- Большие файлы и результаты прогонов указаны в .gitignore.
- Finetuned модель доступна по ссылке: https://huggingface.co/crie123/yolov3s-finetuned-kyrgyz-plates