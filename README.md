# Kyrgyz License Plate Recognition

Small repository with scripts for detecting and recognizing Kyrgyz car license plates.

Contents
- create_dets_from_ocr.py  — convert OCR CSV output to a detections CSV
- detect_kyrgyz_plates.py  — run detection (YOLOv8 based)
- draw_boxes.py            — draw detection boxes on images
- match_ocr_with_dets.py   — associate OCR results with detections
- train_kyrgyz_plates.py   — training helper for plate detector
- datasets/                — dataset archives and images

Quick requirements
- Python 3.8+
- pandas
- ultralytics / yolov8 (optional, for detection/training)
- OpenCV (optional, for drawing and image IO)

Install example

pip install pandas opencv-python
# install ultralytics if you plan to run detection or training
pip install ultralytics

Examples

# Convert OCR CSV to detections CSV
python create_dets_from_ocr.py --ocr-csv path/to/ocr.csv --out detections.csv

# Run detection (example; depends on detect_kyrgyz_plates.py implementation)
python detect_kyrgyz_plates.py --source images/ --weights yolov8n.pt --out runs/detect

Notes
- This repository contains utility scripts rather than a packaged application. Inspect scripts to adapt parameters for your environment.
- Large files and run outputs are gitignored (see .gitignore).
- Finetuned model available at https://huggingface.co/crie123/yolov3s-finetuned-kyrgyz-plates
