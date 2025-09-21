#!/usr/bin/env python3
# ...new file...
"""
draw_boxes.py

Reads a merged CSV (merged_detections_ocr.csv) with columns including 'image', 'det_bbox', 'ocr_bbox', 'ocr_text', 'det_conf'
Draws boxes and text on the corresponding frame images and saves annotated images to an output directory.
"""
import argparse
import os
from pathlib import Path
import ast
import cv2
import pandas as pd


def parse_bbox(s):
    if pd.isna(s):
        return None
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)) and len(v) == 4:
            return [int(float(x)) for x in v]
    except Exception:
        pass
    parts = [p.strip().strip('"\'') for p in str(s).split(',') if p.strip()]
    if len(parts) == 4:
        try:
            return [int(float(p)) for p in parts]
        except Exception:
            return None
    return None


def draw_on_image(img, dets, ocrs):
    # dets: list of {'bbox': [x1,y1,x2,y2], 'conf': float}
    # ocrs: list of {'bbox': [x1,y1,x2,y2], 'text': str, 'conf': float}
    h, w = img.shape[:2]
    annotated = img.copy()
    # scale thickness with image size
    thickness = max(1, int(round(min(w, h) / 200)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for d in dets:
        x1,y1,x2,y2 = d['bbox']
        color = (0, 200, 0)  # green for det
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, thickness)
        label = f"det {d.get('conf',0):.2f}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, thickness)
        ty = max(0, y1 - 6)
        cv2.rectangle(annotated, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
        cv2.putText(annotated, label, (x1 + 2, ty - 2), font, 0.5, (0,0,0), thickness=1, lineType=cv2.LINE_AA)
    for o in ocrs:
        x1,y1,x2,y2 = o['bbox']
        color = (0, 120, 255)  # orange for ocr
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, thickness)
        text = str(o.get('text',''))
        label = f"ocr:{text} {o.get('conf',0):.2f}"
        # put text below box if possible
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, thickness)
        by = min(h, y2 + th + 8)
        cv2.rectangle(annotated, (x1, by - th - 4), (x1 + tw + 4, by), color, -1)
        cv2.putText(annotated, label, (x1 + 2, by - 2), font, 0.5, (0,0,0), thickness=1, lineType=cv2.LINE_AA)
    return annotated


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--merged-csv', required=True)
    p.add_argument('--frames-dir', required=True)
    p.add_argument('--out-dir', required=False)
    args = p.parse_args()

    merged = pd.read_csv(args.merged_csv, dtype={'image': str})
    merged['det_bbox_parsed'] = merged['det_bbox'].apply(parse_bbox)
    merged['ocr_bbox_parsed'] = merged['ocr_bbox'].apply(parse_bbox)

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir) if args.out_dir else frames_dir.parent / 'annotated_frames'
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = merged.groupby('image')
    for image_name, group in grouped:
        img_path = frames_dir / image_name
        if not img_path.exists():
            print(f"[WARN] Frame not found: {img_path}")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue
        dets = []
        ocrs = []
        for _, row in group.iterrows():
            db = row.get('det_bbox_parsed')
            if db:
                dets.append({'bbox': db, 'conf': float(row.get('det_conf', 0.0)) if 'det_conf' in row else 0.0})
            ob = row.get('ocr_bbox_parsed')
            if ob:
                ocrs.append({'bbox': ob, 'text': row.get('ocr_text',''), 'conf': float(row.get('ocr_conf', 0.0)) if 'ocr_conf' in row else 0.0})
        ann = draw_on_image(img, dets, ocrs)
        out_path = out_dir / image_name
        cv2.imwrite(str(out_path), ann)
    print(f"Wrote annotated frames to {out_dir}")

if __name__ == '__main__':
    main()
