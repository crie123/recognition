#!/usr/bin/env python3
# ...new file...
"""
match_ocr_with_dets.py

Usage examples:
  python match_ocr_with_dets.py \
    --ocr-csv "output_clips\almaty-kak-ezdiat-mashiny-s-kirgizskimi-nomerami-ytshorts.savetube.me\ocr_results.csv" \
    --dets-csv detections.csv \
    --out merged.csv

Or, if you have YOLO txt labels (normalized xywh) saved per image:
  python match_ocr_with_dets.py \
    --ocr-csv path/to/ocr_results.csv \
    --labels-dir runs/detect/exp/labels \
    --images-dir output_clips/.../frames \
    --out merged.csv

The script will perform greedy matching by IoU and produce a CSV with merged rows.
"""
import ast
import os
import csv
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
try:
    from PIL import Image
except Exception:
    Image = None


def parse_bbox_str(s):
    # accepts formats like "x1,y1,x2,y2" or "[x1, y1, x2, y2]"
    if pd.isna(s):
        return None
    if isinstance(s, (list, tuple)):
        return [int(v) for v in s]
    s = str(s).strip()
    try:
        # try ast literal eval for list-like
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)) and len(v) == 4:
            return [int(float(x)) for x in v]
    except Exception:
        pass
    # fallback split by comma
    parts = [p.strip().strip('\"\'') for p in s.split(',') if p.strip()]
    if len(parts) == 4:
        try:
            return [int(float(p)) for p in parts]
        except Exception:
            return None
    return None


def iou_xyxy(a, b):
    xa1,ya1,xa2,ya2 = a
    xb1,yb1,xb2,yb2 = b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    area_a = max(0, xa2-xa1) * max(0, ya2-ya1)
    area_b = max(0, xb2-xb1) * max(0, yb2-yb1)
    union = area_a + area_b - inter
    return inter/union if union>0 else 0.0


def read_ocr_csv(path):
    df = pd.read_csv(path, dtype={'image': str})
    df['ocr_bbox'] = df['bbox'].apply(parse_bbox_str)
    # ensure ints
    df = df[df['ocr_bbox'].notnull()].copy()
    df['frame'] = df['frame'].astype(int)
    return df


def read_dets_csv(path):
    df = pd.read_csv(path, dtype={'image': str})
    # expect a column named 'bbox' or 'det_bbox'
    if 'det_bbox' in df.columns:
        df['det_bbox'] = df['det_bbox'].apply(parse_bbox_str)
    elif 'bbox' in df.columns:
        df['det_bbox'] = df['bbox'].apply(parse_bbox_str)
    else:
        raise RuntimeError('Detections CSV must contain a bbox or det_bbox column with x1,y1,x2,y2')
    df = df[df['det_bbox'].notnull()].copy()
    if 'frame' in df.columns:
        df['frame'] = df['frame'].astype(int)
    return df


def read_yolo_labels(labels_dir, images_dir=None, cls_names=None):
    # labels_dir: folder with per-image .txt files (same basename as image)
    # each line: class x_center y_center w h (normalized 0..1)
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir) if images_dir else None
    rows = []
    for p in labels_dir.glob('*.txt'):
        # try to locate corresponding image to get width/height
        img_path = None
        if images_dir:
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                candidate = images_dir / p.with_suffix(ext).name
                if candidate.exists():
                    img_path = candidate
                    break
        else:
            # try to find an image sibling next to the label file
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                c = p.with_suffix(ext)
                if c.exists():
                    img_path = c
                    break
        if img_path and Image:
            try:
                w, h = Image.open(img_path).size
            except Exception:
                w, h = None, None
        else:
            # image size unknown -> skip converting normalized labels
            w, h = None, None
        with p.open('r', encoding='utf8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = parts[0]
                try:
                    xc, yc, ww, hh = map(float, parts[1:5])
                except Exception:
                    continue
                if w and h:
                    x1 = int((xc - ww/2) * w)
                    y1 = int((yc - hh/2) * h)
                    x2 = int((xc + ww/2) * w)
                    y2 = int((yc + hh/2) * h)
                else:
                    # cannot convert without image size
                    continue
                frame = None
                # try parse frame number from image filename if like frame_123.jpg
                basename = p.with_suffix('.jpg').name
                import re
                m = re.search(r"(\d+)", basename)
                try:
                    frame = int(m.group(1)) if m else None
                except Exception:
                    frame = None
                rows.append({'image': basename, 'frame': frame, 'det_bbox': [x1,y1,x2,y2], 'det_class': cls})
    return pd.DataFrame(rows)


def greedy_match(dets, ocrs, iou_thresh=0.3):
    matches = []
    if dets.empty or ocrs.empty:
        return matches
    D = len(dets)
    O = len(ocrs)
    iou_mat = np.zeros((D, O))
    for i, drow in dets.reset_index(drop=True).iterrows():
        for j, orow in ocrs.reset_index(drop=True).iterrows():
            iou_mat[i, j] = iou_xyxy(drow['det_bbox'], orow['ocr_bbox'])
    used_i = set()
    used_j = set()
    while True:
        idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        i, j = int(idx[0]), int(idx[1])
        best = iou_mat[i, j]
        if best <= 0 or best < iou_thresh:
            break
        drow = dets.reset_index(drop=True).iloc[i]
        orow = ocrs.reset_index(drop=True).iloc[j]
        matches.append({
            'frame': int(drow.get('frame') if 'frame' in drow else orow.get('frame')),
            'image': drow.get('image', orow.get('image', '')),
            'det_class': drow.get('det_class', ''),
            'det_conf': drow.get('det_conf', 0.0),
            'det_bbox': drow['det_bbox'],
            'ocr_text': orow.get('ocr_text', ''),
            'ocr_conf': orow.get('ocr_conf', 0.0),
            'ocr_bbox': orow['ocr_bbox'],
            'iou': float(best)
        })
        iou_mat[i, :] = -1
        iou_mat[:, j] = -1
    return matches


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ocr-csv', required=True, help='path to ocr_results.csv')
    p.add_argument('--dets-csv', help='optional detections CSV with bbox column')
    p.add_argument('--labels-dir', help='optional YOLO labels dir (txt files)')
    p.add_argument('--images-dir', help='images dir used to convert normalized labels to pixels (required for labels-dir)')
    p.add_argument('--iou-threshold', type=float, default=0.3)
    p.add_argument('--out', default='merged_detections_ocr.csv')
    args = p.parse_args()

    ocr = read_ocr_csv(args.ocr_csv)
    if args.dets_csv:
        dets = read_dets_csv(args.dets_csv)
    elif args.labels_dir:
        dets = read_yolo_labels(args.labels_dir, images_dir=args.images_dir)
    else:
        raise SystemExit('Provide either --dets-csv or --labels-dir')

    # ensure frame column exists on both
    if 'frame' not in dets.columns:
        dets['frame'] = None
    if 'frame' not in ocr.columns:
        ocr['frame'] = None

    all_matches = []
    frames = sorted(set(list(ocr['frame'].dropna().unique()) + list(dets['frame'].dropna().unique())))
    for f in frames:
        dets_f = dets[dets['frame'] == f] if f is not None else dets[dets['frame'].isnull()]
        ocrs_f = ocr[ocr['frame'] == f] if f is not None else ocr[ocr['frame'].isnull()]
        mm = greedy_match(dets_f, ocrs_f, iou_thresh=args.iou_threshold)
        all_matches.extend(mm)

    out_df = pd.DataFrame(all_matches)
    out_df.to_csv(args.out, index=False)
    print(f'Wrote {len(out_df)} matched rows to {args.out}')


if __name__ == '__main__':
    main()
