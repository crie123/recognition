#!/usr/bin/env python3
# ...new file...
"""create_dets_from_ocr.py
Reads an OCR results CSV and writes a detections CSV with columns: frame,image,det_conf,det_bbox,det_class
"""
import argparse
import pandas as pd
import ast


def parse_bbox_str(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)) and len(v) == 4:
            return [int(float(x)) for x in v]
    except Exception:
        pass
    parts = [p.strip().strip('"\'') for p in s.split(',') if p.strip()]
    if len(parts) == 4:
        try:
            return [int(float(p)) for p in parts]
        except Exception:
            return None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ocr-csv', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    df = pd.read_csv(args.ocr_csv, dtype={'image': str})
    df['det_bbox'] = df['bbox'].apply(parse_bbox_str)
    df['det_conf'] = df.get('box_conf', 0.0)
    df['det_class'] = 'plate'
    # keep relevant columns
    out = df[['frame', 'image', 'det_conf', 'det_bbox', 'det_class']].copy()
    out.to_csv(args.out, index=False)
    print(f'Wrote {len(out)} detections to {args.out}')

if __name__ == '__main__':
    main()
