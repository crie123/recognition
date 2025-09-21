# filepath: d:\recognition\detect_kyrgyz_plates.py
import cv2
import os
from ultralytics import YOLO
from moviepy import VideoFileClip
from collections import deque
import glob
import subprocess
import sys
import tempfile

# === Настройки ===
# VIDEO_SOURCE can be a single file or a directory with subfolders (e.g. "20250814")
VIDEO_SOURCE = "20250814"
OUTPUT_DIR = "output_clips"
MODEL_PATH = "runs/detect/yolo-plates-kg3/weights/best.pt"  # обученная модель
CLIP_DURATION = 5  # секунды до и после обнаружения для сохранения фрагмента

# Automatic trimming: enable and set start/duration (seconds)
TRIM_ENABLED = True
TRIM_START_SECONDS = 0  # 2 minutes
TRIM_DURATION_SECONDS = 60  # 30 seconds (so segment is 2:00..2:30)

# Control whether temporary files/frames are deleted after processing
CLEANUP_TMP = False  # <-- set to False to disable cleanup of extracted frames / temp transcodes

# Path to ffmpeg executable (local copy in project). You can override with an absolute path.
FFMPEG_BIN = os.path.join(os.getcwd(), "ffmpeg-2025-08-18-git-0226b6fb2c-full_build", "bin", "ffmpeg.exe")
# Optional VLC path (used as last resort). Common default on Windows:
VLC_BIN = r"C:\Program Files\VideoLAN\VLC\vlc.exe" if os.path.exists(r"C:\Program Files\VideoLAN\VLC\vlc.exe") else None

# Предобработка (для ночного/ИК/бликов)
CLAHE_CLIP = 3.0
CLAHE_TILE = (8, 8)
GAMMA = 0.9
DENOISE_H = 10
TEMPORAL_WINDOW = 5  # количество кадров для медианного усреднения (1 отключает)

# Dynamic/enable flags: set to False to disable a step, or enable dynamic behavior
ENABLE_CLAHE = True         # применять CLAHE
ENABLE_INPAINT = True       # удаление бликов и инпейнт
ENABLE_DENOISE = True       # применять denoise
ENABLE_GAMMA = True         # применять гамма-коррекцию
DYNAMIC_GAMMA = True        # если True, GAMMA будет рассчитываться по яркости кадра
GAMMA_MIN = 0.7
GAMMA_MAX = 1.6
TARGET_MEAN_V = 120.0  # цель по среднему значению V (яркости) для динамической гаммы

prev_frames = deque(maxlen=TEMPORAL_WINDOW)

# Создаём выходные папки
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === EasyOCR dynamic import/installation ===
import importlib
try:
    easyocr = importlib.import_module('easyocr')
except Exception:
    print("[INFO] Устанавливаем easyocr и opencv-python-headless...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "easyocr", "opencv-python-headless"])
    easyocr = importlib.import_module('easyocr')

# Создаём reader (по умолчанию без GPU)
reader = easyocr.Reader(['en', 'ru'], gpu=False)

import numpy as np
import csv

def adjust_gamma(img, gamma=1.0):
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv * 255.0
    table = table.astype("uint8")
    return cv2.LUT(img, table)

def remove_specular_and_inpaint(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Маска бликов: яркие и низко насыщенные области
    _, mask_v = cv2.threshold(v, 240, 255, cv2.THRESH_BINARY)
    _, mask_s = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_and(mask_v, mask_s)
    if mask.sum() == 0:
        return bgr
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)), iterations=1)
    inpainted = cv2.inpaint(bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted

def apply_clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def preprocess_frame(frame):
    # temporal median filtering to reduce flicker/noise
    if TEMPORAL_WINDOW > 1:
        prev_frames.append(frame.copy())
        if len(prev_frames) == TEMPORAL_WINDOW:
            stack = np.stack(list(prev_frames), axis=0)
            frame_med = np.median(stack, axis=0).astype(np.uint8)
        else:
            frame_med = frame
    else:
        frame_med = frame

    # remove specular highlights and inpaint
    frame_ip = remove_specular_and_inpaint(frame_med) if ENABLE_INPAINT else frame_med

    # gamma correction
    if ENABLE_GAMMA:
        use_gamma = GAMMA
        if DYNAMIC_GAMMA:
            try:
                hsv = cv2.cvtColor(frame_ip, cv2.COLOR_BGR2HSV)
                _, _, v = cv2.split(hsv)
                mean_v = float(np.mean(v))
                # Map mean_v -> gamma in [GAMMA_MIN, GAMMA_MAX]. If frame is darker than TARGET_MEAN_V, choose gamma < 1 to brighten (towards GAMMA_MIN),
                # if brighter, choose gamma > 1 to slightly darken (towards GAMMA_MAX). Linear interpolation around TARGET_MEAN_V.
                if mean_v <= TARGET_MEAN_V:
                    t = 1.0 - (mean_v / (TARGET_MEAN_V + 1e-6))
                    use_gamma = float(np.clip(GAMMA_MIN + (1.0 - GAMMA_MIN) * (1.0 - (1.0 - t)), GAMMA_MIN, GAMMA_MAX))
                else:
                    t = (mean_v - TARGET_MEAN_V) / (255.0 - TARGET_MEAN_V + 1e-6)
                    use_gamma = float(np.clip(1.0 + (GAMMA_MAX - 1.0) * t, GAMMA_MIN, GAMMA_MAX))
            except Exception:
                use_gamma = GAMMA
        frame_gamma = adjust_gamma(frame_ip, use_gamma)
    else:
        frame_gamma = frame_ip

    # local contrast enhancement
    frame_clahe = apply_clahe(frame_gamma) if ENABLE_CLAHE else frame_gamma

    # denoise while preserving edges
    frame_denoised = frame_clahe
    if ENABLE_DENOISE:
        frame_denoised = cv2.fastNlMeansDenoisingColored(frame_clahe, None, h=DENOISE_H, hColor=DENOISE_H, templateWindowSize=7, searchWindowSize=21)

    return frame_denoised

# === Хелперы для видео и ffmpeg ===

def ffmpeg_available():
    # Prefer explicit FFMPEG_BIN if it exists, otherwise try system ffmpeg
    if 'FFMPEG_BIN' in globals() and FFMPEG_BIN and os.path.exists(FFMPEG_BIN):
        cmd = [FFMPEG_BIN, "-version"]
    else:
        cmd = ["ffmpeg", "-version"]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def ensure_compatible_video(path):
    """Return a path that OpenCV can open. Try multiple backends (FFMPEG, MSMF, DSHOW).
    If none work, try (in order): remux to MKV (copy), extract raw .h265 stream, transcode to H.264 (ffmpeg),
    finally fallback to VLC transcode if available. Returns (usable_path, tmp_created_path_or_None).
    """
    # Try OpenCV backends first
    backends = []
    try:
        backends.append(cv2.CAP_FFMPEG)
    except Exception:
        pass
    backends.extend([cv2.CAP_MSMF, cv2.CAP_DSHOW])

    for backend in backends:
        try:
            cap = cv2.VideoCapture(path, backend)
        except Exception:
            cap = cv2.VideoCapture(path)
        ok, _ = cap.read()
        cap.release()
        if ok:
            return path, None

    # Try default open
    cap = cv2.VideoCapture(path)
    ok, _ = cap.read()
    cap.release()
    if ok:
        return path, None

    # Prepare tmp directory
    tmp_dir = os.path.join(OUTPUT_DIR, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]

    # Helper to run ffmpeg command
    def run_ffmpeg(cmd):
        try:
            print(f"[INFO] Running ffmpeg: {' '.join(cmd)}")
            subprocess.check_call(cmd)
            return True
        except Exception as e:
            print(f"[WARN] ffmpeg step failed: {e}")
            return False

    # Ensure we have ffmpeg
    if not ffmpeg_available():
        print("[WARN] ffmpeg not available; will try VLC if present.")
    else:
        ffmpeg_exe = FFMPEG_BIN if ('FFMPEG_BIN' in globals() and FFMPEG_BIN and os.path.exists(FFMPEG_BIN)) else 'ffmpeg'

        # 1) Try remux (copy streams) to MKV
        remux_path = os.path.join(tmp_dir, f"{base}_remux.mkv")
        cmd_remux = [ffmpeg_exe, "-probesize", "50M", "-analyzeduration", "100M", "-i", path, "-c", "copy", remux_path]
        if run_ffmpeg(cmd_remux):
            cap2 = cv2.VideoCapture(remux_path)
            ok2, _ = cap2.read()
            cap2.release()
            if ok2:
                return remux_path, remux_path
            else:
                print("[WARN] Remux produced file but OpenCV still cannot read it.")

        # 2) Try extract raw video stream (.h265) then transcode that
        raw_h265 = os.path.join(tmp_dir, f"{base}.h265")
        cmd_extract = [ffmpeg_exe, "-probesize", "50M", "-analyzeduration", "100M", "-i", path, "-map", "0:0", "-c", "copy", raw_h265]
        if run_ffmpeg(cmd_extract) and os.path.exists(raw_h265):
            # transcode raw h265 to mp4
            from_h265 = os.path.join(tmp_dir, f"{base}_from_h265.mp4")
            cmd_from_h265 = [ffmpeg_exe, "-f", "hevc", "-i", raw_h265, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an", from_h265]
            if run_ffmpeg(cmd_from_h265):
                cap3 = cv2.VideoCapture(from_h265)
                ok3, _ = cap3.read()
                cap3.release()
                if ok3:
                    # cleanup raw_h265
                    try:
                        os.remove(raw_h265)
                    except Exception:
                        pass
                    return from_h265, from_h265

        # 3) Full transcode to H.264 mp4
        transcode_path = os.path.join(tmp_dir, f"{base}_transcode.mp4")
        cmd_trans = [ffmpeg_exe, "-probesize", "50M", "-analyzeduration", "100M", "-i", path, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an", transcode_path]
        if run_ffmpeg(cmd_trans):
            cap4 = cv2.VideoCapture(transcode_path)
            ok4, _ = cap4.read()
            cap4.release()
            if ok4:
                return transcode_path, transcode_path
            else:
                print("[WARN] Full transcode produced file but OpenCV still cannot read it.")

    # If ffmpeg strategies failed, try VLC if available
    vlc_used = None
    if 'VLC_BIN' in globals() and VLC_BIN and os.path.exists(VLC_BIN):
        vlc_exe = VLC_BIN
    else:
        # try common locations
        common_vlc = [r"C:\Program Files\VideoLAN\VLC\vlc.exe", r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe"]
        vlc_exe = next((p for p in common_vlc if os.path.exists(p)), None)

    if vlc_exe:
        out_vlc = os.path.join(tmp_dir, f"{base}_vlc.mp4")
        cmd_vlc = [vlc_exe, "-I", "dummy", path, "--sout=#transcode{vcodec=h264,vb=1500,acodec=none}:std{access=file,mux=mp4,dst=\"%s\"}" % out_vlc, "vlc://quit"]
        try:
            print(f"[INFO] Trying VLC transcode: {vlc_exe}")
            subprocess.check_call(cmd_vlc)
            capv = cv2.VideoCapture(out_vlc)
            okv, _ = capv.read()
            capv.release()
            if okv:
                vlc_used = out_vlc
                return out_vlc, out_vlc
            else:
                print("[WARN] VLC transcode produced file but OpenCV cannot read it.")
        except Exception as e:
            print(f"[WARN] VLC transcode failed: {e}")

    # All attempts failed
    raise RuntimeError(f"Cannot open or transcode {path} with available tools (ffmpeg/vlc).")

def ffprobe_get_fps(path):
    """Return framerate as float using ffprobe, or None."""
    ffprobe_exe = None
    if 'FFMPEG_BIN' in globals() and FFMPEG_BIN and os.path.exists(FFMPEG_BIN):
        ffprobe_exe = os.path.join(os.path.dirname(FFMPEG_BIN), 'ffprobe.exe')
    if not ffprobe_exe or not os.path.exists(ffprobe_exe):
        ffprobe_exe = 'ffprobe'
    try:
        cmd = [ffprobe_exe, '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', path]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        out = out.strip()
        if '/' in out:
            num, den = out.split('/')
            return float(num) / float(den)
        return float(out)
    except Exception:
        return None


def extract_frames_with_ffmpeg(path, tmp_dir, fps=None):
    """Extract frames to tmp_dir using ffmpeg; return True on success.
    By default extracts 5 frames per second unless fps is provided.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    ffmpeg_exe = FFMPEG_BIN if ('FFMPEG_BIN' in globals() and FFMPEG_BIN and os.path.exists(FFMPEG_BIN)) else 'ffmpeg'
    # build command: use large probesize/analyzeduration to help weird containers
    cmd = [ffmpeg_exe, '-probesize', '50M', '-analyzeduration', '100M', '-i', path]
    # Default to extracting 5 frames per second if caller doesn't request another rate
    use_fps = fps if fps is not None else 5
    if use_fps:
        cmd += ['-vf', f'fps={use_fps}']
    cmd += ['-q:v', '2', os.path.join(tmp_dir, 'frame_%08d.jpg')]
    try:
        print(f"[INFO] Extracting frames with ffmpeg to {tmp_dir} (fps={use_fps})...")
        subprocess.check_call(cmd)
        # verify at least one frame
        imgs = glob.glob(os.path.join(tmp_dir, 'frame_*.jpg'))
        return len(imgs) > 0
    except Exception as e:
        print(f"[WARN] ffmpeg frame extraction failed: {e}")
        return False

# === 1. Загружаем модель ===
model = YOLO(MODEL_PATH)

# === 2. Сбор списка видео ===
video_list = []
if os.path.isdir(VIDEO_SOURCE):
    # common extensions
    exts = ("*.mp4", "*.avi", "*.mkv", "*.mov")
    for ext in exts:
        video_list.extend(glob.glob(os.path.join(VIDEO_SOURCE, "**", ext), recursive=True))
    video_list = sorted(video_list)
else:
    video_list = [VIDEO_SOURCE]

if not video_list:
    print(f"[ERROR] Видео не найдены по пути {VIDEO_SOURCE}")
    sys.exit(1)

print(f"[INFO] Найдено {len(video_list)} видео для обработки")

# Обрабатываем каждое видео по очереди, создавая поддиректорию в OUTPUT_DIR
for vid_idx, video_path in enumerate(video_list):
    print(f"[INFO] Обработка ({vid_idx+1}/{len(video_list)}): {video_path}")
    try:
        # First, ensure we have a path OpenCV/ffmpeg can read (may remux/transcode full file)
        use_path, tmp_created = ensure_compatible_video(video_path)

        # If trimming is enabled and use_path is a real video file, create a trimmed temporary file from use_path
        proc_video_path = use_path
        if TRIM_ENABLED and os.path.isfile(use_path):
            tmp_dir = os.path.join(OUTPUT_DIR, "_tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            ffmpeg_exe = FFMPEG_BIN if ('FFMPEG_BIN' in globals() and FFMPEG_BIN and os.path.exists(FFMPEG_BIN)) else 'ffmpeg'
            base = os.path.splitext(os.path.basename(video_path))[0]
            trimmed_path = os.path.join(tmp_dir, f"{base}_trim.mp4")

            if not os.path.exists(trimmed_path):
                # try fast trim by stream copy
                cmd_copy = [ffmpeg_exe, '-y', '-ss', str(TRIM_START_SECONDS), '-i', use_path, '-t', str(TRIM_DURATION_SECONDS), '-c', 'copy', trimmed_path]
                try:
                    print(f"[INFO] Trimming (copy) {use_path} -> {trimmed_path} ({TRIM_START_SECONDS}s..+{TRIM_DURATION_SECONDS}s)")
                    subprocess.check_call(cmd_copy)
                except Exception:
                    # fallback to re-encode to H.264 if copy fails
                    cmd_reenc = [ffmpeg_exe, '-y', '-ss', str(TRIM_START_SECONDS), '-i', use_path, '-t', str(TRIM_DURATION_SECONDS), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-c:a', 'copy', trimmed_path]
                    try:
                        print(f"[INFO] Stream-copy trim failed, re-encoding trim to {trimmed_path}...")
                        subprocess.check_call(cmd_reenc)
                    except Exception as e:
                        print(f"[WARN] Trim failed; will continue processing the full usable file: {e}")
            if os.path.exists(trimmed_path):
                proc_video_path = trimmed_path

    except Exception as e:
        print(f"[WARN] Пропускаем {video_path}: {e}")
        continue

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_root = os.path.join(OUTPUT_DIR, base_name)
    frames_dir = os.path.join(out_root, "frames")
    clips_dir = os.path.join(out_root, "clips")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)

    # === 3. Читаем видео и делаем инференс ===
    # If use_path is a frames directory (from extract_frames_with_ffmpeg), iterate images instead of VideoCapture
    is_frames_dir = os.path.isdir(use_path) and len(glob.glob(os.path.join(use_path, 'frame_*.jpg'))) > 0
    if is_frames_dir:
        # get fps from original video or default
        fps_val = ffprobe_get_fps(video_path) or 25.0
        frame_files = sorted(glob.glob(os.path.join(use_path, 'frame_*.jpg')))
        total_frames = len(frame_files)
        print(f"[INFO] Processing extracted frames: {total_frames} images, assumed FPS={fps_val}")

        ocr_results = []
        detections = []
        for frame_idx, frame_file in enumerate(frame_files):
            frame = cv2.imread(frame_file)
            if frame is None:
                continue
            proc = preprocess_frame(frame) if 'preprocess_frame' in globals() else frame
            results = model(proc)
            for r in results:
                boxes = r.boxes
                if len(boxes) > 0:
                    frame_name = os.path.basename(frame_file)
                    det_time = frame_idx / fps_val
                    detections.append(det_time)
                    print(f"[INFO] Найден номер на кадре {frame_idx} (время {det_time:.2f}s)")

                    try:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                    except Exception:
                        xyxy = np.array(boxes.xyxy)
                        confs = np.array(boxes.conf)

                    for (box, bconf) in zip(xyxy, confs):
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        crop = proc[y1:y2, x1:x2]
                        try:
                            ocr_out = reader.readtext(crop)
                        except Exception as e:
                            print("[WARN] EasyOCR failed on crop:", e)
                            ocr_out = []

                        if ocr_out:
                            best = max(ocr_out, key=lambda x: x[2])
                            text = best[1].replace(',', ' ')
                            tconf = float(best[2])
                        else:
                            text = ""
                            tconf = 0.0

                        ocr_results.append({
                            'frame': frame_idx,
                            'time_s': round(det_time, 3),
                            'image': frame_name,
                            'box_conf': float(bconf),
                            'ocr_text': text,
                            'ocr_conf': tconf,
                            'bbox': f"{x1},{y1},{x2},{y2}"
                        })

        # Save OCR results
        ocr_csv = os.path.join(out_root, "ocr_results.csv")
        with open(ocr_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['frame', 'time_s', 'image', 'box_conf', 'ocr_text', 'ocr_conf', 'bbox'])
            writer.writeheader()
            for row in ocr_results:
                writer.writerow(row)

        print(f"[INFO] OCR results saved to {ocr_csv}")

        # For clips, use ffmpeg to cut original video at detections (ffmpeg must be able to read original)
        if detections:
            ffmpeg_exe = FFMPEG_BIN if ('FFMPEG_BIN' in globals() and FFMPEG_BIN and os.path.exists(FFMPEG_BIN)) else 'ffmpeg'
            for i, det_time in enumerate(detections):
                start = max(0, det_time - CLIP_DURATION)
                duration = min(CLIP_DURATION*2, 99999)
                out_clip = os.path.join(clips_dir, f"clip_{i}.mp4")
                cmd = [ffmpeg_exe, '-ss', str(start), '-i', video_path, '-t', str(CLIP_DURATION*2), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-an', out_clip]
                try:
                    subprocess.check_call(cmd)
                except Exception as e:
                    print(f"[WARN] ffmpeg clip failed: {e}")
            print(f"[INFO] Сохранено {len(detections)} клипов в {clips_dir}.")
        else:
            print("[INFO] Номера не найдены.")

        # cleanup tmp frames if needed
        if tmp_created and CLEANUP_TMP:
            try:
                for f in glob.glob(os.path.join(use_path, 'frame_*.jpg')):
                    os.remove(f)
                os.rmdir(use_path)
            except Exception:
                pass

        continue

    # === 3. Читаем видео и делаем инференс ===
    cap = cv2.VideoCapture(use_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] Видео: {total_frames} кадров, FPS={fps}")

    ocr_results = []
    detections = []

    frame_idx = 0
    # reset temporal buffer per video
    prev_frames.clear()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        proc = preprocess_frame(frame) if 'preprocess_frame' in globals() else frame

        results = model(proc)

        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                frame_name = f"frame_{frame_idx}.jpg"
                frame_path = os.path.join(frames_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                det_time = frame_idx / fps
                detections.append(det_time)
                print(f"[INFO] Найден номер на кадре {frame_idx} (время {det_time:.2f}s)")

                try:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                except Exception:
                    xyxy = np.array(boxes.xyxy)
                    confs = np.array(boxes.conf)

                for (box, bconf) in zip(xyxy, confs):
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = proc[y1:y2, x1:x2]
                    try:
                        ocr_out = reader.readtext(crop)
                    except Exception as e:
                        print("[WARN] EasyOCR failed on crop:", e)
                        ocr_out = []

                    if ocr_out:
                        best = max(ocr_out, key=lambda x: x[2])
                        text = best[1].replace(',', ' ')
                        tconf = float(best[2])
                    else:
                        text = ""
                        tconf = 0.0

                    ocr_results.append({
                        'frame': frame_idx,
                        'time_s': round(det_time, 3),
                        'image': frame_name,
                        'box_conf': float(bconf),
                        'ocr_text': text,
                        'ocr_conf': tconf,
                        'bbox': f"{x1},{y1},{x2},{y2}"
                    })

        frame_idx += 1

    cap.release()

    # Сохраняем OCR результаты в CSV
    ocr_csv = os.path.join(out_root, "ocr_results.csv")
    with open(ocr_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['frame', 'time_s', 'image', 'box_conf', 'ocr_text', 'ocr_conf', 'bbox'])
        writer.writeheader()
        for row in ocr_results:
            writer.writerow(row)

    print(f"[INFO] OCR results saved to {ocr_csv}")

    # Нарезка клипов
    if detections:
        try:
            video = VideoFileClip(use_path)
            for i, det_time in enumerate(detections):
                start = max(0, det_time - CLIP_DURATION)
                end = min(video.duration, det_time + CLIP_DURATION)
                clip = video.subclip(start, end)
                clip.write_videofile(os.path.join(clips_dir, f"clip_{i}.mp4"), codec="libx264", verbose=False, logger=None)
            print(f"[INFO] Сохранено {len(detections)} клипов в {clips_dir}.")
        except Exception as e:
            print(f"[WARN] Не удалось нарезать клипы через moviepy: {e}")
    else:
        print("[INFO] Номера не найдены.")

    # cleanup tmp
    if tmp_created and CLEANUP_TMP:
        try:
            os.remove(tmp_created)
        except Exception:
            pass

print("[INFO] Обработка всех видео завершена.")
