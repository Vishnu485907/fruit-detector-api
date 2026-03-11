"""
Fruit Adulteration Detection API
FastAPI backend wrapping YOLO + Keras classifier pipeline
Deploy on Render.com
"""

import os
import cv2
import numpy as np
import base64
import io
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import torch
import ultralytics
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

# ── lazy-load heavy deps so Render startup is faster ──
_yolo = None
_classifier = None

YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL", "yolov8n.pt")
CLASSIFIER_PATH = os.environ.get("CLASSIFIER_MODEL", "models/fruit_adulteration_final.h5")

IMG_SIZE = 224
CROP_PADDING = 8
LABEL_ORDER = ['adulterated', 'fresh', 'rotten']   # ← match your training order
CLASS_NAMES = LABEL_ORDER.copy()

CLASS_COLORS_BGR = {
    'fresh':      (94, 197, 34),
    'rotten':     (68,  68, 239),
    'adulterated':(11, 158, 245),
}

PREPROCESSING_MODES = ['tf', 'mobilenet', 'imagenet']
IMAGENET_MEAN_RGB = np.array([123.68, 116.779, 103.939], dtype=np.float32)

# ───────────────────────── App ─────────────────────────

app = FastAPI(
    title="Fruit Adulteration Detector",
    description="YOLO + Keras classifier for fresh / rotten / adulterated fruit",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────── Model loading ─────────────────────────

def get_yolo():
    global _yolo
    if _yolo is None:
        import torch
        import ultralytics
        torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
        from ultralytics import YOLO
        _yolo = YOLO(YOLO_MODEL_PATH)
        print("✓ YOLO loaded")
    return _yolo

def get_classifier():
    global _classifier
    if _classifier is None:
        from tensorflow import keras
        if not os.path.exists(CLASSIFIER_PATH):
            raise FileNotFoundError(f"Classifier not found: {CLASSIFIER_PATH}")
        _classifier = keras.models.load_model(CLASSIFIER_PATH)
        print("✓ Classifier loaded")
    return _classifier

# ───────────────────────── Helpers ─────────────────────────

def preprocess_crop(crop_bgr: np.ndarray, mode: str) -> np.ndarray:
    img = cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    if mode == 'tf':
        img = img / 255.0
    elif mode == 'mobilenet':
        img = img / 127.5 - 1.0
    elif mode == 'imagenet':
        img = img - IMAGENET_MEAN_RGB
    else:
        img = img / 255.0
    return img.astype(np.float32)


def select_preprocessing_mode(crops: list) -> str:
    if not crops:
        return 'tf'
    clf = get_classifier()
    mode_scores = {}
    for mode in PREPROCESSING_MODES:
        batch = np.stack([preprocess_crop(c, mode) for c in crops])
        try:
            preds = clf.predict(batch, verbose=0)
        except Exception:
            preds = np.zeros((len(crops), len(CLASS_NAMES)), dtype=np.float32)
        max_conf = float(np.mean(np.max(preds, axis=1)))
        pred_idx = np.argmax(preds, axis=1)
        counts = np.bincount(pred_idx, minlength=len(CLASS_NAMES))
        max_frac = float(np.max(counts) / np.sum(counts))
        mode_scores[mode] = max_conf * (1.0 - max_frac)
    return max(mode_scores, key=mode_scores.get)


def scale_params(image: np.ndarray):
    h, w = image.shape[:2]
    diag = (w*w + h*h) ** 0.5
    font_scale = max(0.4, diag / 3000)
    thickness  = max(1, int(diag // 900))
    padding    = max(6, int(diag // 220))
    return font_scale, thickness, padding


def draw_transparent_rect(image, pt1, pt2, color, alpha=0.45):
    overlay = image.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def annotate_image(image: np.ndarray, detections: list) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls   = det['class']
        conf  = det['class_conf']
        color = CLASS_COLORS_BGR.get(cls, (200, 200, 200))
        font_scale, thickness, pad = scale_params(image)

        # bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # label background + text
        label = f"{cls.upper()} {int(conf*100)}%"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        lh = th + base + pad * 2
        lx1 = x1
        ly2 = max(lh, y1 - 4)
        ly1 = ly2 - lh
        if ly1 < 0:
            ly1, ly2 = y2 + 4, y2 + 4 + lh
        lx2 = lx1 + tw + pad * 2
        h_, w_ = image.shape[:2]
        lx2 = min(lx2, w_ - 2)

        draw_transparent_rect(image, (lx1, ly1), (lx2, ly2), color, alpha=0.55)
        cv2.putText(image, label, (lx1 + pad, ly2 - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)

        # confidence bar
        bar_w = max(40, x2 - x1)
        filled = int(bar_w * conf)
        bh = max(6, int(min(image.shape[:2]) / 160))
        by1 = min(image.shape[0] - bh - 2, y2 + lh + 6)
        draw_transparent_rect(image, (x1, by1), (x1 + bar_w, by1 + bh), (50,50,50), alpha=0.35)
        cv2.rectangle(image, (x1, by1), (x1 + filled, by1 + bh), color, -1)

    # summary banner
    counts = {cls: sum(1 for d in detections if d['class'] == cls) for cls in CLASS_NAMES}
    mode = detections[0].get('mode', 'auto') if detections else 'auto'
    summary = (f"Detected:{len(detections)}  "
               f"Fresh:{counts['fresh']}  "
               f"Rotten:{counts['rotten']}  "
               f"Adulterated:{counts['adulterated']}")
    fs, th2, pd2 = scale_params(image)
    (sw, sth), sbl = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, fs, th2)
    draw_transparent_rect(image, (10, 10), (14 + sw + pd2*2, 14 + sth + pd2*2), (0,0,0), alpha=0.55)
    cv2.putText(image, summary, (12 + pd2, 12 + sth + pd2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), th2, cv2.LINE_AA)
    return image


def image_to_base64(img_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buffer).decode('utf-8')

# ───────────────────────── Routes ─────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Fruit Adulteration Detector API"}


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Upload a fruit image. Returns:
      - annotated_image: base64 JPEG
      - detections: list of per-object results
      - summary: counts per class
    """
    # ── read image ──
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    original = image.copy()
    h, w = image.shape[:2]

    # ── YOLO detection ──
    yolo = get_yolo()
    yolo_results = yolo(image, conf=0.25, verbose=False)

    raw_detections = []
    for res in yolo_results:
        for box in res.boxes:
            try:
                coords = box.xyxy[0].cpu().numpy()
            except Exception:
                coords = box.xyxy[0]
            x1, y1, x2, y2 = map(int, coords)
            cls_id = int(box.cls[0])
            raw_detections.append({
                'bbox': [x1, y1, x2, y2],
                'yolo_conf': float(box.conf[0]),
                'yolo_class': yolo.names[cls_id] if hasattr(yolo, 'names') else str(cls_id),
            })

    if not raw_detections:
        # return original image with "no detection" overlay
        annotated = original.copy()
        cv2.putText(annotated, "No fruits detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        return {
            "annotated_image": image_to_base64(annotated),
            "detections": [],
            "summary": {cls: 0 for cls in CLASS_NAMES},
            "mode": "n/a",
            "message": "No objects detected by YOLO."
        }

    # ── collect crops ──
    crops = []
    for det in raw_detections:
        x1, y1, x2, y2 = det['bbox']
        x1c = max(0, x1 - CROP_PADDING);  y1c = max(0, y1 - CROP_PADDING)
        x2c = min(w, x2 + CROP_PADDING);  y2c = min(h, y2 + CROP_PADDING)
        crops.append(original[y1c:y2c, x1c:x2c])

    # ── auto-select preprocessing mode ──
    chosen_mode = select_preprocessing_mode(crops)

    # ── classify ──
    clf = get_classifier()
    batch = np.stack([preprocess_crop(c, chosen_mode) for c in crops])
    preds = clf.predict(batch, verbose=0)

    # ── assemble final detections ──
    final_detections = []
    for i, (det, pred) in enumerate(zip(raw_detections, preds)):
        idx = int(np.argmax(pred))
        cls_name = CLASS_NAMES[idx]
        final_detections.append({
            'bbox':       det['bbox'],
            'yolo_class': det['yolo_class'],
            'yolo_conf':  round(det['yolo_conf'], 4),
            'class':      cls_name,
            'class_conf': round(float(pred[idx]), 4),
            'mode':       chosen_mode,
            'probabilities': {
                CLASS_NAMES[j]: round(float(pred[j]), 4)
                for j in range(len(CLASS_NAMES))
            },
        })

    # ── annotate ──
    annotated = annotate_image(original.copy(), final_detections)

    summary = {cls: sum(1 for d in final_detections if d['class'] == cls)
               for cls in CLASS_NAMES}

    return {
        "annotated_image": image_to_base64(annotated),
        "detections": final_detections,
        "summary": summary,
        "mode": chosen_mode,
        "total": len(final_detections),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
