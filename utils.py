import os, cv2
from datetime import datetime

NEEDED_DIRS = [
    "data/raw", "data/processed", "data/augmented",
    "data/outputs", "embeddings", "keys"
]

def ensure_dirs():
    for p in NEEDED_DIRS:
        os.makedirs(p, exist_ok=True)

def timestamp():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def save_image(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]
