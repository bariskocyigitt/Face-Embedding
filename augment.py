import os, cv2
from tqdm import tqdm
from albumentations import Compose, RandomBrightnessContrast, ShiftScaleRotate, GaussNoise, HorizontalFlip
from utils import ensure_dirs, list_images

augmenter = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=15, p=0.7),
    RandomBrightnessContrast(p=0.7),
    GaussNoise(var_limit=(5.0, 30.0), p=0.3),
])

def augment_all(per_image=5):
    ensure_dirs()
    os.makedirs("data/augmented", exist_ok=True)
    imgs = list_images("data/processed")
    for path in tqdm(imgs, desc="Augmenting"):
        img = cv2.imread(path)
        base = os.path.splitext(os.path.basename(path))[0]
        for i in range(per_image):
            aug = augmenter(image=img)['image']
            outp = os.path.join("data/augmented", f"{base}_aug{i}.jpg")
            cv2.imwrite(outp, aug)
    print("Augment tamam.")

if __name__ == "__main__":
    augment_all(per_image=5)
