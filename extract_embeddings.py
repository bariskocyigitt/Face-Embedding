import os, cv2, numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
from utils import ensure_dirs, list_images

# Bu script SADECE embedding çıkarır. Karar/karşılaştırma/threshold YOK.
def run(folder_in, folder_out):
    os.makedirs(folder_out, exist_ok=True)
    imgs = list_images(folder_in)
    if not imgs:
        print("Görüntü bulunamadı:", folder_in); return
    print("Model yükleniyor...")
    app = FaceAnalysis(allowed_modules=['detection','recognition'])
    app.prepare(ctx_id=-1, det_size=(640,640))  # CPU için ctx_id=-1

    for p in tqdm(imgs, desc=f"Embeddings from {folder_in}"):
        img = cv2.imread(p)
        if img is None: 
            print("Okunamadı:", p); continue
        faces = app.get(img)
        if len(faces)==0:
            print("Yüz yok:", p); continue
        emb = faces[0].embedding  # (512,)
        outp = os.path.join(folder_out, os.path.basename(p) + ".npy")
        np.save(outp, emb.astype(np.float32))

if __name__ == "__main__":
    ensure_dirs()
    run("data/processed", "embeddings/processed")
    run("data/augmented", "embeddings/augmented")
    print("Embedding çıkarma tamam.")
