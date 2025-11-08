import os, math, cv2
import mediapipe as mp
from utils import ensure_dirs, list_images, save_image

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [33, 133, 160, 159, 158, 157]
RIGHT_EYE = [263, 362, 387, 386, 385, 384]

def center_of(lms, w, h, idxs):
    xs, ys = [], []
    for i in idxs:
        xs.append(int(lms[i].x * w))
        ys.append(int(lms[i].y * h))
    return (sum(xs)//len(xs), sum(ys)//len(ys))

def align_one(path_in, path_out, out_size=(112,112)):
    img = cv2.imread(path_in)
    if img is None:
        print("Okunamadı:", path_in); return False
    h, w = img.shape[:2]

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            print("Yüz bulunamadı:", path_in); return False
        lm = res.multi_face_landmarks[0].landmark

        L = center_of(lm, w, h, LEFT_EYE)
        R = center_of(lm, w, h, RIGHT_EYE)
        dy, dx = R[1]-L[1], R[0]-L[0]
        angle = math.degrees(math.atan2(dy, dx))

        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

        # merkezden güvenli crop
        y0, y1 = h//6, h - h//6
        x0, x1 = w//6, w - w//6
        crop = rotated[y0:y1, x0:x1]
        if crop.size == 0:
            print("Boş crop:", path_in); return False
        face = cv2.resize(crop, out_size)
        save_image(path_out, face)
        return True

if __name__ == "__main__":
    ensure_dirs()
    imgs = list_images("data/raw")
    os.makedirs("data/processed", exist_ok=True)
    ok, fail = 0, 0
    for p in imgs:
        outp = os.path.join("data/processed", os.path.basename(p))
        ok |= align_one(p, outp)
        if not ok: fail += 1
    print("Bitti. İşlenen:", len(imgs)-fail, "Hatalı:", fail)
