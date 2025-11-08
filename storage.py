import os, numpy as np
from cryptography.fernet import Fernet

KEY_PATH = "keys/fernet.key"

def gen_key():
    os.makedirs(os.path.dirname(KEY_PATH), exist_ok=True)
    if os.path.exists(KEY_PATH):
        print("Anahtar zaten var:", KEY_PATH); return
    key = Fernet.generate_key()
    with open(KEY_PATH, "wb") as f:
        f.write(key)
    print("Anahtar üretildi:", KEY_PATH)

def _load_key():
    if not os.path.exists(KEY_PATH):
        raise FileNotFoundError("Anahtar yok. Önce gen_key() çalıştır.")
    with open(KEY_PATH, "rb") as f:
        return f.read()

def encrypt_embedding(ndarray: np.ndarray, out_path: str):
    key = _load_key()
    f = Fernet(key)
    token = f.encrypt(ndarray.tobytes())
    with open(out_path, "wb") as fp:
        fp.write(token)
    print("Şifreli embedding yazıldı:", out_path)

def decrypt_embedding(in_path: str, shape, dtype=np.float32) -> np.ndarray:
    key = _load_key()
    f = Fernet(key)
    with open(in_path, "rb") as fp:
        token = fp.read()
    raw = f.decrypt(token)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)
