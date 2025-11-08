import cv2
import os
import sys
from utils import ensure_dirs, save_image, timestamp

def capture_from_webcam(
    n=8,
    wait_ms=800,
    device_index=0,
    show_preview=True,
    width=None,
    height=None,
    warmup_frames=5
):
    """
    n: Kaç kare kaydedilecek.
    wait_ms: Kareler arasında bekleme (ms).
    device_index: Kamera index'i (0/1/2...).
    show_preview: True ise imshow ile canlı önizleme.
    width, height: Çözünürlük hedefi (örn. 640x480). None ise varsayılan.
    warmup_frames: Otomatik odak/poz ölçümü için ilk kaç kareyi atlayalım.
    """
    ensure_dirs()

    # Windows'ta bazı cihazlarda CAP_DSHOW daha stabil olabilir:
    # cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Webcam açılamadı (index={device_index}). "
                           f"Başka bir uygulama kamerayı kullanıyor olabilir.")

    try:
        # Çözünürlük hedefi (opsiyonel)
        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Isınma kareleri (otofokus/white balance otursun)
        for _ in range(max(0, warmup_frames)):
            ok, _ = cap.read()
            if not ok:
                raise RuntimeError("Kameradan görüntü okunamıyor (ısınma aşaması).")

        if show_preview:
            print("Pencereye odaklan: 'q' ile çık. Otomatik kayıt yapılıyor...")
        else:
            print("Önizlemesiz kayıt başlıyor...")

        saved = 0
        while saved < n:
            ok, frame = cap.read()
            if not ok:
                print("Uyarı: Kare okunamadı, döngü sonlandırılıyor.")
                break

            fname = os.path.join("data", "raw", f"{timestamp()}_{saved}.jpg")
            save_image(fname, frame)
            print("Kaydedildi:", fname)

            if show_preview:
                cv2.imshow("Capture (q=çık)", frame)
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord('q'):
                    print("Kullanıcı çıkış istedi (q).")
                    break
            else:
                # Önizleme yoksa yine de tempo için bekle:
                # Not: waitKey yokken time.sleep kullanılabilir
                # ama burada basitçe atlıyoruz ya da kısa bir bekleme ekleyebilirsin.
                pass

            saved += 1

    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # Örnek: 8 kare, 800 ms, 640x480 çözünürlük, 3 ısınma karesi
    capture_from_webcam(n=8, wait_ms=800, width=640, height=480, warmup_frames=3)
