
# Face Embedding Lab 🧠

Bu proje, yüz görüntülerinden **embedding (vektörel yüz temsili)** çıkarma ve bu embedding’lerin farklı koşullar altında **nasıl değiştiğini analiz etme** amaçlıdır.

Proje **yüz tanıma (kimlik doğrulama)** yapmaz.  
Sadece **çıkarım + analiz + görselleştirme** içerir.  
Tüm veriler **lokalde** saklanır.

---

## 📌 Ne Yapıyoruz?

1. Webcam veya dosya ile yüz görüntüsü alıyoruz.
2. Yüzü algılayıp **hizalıyoruz** (göz çizgisine göre döndürme).
3. Görüntüye farklı ışık, açı ve gürültü etkileri ekleyerek **augmentasyon** uyguluyoruz.
4. Önceden eğitilmiş bir model (ArcFace / InsightFace) ile **embedding (512-dim vektör)** çıkarıyoruz.
5. Bu embedding’leri **PCA veya t-SNE** ile **2 boyutlu grafiğe** dökerek inceliyoruz.

---

## 🔍 Neden Yapıyoruz?

Amaç, şu soruları **bilimsel olarak anlamaktır**:

- Aynı yüz farklı ışıkta nasıl temsil edilir?
- Açısal değişim embedding’leri ne kadar uzaklaştırır?
- Augmentasyonun etkisi nedir?
- Model neye duyarlı, neye dayanıklı?

Bu, **yüz tanıma sistemlerinin mantığını anlamanın temelidir**.

---

## 💡 Etik ve Yasal Not

Bu proje:
- Kimlik eşleştirme yapmaz.
- "Bu kişi X mi?" şeklinde **karar üretmez**.
- Sadece **analiz** yapar.

Kullanım kuralları:
- Sadece **kendi yüzünüzü** veya **izin aldığınız kişilerin** yüzünü kullanın.
- Ham fotoğraflar ve embedding dosyalarını **GitHub'a yüklemeyin**.
- Veri **lokalde** kalmalıdır.

---

## 🚀 Çalıştırma Sırası

```bash
python capture.py         # Görüntü al
python align.py           # Yüz hizala
python augment.py         # Veri çoğalt
python extract_embeddings.py   # Embedding çıkar
python visualize.py       # PCA / t-SNE grafiklerini oluştur

