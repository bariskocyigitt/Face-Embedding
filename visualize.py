import os, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_folder(folder):
    X, names = [], []
    if not os.path.isdir(folder): return np.zeros((0,)), []
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".npy"):
            arr = np.load(os.path.join(folder, fn))
            X.append(arr); names.append(fn.replace(".jpg.npy",""))
    return (np.vstack(X) if X else np.zeros((0,)), names)

def scatter_and_save(pts, title, out_path, labels=None):
    plt.figure(figsize=(8,6))
    plt.scatter(pts[:,0], pts[:,1], s=40)
    if labels:
        for i, t in enumerate(labels):
            plt.annotate(t, (pts[i,0], pts[i,1]), fontsize=7)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()

if __name__ == "__main__":
    X1, n1 = load_folder("embeddings/processed")
    X2, n2 = load_folder("embeddings/augmented")
    if X1.size == 0 and X2.size == 0:
        print("Embedding bulunamadı. Önce extract_embeddings.py çalıştır.")
        raise SystemExit

    X = X1
    labels = n1
    if X2.size > 0:
        X = np.vstack([X1, X2]) if X1.size > 0 else X2
        labels = (n1 + n2) if X1.size > 0 else n2

    # PCA
    if X.shape[0] >= 2:
        pca = PCA(n_components=2)
        p2 = pca.fit_transform(X)
        scatter_and_save(p2, "PCA projection of face embeddings", "data/outputs/pca_plot.png", labels)
        print("Kaydedildi: data/outputs/pca_plot.png")
    else:
        print("PCA için en az 2 örnek gerekli.")

    # t-SNE (opsiyonel; daha ağır)
    if X.shape[0] >= 3:
        tsne = TSNE(n_components=2, perplexity=min(5, max(2, X.shape[0]-1)), random_state=42, init="random", learning_rate="auto")
        t2 = tsne.fit_transform(X)
        scatter_and_save(t2, "t-SNE projection of face embeddings", "data/outputs/tsne_plot.png", labels)
        print("Kaydedildi: data/outputs/tsne_plot.png")
    else:
        print("t-SNE için en az 3 örnek gerekli.")
