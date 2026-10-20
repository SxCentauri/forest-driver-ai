import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from data import make_dataset, make_patch, CLASSES
from features import handcrafted

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def plot_samples():
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(2, 5, figsize=(12, 5))
    for c in range(5):
        p = make_patch(c, rng)
        ax[0, c].imshow(p[0], cmap="RdYlGn", vmin=-.7, vmax=.2); ax[0, c].set_title(CLASSES[c], fontsize=9)
        ax[1, c].imshow(p[2], cmap="magma", vmin=0, vmax=.8)
        ax[0, c].axis("off"); ax[1, c].axis("off")
    fig.text(0.01, 0.72, "Optik (ΔNDVI)", rotation=90, va="center")
    fig.text(0.01, 0.28, "Radar (ΔVV)", rotation=90, va="center")
    plt.tight_layout(rect=(0.03, 0, 1, 1)); plt.savefig(f"{OUT}/contoh_patch.png", dpi=130); plt.close()


def plot_cm(cm, title, path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Greens")
    plt.xticks(range(5), CLASSES, rotation=40, ha="right"); plt.yticks(range(5), CLASSES)
    for i in range(5):
        for j in range(5):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
    plt.title(title); plt.xlabel("Prediksi"); plt.ylabel("Aktual")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="sampel per kelas")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--skip-cnn", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    X, y = make_dataset(a.n)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    plot_samples()
    lines = [f"Data: {len(y)} patch, train={len(ytr)}, test={len(yte)}"]

    print("[1] Baseline: Random Forest + fitur manual")
    rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(handcrafted(Xtr), ytr)
    p = rf.predict(handcrafted(Xte))
    acc = accuracy_score(yte, p)
    print(classification_report(yte, p, target_names=CLASSES))
    plot_cm(confusion_matrix(yte, p), f"Random Forest (acc={acc:.2f})", f"{OUT}/cm_rf.png")
    lines.append(f"Random Forest accuracy: {acc:.3f}")

    if not a.skip_cnn:
        print("[2] CNN (PyTorch)")
        from cnn import train_cnn
        p = train_cnn(Xtr, ytr, Xte, epochs=a.epochs)
        acc = accuracy_score(yte, p)
        print(classification_report(yte, p, target_names=CLASSES))
        plot_cm(confusion_matrix(yte, p), f"CNN (acc={acc:.2f})", f"{OUT}/cm_cnn.png")
        lines.append(f"CNN accuracy: {acc:.3f}")

    open(f"{OUT}/ringkasan.txt", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
