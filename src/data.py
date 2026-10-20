import numpy as np

CLASSES = ["Smallholder", "Agroindustri", "Penebangan selektif", "Tambang", "Kebakaran"]
S = 16


def _mask(cls, rng):
    m = np.zeros((S, S), dtype=bool)
    if cls == 0:  # beberapa bercak kecil tak beraturan
        for _ in range(rng.integers(2, 5)):
            x, y = rng.integers(2, S - 2, 2)
            m[y - 1:y + 2, x - 1:x + 2] = True
    elif cls == 1:  # blok besar persegi panjang
        w, h = rng.integers(8, 13, 2)
        x, y = rng.integers(0, S - w + 1), rng.integers(0, S - h + 1)
        m[y:y + h, x:x + w] = True
    elif cls == 2:  # jalur tipis + bercak kecil (jalan log)
        y = rng.integers(3, S - 3)
        m[y:y + 2, :] = True
        for _ in range(3):
            x, yy = rng.integers(1, S - 1, 2)
            m[yy, x] = True
    elif cls == 3:  # area sedang, kompak
        cx, cy = rng.integers(5, S - 5, 2)
        yy, xx = np.ogrid[:S, :S]
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= rng.integers(12, 25)
    else:  # kebakaran: bercak luas bergerigi
        base = rng.random((S, S))
        yy, xx = np.ogrid[:S, :S]
        cx, cy = rng.integers(4, S - 4, 2)
        m = ((xx - cx) ** 2 + (yy - cy) ** 2 < 55) & (base > 0.25)
    return m


def make_patch(cls, rng):
    m = _mask(cls, rng).astype(np.float32)
    x = rng.normal(0, 0.05, (4, S, S)).astype(np.float32)  # noise dasar
    # Efek optik (NDVI turun). Kebakaran: NIR turun lebih tajam.
    x[0] -= m * rng.uniform(0.35, 0.6)
    x[1] -= m * (rng.uniform(0.4, 0.7) if cls == 4 else rng.uniform(0.15, 0.35))
    # Efek radar (stabil terhadap awan). Tambang: perubahan VV kuat; kebakaran: lemah.
    vv = {0: .25, 1: .3, 2: .2, 3: .6, 4: .08}[cls]
    x[2] += m * rng.normal(vv, 0.05)
    x[3] += m * rng.normal(vv * 0.7, 0.05)
    # Gangguan awan pada kanal optik (radar tidak terpengaruh)
    if rng.random() < 0.3:
        cm = rng.random((S, S)) > 0.8
        x[0][cm] += rng.normal(0, 0.3)
        x[1][cm] += rng.normal(0, 0.3)
    return x


def make_dataset(n_per_class=400, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(len(CLASSES)):
        for _ in range(n_per_class):
            X.append(make_patch(c, rng)); y.append(c)
    X, y = np.stack(X), np.array(y)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]
