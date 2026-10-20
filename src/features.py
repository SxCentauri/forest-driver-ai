import numpy as np


def handcrafted(X):
    f = []
    for p in X:
        m = p[0] < -0.2                    # piksel yang "hilang"
        area = m.sum()
        rows, cols = np.where(m) if area else ([0], [0])
        h, w = np.ptp(rows) + 1, np.ptp(cols) + 1
        f.append([
            area, area / (h * w), h, w,
            *p.mean((1, 2)), *p.std((1, 2)),
            p[2][m].mean() if area else 0, p[3][m].mean() if area else 0,
            p[1][m].mean() if area else 0,
        ])
    return np.array(f, dtype=np.float32)
