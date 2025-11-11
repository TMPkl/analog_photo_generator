import numpy as np

import numpy as np

def BGR2XYZ(bgr, matrix=None):
    """
    Konwersja obrazu BGR (OpenCV) do przestrzeni XYZ.
    Obsługuje zarówno pojedynczy piksel [B, G, R],
    jak i cały obraz 3-kanałowy np.ndarray o kształcie (h, w, 3).
    """
    bgr = np.array(bgr, dtype=np.float32)
    if bgr.max() > 1.0:
        bgr /= 255.0

    if matrix is None:
        matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=np.float32)

    if bgr.ndim == 3 and bgr.shape[2] == 3:
        rgb = bgr[..., ::-1]
        rgb_flat = rgb.reshape(-1, 3)
        xyz_flat = np.dot(rgb_flat, matrix.T)
        xyz = xyz_flat.reshape(bgr.shape)
    else:
        rgb = np.array([bgr[2], bgr[1], bgr[0]], dtype=np.float32)
        xyz = matrix.dot(rgb)

    return xyz



def XYZ2BGR(xyz, matrix=None):
    """
    Konwersja obrazu XYZ do przestrzeni BGR (OpenCV).
    Obsługuje zarówno pojedynczy piksel [X, Y, Z],
    jak i cały obraz 3-kanałowy np.ndarray o kształcie (h, w, 3).
    """
    xyz = np.array(xyz, dtype=np.float32)

    if matrix is None:
        matrix = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ], dtype=np.float32)

    if xyz.ndim == 3 and xyz.shape[2] == 3:
        xyz_flat = xyz.reshape(-1, 3)
        rgb_flat = np.dot(xyz_flat, matrix.T)
        rgb = rgb_flat.reshape(xyz.shape)
        bgr = rgb[..., ::-1]
    else:
        rgb = matrix.dot(xyz)
        bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)

    bgr = np.clip(bgr, 0.0, 1.0)
    bgr = (bgr * 255).astype(np.uint8)
    return bgr
