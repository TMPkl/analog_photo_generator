import numpy as np

def BGR2XYZ(bgr, matrix=None):
    """
    From BGR cv2 like format to XYZ color space wich is representing film negative leyers (C41 like).
    input: bgr - list or np.array with 3 elements (B, G, R) in range [0, 255]
    output: np.array with 3 elements (X, Y, Z)
    """
    bgr = np.array(bgr, dtype=np.float32)
    if bgr.max() > 1.0:
        bgr /= 255.0

    if matrix is None:
        matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])


    rgb = np.array([bgr[2], bgr[1], bgr[0]])
    xyz = matrix.dot(rgb)
    return xyz

def XYZ2BGR(xyz, matrix=None):
    import cv2 as cv
    """
    From XYZ color space to BGR cv2 like format.
    input: xyz - list or np.array with 3 elements (X, Y, Z)
    output: np.array with 3 elements (B, G, R) in range [0, 255]
    """
    xyz = np.array(xyz, dtype=np.float32)

    if matrix is None:
        matrix = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ])

    rgb = matrix.dot(xyz)
    bgr = np.array([rgb[2], rgb[1], rgb[0]])
    bgr = np.clip(bgr, 0.0, 1.0)
    bgr = (bgr * 255).astype(np.uint8)
    return bgr
