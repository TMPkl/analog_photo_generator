"""Smooth a .cube LUT by applying a 3D Gaussian filter in Lab space.

Input: existing .cube file
Output: smoothed .cube file

This script:
- parses an input .cube
- converts RGB grid to Lab (OpenCV)
- applies 3D Gaussian smoothing per Lab channel (scipy.ndimage.gaussian_filter)
- converts smoothed Lab back to RGB and writes a new .cube

Usage:
  python3 -m src.LUT.smooth_lut --in input.cube --out smoothed.cube --sigma 1.0
"""

from pathlib import Path
import argparse
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def parse_cube(path: Path):
    size = None
    entries = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0].upper() == 'TITLE':
                continue
            if parts[0].upper() == 'LUT_3D_SIZE':
                size = int(parts[1])
                continue
            if parts[0].upper() in ('DOMAIN_MIN', 'DOMAIN_MAX'):
                continue
            if len(parts) >= 3:
                try:
                    rgb = [float(parts[0]), float(parts[1]), float(parts[2])]
                    entries.append(rgb)
                except ValueError:
                    continue

    if size is None:
        n = len(entries)
        cube_size = round(n ** (1 / 3))
        if cube_size ** 3 == n:
            size = cube_size
        else:
            raise ValueError('Could not determine LUT size.')

    arr = np.array(entries, dtype=np.float32)
    arr = arr.reshape((size, size, size, 3))
    # ensure in 0..1
    if arr.max() > 1.01:
        arr = np.clip(arr / 255.0, 0.0, 1.0)
    return arr


def write_cube(path: Path, arr_rgb: np.ndarray):
    size = arr_rgb.shape[0]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# Smoothed LUT\n')
        f.write('TITLE "Smoothed LUT"\n')
        f.write(f'LUT_3D_SIZE {size}\n')
        f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
        f.write('DOMAIN_MAX 1.0 1.0 1.0\n')
        # write entries B outer, G middle, R inner
        vals = arr_rgb.reshape((-1, 3))
        for rgb in vals:
            f.write('%.6f %.6f %.6f\n' % (float(rgb[0]), float(rgb[1]), float(rgb[2])))


def smooth_lut(in_cube: Path, out_cube: Path, sigma: float):
    arr_rgb = parse_cube(in_cube)  # shape (size,size,size,3) in 0..1
    size = arr_rgb.shape[0]

    # convert to BGR uint8 for Lab conversion
    bgr = (arr_rgb[..., ::-1] * 255.0).astype(np.uint8)
    bgr_flat = bgr.reshape((-1, 1, 3))
    lab_flat = cv2.cvtColor(bgr_flat, cv2.COLOR_BGR2LAB).reshape((size, size, size, 3)).astype(np.float32)

    # apply gaussian filter per Lab channel
    lab_smooth = np.empty_like(lab_flat, dtype=np.float32)
    for c in range(3):
        lab_smooth[..., c] = gaussian_filter(lab_flat[..., c], sigma=sigma, mode='mirror')

    # convert smoothed Lab back to RGB
    lab_uint8 = np.clip(lab_smooth, 0, 255).astype(np.uint8)
    lab_flat = lab_uint8.reshape((-1, 1, 3))
    bgr_smooth = cv2.cvtColor(lab_flat, cv2.COLOR_LAB2BGR).reshape((size, size, size, 3))
    rgb_smooth = bgr_smooth[..., ::-1].astype(np.float32) / 255.0

    write_cube(out_cube, rgb_smooth)


def main():
    parser = argparse.ArgumentParser(description='Smooth a .cube LUT using 3D Gaussian in Lab space')
    parser.add_argument('--in', dest='in_cube', required=True, help='Input .cube file')
    parser.add_argument('--out', dest='out_cube', required=True, help='Output .cube file')
    parser.add_argument('--sigma', type=float, default=1.0, help='Gaussian sigma (in grid units)')
    args = parser.parse_args()

    in_cube = Path(args.in_cube)
    out_cube = Path(args.out_cube)
    if not in_cube.exists():
        print('Input .cube not found:', in_cube)
        return
    smooth_lut(in_cube, out_cube, args.sigma)
    print('Wrote smoothed LUT to', out_cube)


if __name__ == '__main__':
    main()
