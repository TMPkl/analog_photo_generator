"""Color transfer using 3D color clusters + Optimal Transport.

Approach:
- Sample pixels from source and target (optionally from folders).
- Cluster colors with k-means (OpenCV) into K centroids for each set.
- Compute Optimal Transport plan between source and target centroids (POT library).
- Map source centroids to barycentric target colors using the transport plan.
- Replace pixels in source images by mapping each pixel to its nearest source centroid
  and substituting the barycentric-mapped color.

This is memory-friendly because OT is solved on the reduced atom set (K clusters)
instead of full per-pixel histograms.

Usage examples:
  # Single source image, single target image
  python3 -m src.LUT.ot_color_transfer --source src.jpg --target ref.jpg --out out.jpg --k 256

  # Folder to folder
  python3 -m src.LUT.ot_color_transfer --source folderA --target folderB --out out_folder --k 256 --samples 200000
"""

from pathlib import Path
import argparse
import os
import cv2
import numpy as np
import ot
import math
import random
from tqdm import tqdm


def sample_pixels_from_path(path: Path, max_samples=100000):
    """Read image(s) from `path` (file or folder) and return sampled Lab pixels (uint8 OpenCV LAB space).

    Returns numpy array shape (N,3) dtype=float32
    """
    pixels = []
    if path.is_dir():
        imgs = [p for p in sorted(path.iterdir()) if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')]
    else:
        imgs = [path]

    if not imgs:
        return np.zeros((0, 3), dtype=np.float32)

    per_image = max(1, max_samples // len(imgs))
    for p in tqdm(imgs, desc='Sampling images', unit='img'):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        h, w, _ = lab.shape
        flat = lab.reshape((-1, 3))
        if flat.shape[0] <= per_image:
            pixels.append(flat)
        else:
            idx = np.random.choice(flat.shape[0], per_image, replace=False)
            pixels.append(flat[idx])

    if not pixels:
        return np.zeros((0, 3), dtype=np.float32)

    allpix = np.vstack(pixels).astype(np.float32)
    # shuffle and limit exactly to max_samples
    if allpix.shape[0] > max_samples:
        idx = np.random.choice(allpix.shape[0], max_samples, replace=False)
        allpix = allpix[idx]
    return allpix


def compute_kmeans(pixels: np.ndarray, k=256, attempts=8):
    """Compute k-means using OpenCV; return centers (k,3) and weights (k,)

    pixels: float32 in OpenCV LAB scale (0..255)
    """
    if pixels.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float64)

    # OpenCV kmeans expects float32 samples
    samples = pixels.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(samples, k, None, criteria, attempts, flags)
    labels = labels.flatten()
    centers = centers.astype(np.float64)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    weights = counts / counts.sum()
    return centers, weights, labels


def compute_transport_and_map(centers_src, a, centers_tgt, b):
    """Compute OT plan between (centers_src, a) and (centers_tgt, b).

    Returns mapped_centers_src (k_src, 3) in same color space as centers_tgt.
    """
    if centers_src.shape[0] == 0 or centers_tgt.shape[0] == 0:
        return centers_src

    # cost matrix: Euclidean distances between centroids
    C = ot.dist(centers_src, centers_tgt, metric='euclidean')
    # Normalize costs a bit to avoid very large numbers
    C = C / C.max()

    # Solve EMD; returns transport matrix T of shape (k_src, k_tgt)
    T = ot.emd(a, b, C)

    # compute barycentric mapping: mapped_i = sum_j T_ij * centers_tgt_j / row_sum_i
    row_sums = T.sum(axis=1)
    # avoid division by zero
    mapped = np.zeros_like(centers_src, dtype=np.float64)
    for i in range(centers_src.shape[0]):
        r = row_sums[i]
        if r > 0:
            mapped[i] = (T[i:i+1] @ centers_tgt) / r
        else:
            mapped[i] = centers_src[i]
    return mapped


def map_image_using_centers(img_path: Path, centers_src, mapped_centers, out_path: Path, chunk_pixels=200000):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print('Failed to read', img_path)
        return
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    h, w, _ = lab.shape
    flat = lab.reshape(-1, 3).astype(np.float64)

    k = centers_src.shape[0]
    if k == 0:
        print('No centers; copying file')
        cv2.imwrite(str(out_path), img)
        return

    # For each pixel find nearest source center index (vectorized in chunks)
    indices = np.empty((flat.shape[0],), dtype=np.int32)
    bs = chunk_pixels
    total = flat.shape[0]
    blocks = math.ceil(total / bs)
    # iterate in chunks and show progress
    start = 0
    for bi in tqdm(range(blocks), desc='Mapping pixels', unit='chunk'):
        i = bi * bs
        block = flat[i:i+bs]
        # distances: (B, k)
        d = np.linalg.norm(block[:, None, :] - centers_src[None, :, :], axis=2)
        indices[i:i+bs] = np.argmin(d, axis=1)

    # map pixels
    mapped_flat = mapped_centers[indices]
    mapped_img = mapped_flat.reshape((h, w, 3)).astype(np.uint8)
    bgr = cv2.cvtColor(mapped_img, cv2.COLOR_LAB2BGR)
    cv2.imwrite(str(out_path), bgr)
    print('Wrote', out_path)


def main():
    parser = argparse.ArgumentParser(description='Color transfer using 3D clusters + Optimal Transport')
    parser.add_argument('--source', required=True, help='Source image or folder (to be recolored)')
    parser.add_argument('--target', required=True, help='Target image or folder (reference colors)')
    parser.add_argument('--out', required=True, help='Output file or folder')
    parser.add_argument('--k', type=int, default=256, help='Number of color clusters per image set')
    parser.add_argument('--samples', type=int, default=200000, help='Max sampled pixels per dataset (source/target)')
    parser.add_argument('--attempts', type=int, default=8, help='kmeans attempts')
    args = parser.parse_args()

    src_path = Path(args.source)
    tgt_path = Path(args.target)
    out_path = Path(args.out)

    print('Sampling pixels...')
    src_pixels = sample_pixels_from_path(src_path, max_samples=args.samples)
    tgt_pixels = sample_pixels_from_path(tgt_path, max_samples=args.samples)

    if src_pixels.shape[0] == 0:
        print('No source pixels found. Exiting.')
        return
    if tgt_pixels.shape[0] == 0:
        print('No target pixels found. Exiting.')
        return

    print('Computing k-means for source (k=%d)...' % args.k)
    centers_src, a, _ = compute_kmeans(src_pixels, k=args.k, attempts=args.attempts)
    print('Computing k-means for target (k=%d)...' % args.k)
    centers_tgt, b, _ = compute_kmeans(tgt_pixels, k=args.k, attempts=args.attempts)

    print('Solving Optimal Transport... (this may take a while for large k)')
    mapped_centers = compute_transport_and_map(centers_src, a, centers_tgt, b)

    # Prepare outputs
    if src_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)
        imgs = [p for p in sorted(src_path.iterdir()) if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')]
        if not imgs:
            print('No images found in source folder')
            return
        for p in tqdm(imgs, desc='Processing images', unit='img'):
            dst = out_path / p.name
            map_image_using_centers(p, centers_src, mapped_centers, dst)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        map_image_using_centers(src_path, centers_src, mapped_centers, out_path)


if __name__ == '__main__':
    main()
