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
    # Generate a .cube mapping from folderA (A images) to folderB (B images)
    python3 -m src.LUT.ot_color_transfer --source folderA --target folderB --k 256 --cube-out my_lut.cube

Notes:
- This script ONLY accepts folders for `--source` and `--target` and ALWAYS writes a .cube file.
- It does NOT modify or write any images.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
try:
    from sklearn.cluster import MiniBatchKMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


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
        img = None
        try:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            # try Pillow fallback for problematic TIFFs or uncommon formats
            try:
                pil = Image.open(p)
                pil = pil.convert('RGB')
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception:
                # could not read image; skip and continue
                # print a short warning to stderr
                print(f'Warning: failed to read {p}; skipping')
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

    # If dataset is large and sklearn available, use MiniBatchKMeans for speed and progress
    if SKLEARN_AVAILABLE and samples.shape[0] > 50000:
        mbk = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=42)
        # iterate over shuffled chunks and partial_fit with progress
        n = samples.shape[0]
        batch = 10000
        indices = np.arange(n)
        np.random.shuffle(indices)
        labels = np.empty((n,), dtype=np.int32)
        for i in tqdm(range(0, n, batch), desc='MiniBatchKMeans', unit='batch'):
            idx = indices[i:i+batch]
            mbk.partial_fit(samples[idx])
        centers = mbk.cluster_centers_.astype(np.float64)
        # assign labels to compute weights
        labels = mbk.predict(samples)
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        weights = counts / counts.sum()
        return centers, weights, labels

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


def write_cube_file(out_cube: str, centers_src: np.ndarray, mapped_centers: np.ndarray, size: int = 17):
    """Generate a .cube LUT file of given `size` that maps input RGB -> mapped RGB.

    Mapping strategy: for each grid point (r,g,b) in [0,1], convert to Lab, find nearest
    source center in Lab space and use its mapped_centers value (Lab) converted back to RGB.
    """
    out_cube = Path(out_cube)
    # build grid of RGB values ordered R fastest, then G, then B (so we will iterate B,G,R)
    vals = np.linspace(0.0, 1.0, size, dtype=np.float32)
    coords = []
    for bb in vals:
        for gg in vals:
            for rr in vals:
                coords.append([rr, gg, bb])
    coords = np.array(coords, dtype=np.float32)  # (N,3) in RGB

    # convert to BGR uint8 for OpenCV conversion to Lab
    bgr = (coords[:, ::-1] * 255.0).astype(np.uint8)  # rgb -> bgr
    bgr = bgr.reshape((-1, 1, 3))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape((-1, 3)).astype(np.float64)

    # compute distances (N, k) in Lab space and use nearest-centroid mapping (hard assignment)
    d = np.linalg.norm(lab[:, None, :] - centers_src[None, :, :], axis=2)
    idx = np.argmin(d, axis=1)
    mapped_lab = mapped_centers[idx]
    mapped_lab_uint8 = np.clip(mapped_lab, 0, 255).astype(np.uint8)
    mapped_lab_uint8 = mapped_lab_uint8.reshape((-1, 1, 3))
    mapped_bgr = cv2.cvtColor(mapped_lab_uint8, cv2.COLOR_LAB2BGR).reshape((-1, 3))
    mapped_rgb = mapped_bgr[:, ::-1].astype(np.float32) / 255.0

    # write .cube file with header and entries (R G B per line), ordering: B,G,R loops
    with open(out_cube, 'w', encoding='utf-8') as f:
        f.write('# Generated by ot_color_transfer\n')
        f.write('TITLE "OT color transfer LUT"\n')
        f.write('LUT_3D_SIZE %d\n' % (size))
        f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
        f.write('DOMAIN_MAX 1.0 1.0 1.0\n')
        # write entries in same order as coords (B outer, G middle, R inner)
        for rgb in mapped_rgb:
            f.write('%.6f %.6f %.6f\n' % (float(rgb[0]), float(rgb[1]), float(rgb[2])))



def main():
    parser = argparse.ArgumentParser(description='Color transfer using 3D clusters + Optimal Transport')
    parser.add_argument('--source', required=True, help='Source folder (A images)')
    parser.add_argument('--target', required=True, help='Target folder (B images)')
    parser.add_argument('--k', type=int, default=256, help='Number of color clusters per dataset')
    parser.add_argument('--samples', type=int, default=200000, help='Max sampled pixels per dataset (source/target)')
    parser.add_argument('--attempts', type=int, default=8, help='kmeans attempts')
    parser.add_argument('--cube-out', type=str, required=True, help='Path to output .cube LUT (required)')
    parser.add_argument('--cube-size', type=int, default=17, help='Grid size for output .cube (e.g. 17, 33)')
    args = parser.parse_args()

    src_path = Path(args.source)
    tgt_path = Path(args.target)
    cube_out = args.cube_out

    print('Sampling pixels from A and B in parallel...')
    # run sampling for source and target in parallel to overlap I/O and CPU
    with ThreadPoolExecutor(max_workers=2) as exe:
        fut_src = exe.submit(sample_pixels_from_path, src_path, args.samples)
        fut_tgt = exe.submit(sample_pixels_from_path, tgt_path, args.samples)
        # show small progress bar while both tasks complete
        for _ in tqdm(as_completed([fut_src, fut_tgt]), total=2, desc='Sampling sets', unit='set'):
            pass
        src_pixels = fut_src.result()
        tgt_pixels = fut_tgt.result()

    if src_pixels.shape[0] == 0:
        print('No source pixels found in folder. Exiting.')
        return
    if tgt_pixels.shape[0] == 0:
        print('No target pixels found in folder. Exiting.')
        return

    print('Computing k-means for source (k=%d)...' % args.k)
    centers_src, a, _ = compute_kmeans(src_pixels, k=args.k, attempts=args.attempts)
    print('Computing k-means for target (k=%d)...' % args.k)
    centers_tgt, b, _ = compute_kmeans(tgt_pixels, k=args.k, attempts=args.attempts)

    print('Solving Optimal Transport... (this may take a while for large k)')
    mapped_centers = compute_transport_and_map(centers_src, a, centers_tgt, b)

    # Write .cube LUT and exit (script only generates LUT from folders)
    cube_size = args.cube_size
    print(f'Generating .cube file with size {cube_size}...')
    write_cube_file(cube_out, centers_src, mapped_centers, cube_size)
    print('Wrote .cube to', cube_out)
    print('Done.')


if __name__ == '__main__':
    main()

