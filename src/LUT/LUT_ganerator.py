# save_lut_from_unpaired.py
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def load_images_from_folder(folder, ext=("jpg","jpeg","png","TIF","TIFF","tif","JPG","PNG","JPEG")):
    imgs = []
    for e in ext:
        for p in glob(os.path.join(folder, f"**/*.{e}"), recursive=True):
            imgs.append(p)
    return imgs

def compute_lab_mean_std(image_paths, max_samples=200000):
    # sample pixels from images to compute dataset mean/std in Lab
    samples = []
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        h,w,_ = img.shape
        # randomly sample up to some pixels per image
        k = min(h*w, max(5000, max_samples // max(1, len(image_paths))))
        coords = np.random.choice(h*w, k, replace=False)
        pixels = img.reshape(-1,3)[coords]
        samples.append(pixels)
        if sum(s.shape[0] for s in samples) >= max_samples:
            break
    if len(samples) == 0:
        raise ValueError("No images found or readable.")
    allpix = np.vstack(samples).astype(np.float32)
    mean = allpix.mean(axis=0)      # [L_mean, a_mean, b_mean]
    std  = allpix.std(axis=0)       # [L_std, a_std, b_std]
    return mean, std

def reinhard_transfer_rgb(rgb_img, src_mean, src_std, tgt_mean, tgt_std):
    # rgb_img: uint8 BGR image (OpenCV)
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    # apply Reinhard per-channel
    lab_t = (lab - src_mean) / (src_std + 1e-8) * (tgt_std + 1e-8) + tgt_mean
    lab_t = np.clip(lab_t, 0, 255).astype(np.uint8)
    bgr_t = cv2.cvtColor(lab_t, cv2.COLOR_LAB2BGR)
    return bgr_t

def make_3d_lut(grid_size, src_mean, src_std, tgt_mean, tgt_std):
    # grid_size: e.g., 33
    # produce LUT array shape (grid_size, grid_size, grid_size, 3) in 0..1 float
    # iterate over RGB grid in 0..255
    lut = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float32)
    vals = np.linspace(0, 255, grid_size)
    for r_i, r in enumerate(vals):
        for g_i, g in enumerate(vals):
            for b_i, b in enumerate(vals):
                # create single-pixel BGR image
                px = np.array([[[b, g, r]]], dtype=np.uint8)  # OpenCV BGR order
                out = reinhard_transfer_rgb(px, src_mean, src_std, tgt_mean, tgt_std)
                # store normalized 0..1 in RGB order
                bgr = out[0,0].astype(np.float32) / 255.0
                lut[r_i, g_i, b_i] = bgr[::-1]  # convert to RGB order
    return lut

def write_cube(lut, filename, title="Generated LUT", domain_min=0.0, domain_max=1.0):
    # lut: shape (N,N,N,3) in RGB 0..1
    N = lut.shape[0]
    with open(filename, "w") as f:
        f.write(f"# Created by script\nTITLE \"{title}\"\nLUT_3D_SIZE {N}\nDOMAIN_MIN {domain_min} {domain_min} {domain_min}\nDOMAIN_MAX {domain_max} {domain_max} {domain_max}\n")
        # .cube expects ordering: r fastest? Common convention is R major then G then B
        # We'll write in R G B nested loops matching typical players (many accept this).
        for r in range(N):
            for g in range(N):
                for b in range(N):
                    rgb = lut[r, g, b]
                    f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")
    print(f"Wrote {filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folderA", required=True, help="ścieżka do zbioru A (wejście)")
    parser.add_argument("--folderB", required=True, help="ścieżka do zbioru B (docelowy wygląd)")
    parser.add_argument("--out", default="out_lut.cube")
    parser.add_argument("--size", type=int, default=33, help="wymiar LUT (np. 17,33,65). 33 -> 35937 entries")
    args = parser.parse_args()
    print(args.folderA)

    imgsA = load_images_from_folder(args.folderA)
    imgsB = load_images_from_folder(args.folderB)
    print(f"Found {len(imgsA)} images in A, {len(imgsB)} in B")

    src_mean, src_std = compute_lab_mean_std(imgsA)
    tgt_mean, tgt_std = compute_lab_mean_std(imgsB)
    print("src mean/std:", src_mean, src_std)
    print("tgt mean/std:", tgt_mean, tgt_std)

    lut = make_3d_lut(args.size, src_mean, src_std, tgt_mean, tgt_std)
    write_cube(lut, args.out, title=f"Reinhard_from_{os.path.basename(args.folderA)}_to_{os.path.basename(args.folderB)}")
