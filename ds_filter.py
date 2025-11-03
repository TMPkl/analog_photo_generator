import numpy as np
import cv2 as cv
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
import shutil

# ----------------------
# Twoje funkcje filtrów
# ----------------------
def detect_grayscale(image, tolerance=10, max_ratio=0.05):
    """
    Detect whether an image is grayscale (black-and-white).

    Parameters:
    - image: file path (str) or numpy array (H,W) or (H,W,3/4)
    - tolerance: per-pixel per-channel absolute difference allowed to still consider the pixel grayscale (0-255)
    - max_ratio: maximum fraction of pixels allowed to exceed `tolerance` and still be considered grayscale
    Returns: True if the image is effectively grayscale, False otherwise.
    """
    # If already single channel, it's grayscale
    img = cv.imread(image, cv.IMREAD_UNCHANGED) if isinstance(image, str) else image
    if img.ndim == 2:
        return True

    # If image has an alpha channel, ignore it
    if img.ndim == 3 and img.shape[2] >= 3:
        arr = img[:, :, :3].astype(np.int16)
        # split channels (OpenCV uses BGR)
        b = arr[:, :, 0]
        g = arr[:, :, 1]
        r = arr[:, :, 2]

        # compute maximum absolute difference between any two channels per pixel
        diff_rg = np.abs(r - g)
        diff_rb = np.abs(r - b)
        diff_gb = np.abs(g - b)
        max_diff = np.maximum(np.maximum(diff_rg, diff_rb), diff_gb)

        # count pixels that differ more than tolerance
        n_pixels = img.shape[0] * img.shape[1]
        n_bad = np.count_nonzero(max_diff > tolerance)
        ratio = n_bad / float(n_pixels)

        return ratio <= max_ratio

    # Unexpected shape
    return True

def overexposed_filter(image, threshold=0.04, pixel_value=252):
    count = np.count_nonzero(image > pixel_value)
    return (count / image.size) > threshold

def underexposed_filter(image, threshold=0.03, pixel_value=5):
    count = np.count_nonzero(image < pixel_value)
    return (count / image.size) > threshold

def binned_histogram_analysis(image, num_bins=10, threshold=3, tolerance=5):
    if tolerance>num_bins or tolerance<1:
        raise ValueError("tolerance must be between 1 and num_bins")

    bins = np.linspace(0, 256, num_bins + 1)
    hist_R, _ = np.histogram(image[:, :, 0], bins=bins)
    hist_G, _ = np.histogram(image[:, :, 1], bins=bins)
    hist_B, _ = np.histogram(image[:, :, 2], bins=bins)
    hists = np.vstack([hist_R, hist_G, hist_B])
    error_bin = 0

    for bin_idx in range(num_bins):
        ch_delta = (
            abs(hists[0, bin_idx] - hists[1, bin_idx]) +
            abs(hists[0, bin_idx] - hists[2, bin_idx]) +
            abs(hists[1, bin_idx] - hists[2, bin_idx])
        )
        if ch_delta > threshold * np.sum(hists[:, bin_idx])/3:
            error_bin += 1
            if error_bin >= tolerance:
                return False

    return error_bin < tolerance




# ----------------------
# Funkcja dla pojedynczego pliku <- No way Karol ...Polish comments ??!!!
# ----------------------
def process_single_image(args):
    file_path, good_folder, bad_folder, gs_folder = args
    img_g = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    img_bgr = cv.imread(file_path, cv.IMREAD_COLOR)
    if img_g is None or img_bgr is None:
        print("File could not be read:", file_path)
        return "skipped"

    real_flag = True
    if overexposed_filter(img_g) or overexposed_filter(img_g, pixel_value=220, threshold=0.25):
        real_flag = False
        print("Overexposed:", file_path)
    if underexposed_filter(img_g) or underexposed_filter(img_g, pixel_value=10, threshold=0.25):
        real_flag = False
        print("Underexposed:", file_path)
    if detect_grayscale(img_bgr, tolerance=10, max_ratio=0.2):
        print("Black and white:", file_path)
        cv.imwrite(os.path.join(gs_folder, os.path.basename(file_path)), img_bgr)
        return "gs"
    # if not binned_histogram_analysis(img_bgr):
    #     real_flag = False
    #     print("Color imbalance:", file_path)

    target_folder = good_folder if real_flag else bad_folder
    cv.imwrite(os.path.join(target_folder, os.path.basename(file_path)), img_bgr)
    return "good" if real_flag else "bad"

# ----------------------
# Funkcja przetwarzania równoległego
# ----------------------
def process_images_parallel(source_folder, good_folder, bad_folder, gs_folder, max_workers=2):
    # create folders if needed, and ensure they're empty
    for folder in (good_folder, bad_folder, gs_folder):
        os.makedirs(folder, exist_ok=True)
        for entry in os.listdir(folder):
            path = os.path.join(folder, entry)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(f"Could not remove {path}: {e}")

    files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.lower().endswith('.jpg')]
    files.sort()

    results = {"good": 0, "bad": 0, "skipped": 0, "gs": 0}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(process_single_image, [(f, good_folder, bad_folder, gs_folder) for f in files]):
            results[result] += 1

    print("-----Summary-----")
    print("Good images:", results["good"])
    print("Bad images:", results["bad"])
    print("Skipped images:", results["skipped"])
    print("Grayscale images:", results["gs"])

# ----------------------
# Wywołanie z terminala
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter images by exposure and color balance using multiple cores")
    parser.add_argument("source", help="Source folder with images")
    parser.add_argument("good", help="Destination folder for good images")
    parser.add_argument("bad", help="Destination folder for bad images")
    parser.add_argument("gs", help="Destination folder for grayscale images")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers")
    args = parser.parse_args()

    process_images_parallel(args.source, args.good, args.bad, args.gs, args.workers)
