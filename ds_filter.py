import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os 
 
def overexposed_filter(image, threshold=0.04, pixel_value=253):
    """
    Check if the image is overexposed.
    
    Parameters:
    image (numpy.ndarray): Grayscale image to be checked.
    threshold (float): Proportion of pixels that must exceed pixel_value to consider the image overexposed.
    pixel_value (int): Pixel intensity value to check against (0-255).
    
    Returns:
    bool: True if the image is overexposed, False otherwise.
    """
    count = np.count_nonzero(image > pixel_value)
    return (count / image.size) > threshold

def underexposed_filter(image, threshold=0.03, pixel_value=2):
    """
    Check if the image is underexposed.

    Parameters:
    image (numpy.ndarray): Grayscale image to be checked.
    threshold (float): Proportion of pixels that must be below pixel_value to consider the image underexposed.
    pixel_value (int): Pixel intensity value to check against (0-255).

    Returns:
    bool: True if the image is underexposed, False otherwise.
    """
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
            print("Significant color channel difference in bin", bin_idx)
            plt.bar(bin_idx, hists[0, bin_idx], color='r', alpha=0.5)
            plt.bar(bin_idx, hists[1, bin_idx], color='g', alpha=0.5)
            plt.bar(bin_idx, hists[2, bin_idx], color='b', alpha=0.5)

    if error_bin >= tolerance:
        #print("Image flagged for color imbalance: error bins =", error_bin)
        return False
    else:
        #print("Image passed color balance check: error bins =", error_bin)
        return True 



if __name__ == "__main__":

    good , bad = 0, 0

    for x in range(1,300):
        file = f'images/{x:d}.jpg'
        img_g = cv.imread(file, cv.IMREAD_GRAYSCALE)
        img_bgr = cv.imread(file, cv.IMREAD_COLOR_BGR)
        assert img_g is not None, "file could not be read, check with os.path.exists()," + file

        real_flag = True
        if overexposed_filter(img_g) or overexposed_filter(img_g, pixel_value=220, threshold=0.3):
            real_flag = False
            print("overexposed image", file)
        if underexposed_filter(img_g) or underexposed_filter(img_g, pixel_value=10, threshold=0.30):
            real_flag = False
            print("underexposed image", file)
        if not binned_histogram_analysis(img_bgr):
            real_flag = False
            print("color imbalance image", file)
        if real_flag:
            print("good image", file)
            os.makedirs('good_images', exist_ok=True)
            cv.imwrite(f'good_images/{x:d}.jpg', img_bgr)
            good += 1
        else :
            print("bad image", file)
            os.makedirs('bad_images', exist_ok=True)
            cv.imwrite(f'bad_images/{x:d}.jpg', img_bgr)
            bad += 1

    print("-----Summary-----")
    print("Good images:", good)
    print("Bad images:", bad)


