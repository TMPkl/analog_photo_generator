import cv2 as cv
import numpy as np
import os
import argparse
import json

OUTPUT_DIR = "media/tests/pipline/"


def light_source_detection_hsv(HSVimg: np.ndarray, bright_threshold: int = 200) -> np.ndarray: #input HSV image space -> output HSV image space 
    light_mask = cv.inRange(HSVimg, (0, 0, bright_threshold), (180, 255, 255))
    light_sources = cv.bitwise_and(HSVimg, HSVimg, mask=light_mask)
    cv.imwrite(OUTPUT_DIR + "LightSourcesDetected_HSV.jpg", cv.cvtColor(light_sources, cv.COLOR_HSV2BGR))
    return light_sources


def haliation_map_generator(img: np.ndarray, kernel_size:int = 31, sigmaX:int = 50, delta_mode: bool = False) -> np.ndarray:

    """
        input: image in HSV space with detected light sources
        output: haliation map in BGR space
    """

    BGR_img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    
    for channel in range(3):
        light_map = cv.GaussianBlur(BGR_img[:,:,channel], (kernel_size, kernel_size), sigmaX)
        if delta_mode:
            light_map = cv.subtract(light_map, BGR_img[:,:,channel])
            light_map = cv.GaussianBlur(light_map, (3,3))
        img[:,:,channel] = light_map # in BGR space

    cv.imwrite(OUTPUT_DIR + "HaliationMap.jpg", img)

    return img


def add_heliation_effect(image: np.ndarray, haliation_map: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
        function to add haliation effect to the original image using the haliation map
        input: original image in BGR space, haliation map in BGR space, intensity of the effect
        output: haliated image in BGR space
    """
    haliated_image = np.zeros_like(image)
    for channel in range(3):
        haliated_image[:, :, channel] = cv.addWeighted(image[:, :, channel], 1.0, haliation_map[:, :, channel], intensity, 0)
    
    cv.imwrite(OUTPUT_DIR + "final_output.jpg", haliated_image)
    return haliated_image
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect bright light sources and add haliation (halo) effect")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", nargs="?", default="media/tests/pipline/final_output.jpg", help="Output image path")
    parser.add_argument("--bright_threshold", type=int, default=243, help="Threshold for bright light detection (0-255)")
    parser.add_argument("--kernel_size", type=int, default=55, help="Gaussian kernel size for haliation map")
    parser.add_argument("--sigmaX", type=int, default=40, help="Gaussian sigmaX for haliation map")
    parser.add_argument("--delta_mode", action="store_true", help="Use delta mode for haliation map generation")
    parser.add_argument("--intensity", type=float, default=0.1, help="Intensity of haliation effect")

    args = parser.parse_args()

    # prepare output directory and set global OUTPUT_DIR (functions use this global)
    abs_output = os.path.abspath(args.output)
    out_dir = os.path.dirname(abs_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # ensure trailing slash used by functions
    globals()["OUTPUT_DIR"] = out_dir + os.sep

    img = cv.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"No input file was found: {args.input}")

    img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)

    ls = light_source_detection_hsv(img_HSV, bright_threshold=args.bright_threshold)
    hm = haliation_map_generator(ls, kernel_size=args.kernel_size, sigmaX=args.sigmaX, delta_mode=args.delta_mode)
    _ = add_heliation_effect(cv.imread(args.input), hm, intensity=args.intensity)

    # final file written by add_heliation_effect is OUTPUT_DIR + 'final_output.jpg'
    final_path = os.path.join(globals()["OUTPUT_DIR"], "final_output.jpg")
    result = {"filename": os.path.basename(final_path)}
    print(json.dumps(result))

