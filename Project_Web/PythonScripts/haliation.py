import cv2 as cv
import numpy as np
import os
import argparse
import json
import time




def light_source_detection_hsv(HSVimg: np.ndarray, bright_threshold: int = 200) -> np.ndarray: #input HSV image space -> output HSV image space 
    light_mask = cv.inRange(HSVimg, (0, 0, bright_threshold), (180, 255, 255))
    light_sources = cv.bitwise_and(HSVimg, HSVimg, mask=light_mask)
    #cv.imwrite(OUTPUT_DIR + "LightSourcesDetected_HSV.jpg", cv.cvtColor(light_sources, cv.COLOR_HSV2BGR))
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
            light_map = cv.GaussianBlur(light_map, (3,3), sigmaX)
        img[:,:,channel] = light_map # in BGR space

    #cv.imwrite(OUTPUT_DIR + "HaliationMap.jpg", img)

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
    
    #cv.imwrite(OUTPUT_DIR + "final_output.jpg", haliated_image)
    return haliated_image
        
if __name__ == "__main__":

    # create output folder if not exists
    working_directory = os.getcwd()
    folder_name = os.path.join(working_directory, "wwwroot")
    folder_name = os.path.join(folder_name, "output_haliation")
    os.makedirs(folder_name, exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Detect bright light sources and add halation (halo) effect")
    parser.add_argument("input_path", help="Input image path")
    parser.add_argument("--bright_threshold", type=int, default=243, help="Threshold for bright light detection (0-255)")
    parser.add_argument("--kernel_size", type=int, default=55, help="Gaussian kernel size for halation map")
    parser.add_argument("--sigma_x", type=int, default=40, help="Gaussian sigmaX for halation map")
    parser.add_argument("--delta_mode", type=int, help="Use delta mode for halation map generation")
    parser.add_argument("--intensity", type=float, default=0.1, help="Intensity of halation effect")

    args = parser.parse_args()

    # prepare output directory and set global OUTPUT_DIR (functions use this global)
    image_path = args.input_path
    output_path = (folder_name + '/' + os.path.basename(image_path).split('.')[0] + "_" + 
                   str(args.bright_threshold) + "_" + str(args.kernel_size) + "_" + str(args.sigma_x) + "_" + str(args.delta_mode)  + "_" + str(args.intensity) + ".png")
    

    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No input file was found: {image_path}")

    img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)

    ls = light_source_detection_hsv(img_HSV, bright_threshold=args.bright_threshold)
    hm = haliation_map_generator(ls, kernel_size=args.kernel_size, sigmaX=args.sigma_x, delta_mode=args.delta_mode)
    result_image = add_heliation_effect(img, hm, intensity=args.intensity)

    # final file written by add_heliation_effect is OUTPUT_DIR + 'final_output.jpg'
    cv.imwrite(output_path, result_image)
    time.sleep(0.5)
    result = {"filename": os.path.basename(output_path)}
    print(json.dumps(result))

