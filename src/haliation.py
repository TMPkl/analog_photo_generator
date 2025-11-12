import cv2 as cv
import numpy as np
import os
import argparse

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
    print(f"Haliated image saved to: {OUTPUT_DIR + 'final_output.jpg'})")
    return haliated_image
        
if __name__ == "__main__":
    print("------- Light Source Detection -------\n")
    input_img_pth = "media/tests/p2.jpg"
    img_HSV = cv.cvtColor(cv.imread(input_img_pth), cv.COLOR_BGR2HSV_FULL)
    
    ls = light_source_detection_hsv(img_HSV, bright_threshold=243) #threshold
    hm = haliation_map_generator(ls, kernel_size=55, sigmaX=40, delta_mode=False)  # was
    h_img = add_heliation_effect(cv.imread(input_img_pth), hm, intensity=0.1) #inensity
    print("------- Process Finished -------\n")

