'''
    TO DO:
    - color of light sources detection based on the color temperature
    - haliation can not be stronger than the light
    - haliation should not be present on the light source itself
    - better haliation map generation (make it automatic parameters based on light density on the histogram)

'''


import cv2 as cv
import numpy as np
import os
import argparse

OUTPUT_DIR = "../media/tests/haliation/output/"



def light_source_detection(img, bright_threshold=200):
    # Make a copy
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert to grayscale
    # Threshold
    _, thresh = cv.threshold(img_gray, bright_threshold, 255, cv.THRESH_BINARY)
    cv.imwrite("../media/tests/haliation/output/ThresholdedImage.jpg", thresh)

    output_img = cv.dilate(thresh, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), iterations=3)  # jeśli jest otoczony przez małe ciemne piksele, to je wypełnij jeżeli jest pojedyńczy piksel to go zwiększy
    cv.imwrite("../media/tests/haliation/output/dilatedImage.jpg", output_img)

    # Find contours on the single-channel binary image
    # contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # light_sources = []
    # for cnt in contours:
    #     area = cv.contourArea(cnt)
    #     print(f"Contour area: {area}")
    #     x, y, w, h = cv.boundingRect(cnt)
    #     light_sources.append((x, y, w, h))

    # # Draw contours on the original image for visualization
    # output = img.copy()
    # cv.drawContours(output, contours, -1, (0, 255, 0), 2)

    output_path = OUTPUT_DIR + "detected_lights.jpg"
    cv.imwrite(output_path, output_img)
    print(f"Wynik zapisano do pliku: {output_path}")
    return output_img

def haliation_map_generator(image, kernel_size=5, sigmaX=5):
    """
         test function, take image genereated by light_source_detection and create haliation map
    """
    haliation_map = cv.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigmaX)
    cv.imwrite(OUTPUT_DIR + "haliation_map.jpg", haliation_map)
    return haliation_map


def add_heliation_effect(image, haliation_map, intensity=0.5):
    """
        function to add haliation effect to the original image using the haliation map
    """
    # Ensure haliation_map is single channel
    if len(haliation_map.shape) == 3:
        haliation_map = cv.cvtColor(haliation_map, cv.COLOR_BGR2GRAY)

    # Normalize haliation map to range [0, 1]
    normalized_map = haliation_map / 255.0

    # Expand dimensions to match image shape
    if len(image.shape) == 3 and image.shape[2] == 3:
        normalized_map = cv.merge([normalized_map]*3)

    # Create haliated image
    haliated_image = cv.addWeighted(image.astype(np.float32), 1.0, (normalized_map * intensity * 255).astype(np.float32), intensity, 0)
    haliated_image = np.clip(haliated_image, 0, 255).astype(np.uint8)

    output_path = OUTPUT_DIR + "final_output.jpg"
    cv.imwrite(output_path, haliated_image)
    print(f"Haliated image saved to: {output_path}")
    return haliated_image


if __name__ == "__main__":
    print("------- Light Source Detection -------\n")
    input_img_pth = "../media/tests/haliation/robert-tudor.jpg"
    image = cv.imread(input_img_pth)
    light_sources = light_source_detection(image, bright_threshold=210)
    haliation_map = haliation_map_generator(light_sources, kernel_size=31, sigmaX=50)
    haliated_image = add_heliation_effect(image, haliation_map, intensity=0.7)
    print("------- Process Finished -------\n")
    # cv.imshow("Detected Light Sources", light_sources)    # cv.waitKey(0)
    # cv.destroyAllWindows()
