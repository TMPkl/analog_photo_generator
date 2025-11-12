import cv2
import os
import argparse
import numpy as np
import json

working_directory = os.getcwd()
folder_name = os.path.join(working_directory, "wwwroot")
folder_name = os.path.join(folder_name, "output_contour_detection")
os.makedirs(folder_name, exist_ok=True)

# Create the parser
parser = argparse.ArgumentParser(description="Contour detection tool")

# Add arguments
parser.add_argument('input_path', type=str, help="The path of the input image")
args = parser.parse_args()

image_path = args.input_path
output_path = folder_name + '/' + os.path.basename(image_path).split('.')[0] + ".png"

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to prepare for watershed
ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# Noise removal using opening and closing
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Dist Transform (helps in finding sure regions)
dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region (background)
sure_bg = cv2.dilate(thresh, kernel, iterations=3)

# Convert sure_fg to uint8 for subtraction
sure_fg = cv2.convertScaleAbs(sure_fg)

# Subtract to get the unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Applying watershed
markers = markers + 1
markers[unknown == 255] = 0
cv2.watershed(image, markers)

# Mark boundaries
image[markers == -1] = [0, 0, 255]

# Save or display the result
cv2.imwrite(output_path, image)

result = {
    "filename": os.path.basename(output_path),
}
print(json.dumps(result))
