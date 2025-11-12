import cv2
import os
import argparse
import json


working_directory = os.getcwd()
folder_name = os.path.join(working_directory, "wwwroot")
folder_name = os.path.join(folder_name, "output_color_space_transformation")
os.makedirs(folder_name, exist_ok=True)

# Create the parser
parser = argparse.ArgumentParser(description="Resize tool")

# Add arguments
parser.add_argument('input_path', type=str, help="The path of the input image")
parser.add_argument('color_space', type=str, help="The new width")
args = parser.parse_args()

image_path = args.input_path
output_path = folder_name + '/' + os.path.basename(image_path).split('.')[0] + "_" + args.color_space + ".png"

image = cv2.imread(image_path)

if args.color_space == "GRAY":
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
elif args.color_space == "HSV":
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
elif args.color_space == "BGR":
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
else:
    print("The color space must be: GRAY, HSV or BGR")

cv2.imwrite(output_path, new_image)

result = {
    "filename": os.path.basename(output_path),
}
print(json.dumps(result))