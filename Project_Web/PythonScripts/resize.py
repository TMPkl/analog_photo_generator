import cv2
import os
import argparse
import json

working_directory = os.getcwd()
folder_name = os.path.join(working_directory, "wwwroot")
folder_name = os.path.join(folder_name, "output_resize")
os.makedirs(folder_name, exist_ok=True)

# Create the parser
parser = argparse.ArgumentParser(description="Resize tool")

# Add arguments
parser.add_argument('input_path', type=str, help="The path of the input image")
parser.add_argument('-w', '--width', type=int, required=True, help="The new width")
parser.add_argument('-H', '--height', type=int, required=True, help="The new height")
args = parser.parse_args()

image_path = args.input_path
output_path = folder_name + '/' + os.path.basename(image_path).split('.')[0] + "_" + str(args.width) + "x" + str(args.height) + ".png"

image = cv2.imread(image_path)

# image_height, image_width, _ = image.shape
# aspect_ratio = image_width / image_height
# 
# eps = 10**(-7)
# 
# if abs(args.width / args.height - aspect_ratio) > eps:
#     print("The aspect ratio is incorrect")
#     exit()

resized_image = cv2.resize(image, (args.height, args.width), interpolation=cv2.INTER_CUBIC)

cv2.imwrite(output_path, resized_image)

result = {
    "filename": os.path.basename(output_path),
}
print(json.dumps(result))