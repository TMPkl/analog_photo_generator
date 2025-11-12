from rembg import remove
from PIL import Image
import os
import argparse
import json

# create output folder if not exists
working_directory = os.getcwd()
folder_name = os.path.join(working_directory, "wwwroot")
folder_name = os.path.join(folder_name, "output_background_removal")
os.makedirs(folder_name, exist_ok=True)


# Create the parser
parser = argparse.ArgumentParser(description="Background removal tool")

# Add arguments
parser.add_argument('input_path', type=str, help="The path of the input image")
args = parser.parse_args()

image_path = args.input_path
output_path = folder_name + '/' + os.path.basename(image_path).split('.')[0] + ".png"

input_image = Image.open(image_path)

# Remove the background
output_image = remove(input_image)

output_image.save(output_path, format="PNG")

result = {
    "filename": os.path.basename(output_path),
}
print(json.dumps(result))


