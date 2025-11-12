from retinaface import RetinaFace
import cv2
import os
import argparse
import json

working_directory = os.getcwd()
folder_name = os.path.join(working_directory, "wwwroot")
folder_name = os.path.join(folder_name, "output_face_detection")
os.makedirs(folder_name, exist_ok=True)

# Create the parser
parser = argparse.ArgumentParser(description="Face detection tool")

# Add arguments
parser.add_argument('input_path', type=str, help="The path of the input image")
args = parser.parse_args()

image_path = args.input_path
output_path = folder_name + '/' + os.path.basename(image_path).split('.')[0] + ".png"

image = cv2.imread(image_path)

# Detect faces using RetinaFace
faces = RetinaFace.detect_faces(image_path)

# Draw rectangles around faces
for key in faces:
    face = faces[key]
    x1, y1, x2, y2 = face['facial_area']
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite(output_path, image)

result = {
    "filename": os.path.basename(output_path),
}
print(json.dumps(result))