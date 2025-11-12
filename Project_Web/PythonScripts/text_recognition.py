import easyocr
import cv2
import os
import argparse
import json
import math

def sort_results(results, y_threshold=10):
    # Sort by Y first
    results.sort(key=lambda r: r[0][0][1])

    sorted_results = []
    current_line = []
    last_y = None

    for item in results:
        y = item[0][0][1]
        if last_y is None or abs(y - last_y) < y_threshold:
            current_line.append(item)
            last_y = y if last_y is None else (last_y + y) / 2
        else:
            # Sort current line left-to-right and add to final
            current_line.sort(key=lambda r: r[0][0][0])  # X of top-left
            sorted_results.extend(current_line)
            current_line = [item]
            last_y = y

    if current_line:
        current_line.sort(key=lambda r: r[0][0][0])
        sorted_results.extend(current_line)

    return sorted_results


working_directory = os.getcwd()
folder_name = os.path.join(working_directory, "wwwroot")
folder_name = os.path.join(folder_name, "output_text_recognition")
os.makedirs(folder_name, exist_ok=True)

# Create the parser
parser = argparse.ArgumentParser(description="Text detection tool")

# Add arguments
parser.add_argument('input_path', type=str, help="The path of the input image")
args = parser.parse_args()

image_path = args.input_path
output_path = folder_name + '/' + os.path.basename(image_path).split('.')[0] + ".png"

reader = easyocr.Reader(['en'])

# Read the image
image = cv2.imread(image_path)

# Perform text detection and recognition
results = reader.readtext(image)

results = sort_results(results)

all_text = []

# Draw bounding boxes around detected text
for (bbox, text, prob) in results:
    
    all_text.append(text)
    
    (top_left, top_right, bottom_right, bottom_left) = bbox

    # Convert to int tuples
    top_left = (int(top_left[0]), int(top_left[1]))
    bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

    # Draw box and text
    # cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # Determine size of text
    font_scale = 0.5
    font_thickness = 1
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

    # Coordinates for the filled rectangle background
    text_x = top_left[0]
    text_y = (top_left[1] + bottom_right[1]) // 2
    text_y = max(text_y, 0)  # ensure text is not out of bounds

    box_coords = ((text_x, text_y - text_size[1] - 4), (text_x + text_size[0] + 4, text_y))
    cv2.rectangle(image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)

    # Draw text on top
    cv2.putText(image, text, (text_x + 2, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 128, 0), font_thickness)

cv2.imwrite(output_path, image)

result = {
    "filename": os.path.basename(output_path),
    "text": " ".join(all_text)
}
print(json.dumps(result))

