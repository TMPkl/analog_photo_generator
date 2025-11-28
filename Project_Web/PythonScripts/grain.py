import argparse
import os
import json
import re
import cv2 as cv
import numpy as np

def add_multiscale_grain(image, scales=(1, 0.2, 0.4, ), intensity=0.4, grain_amplitude=0.18):
    """
    multiscale grain, input in HLS color space
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.1

    h, w, c = image.shape
    total_noise = np.zeros((h, w, c), dtype=np.float32)

    for scale in scales:
        nh, nw = int(h * scale), int(w * scale)
        noise = np.random.normal(0, grain_amplitude, (nh, nw, c)).astype(np.float32)
        noise = cv.resize(noise, (w, h), interpolation=cv.INTER_CUBIC)
        total_noise += noise

    total_noise /= len(scales)
    total_noise = cv.GaussianBlur(total_noise, (3, 3), 0)
    #cv.imwrite("media/tests/grain/esa_multiscale_noise.jpg", cv.cvtColor((total_noise * 255).astype(np.uint8), cv.COLOR_HLS2BGR_FULL))

    chanel_scale = np.array([0.2, 0.6, 2.3], dtype=np.float32).reshape(1, 1, 3) ##hardcoded but it makes the most sense, different values do not look good
    total_noise *= chanel_scale


    blended = cv.addWeighted(image, 1.0, total_noise, intensity, 0)
    blended = np.clip(blended, 0, 1)

    return (blended * 255).astype(np.uint8)


class Scales:
    def __init__(self, values):
        self.values = tuple(float(v) for v in values)

    @classmethod
    def from_string(cls, s: str):
        # split by commas and/or whitespace
        parts = [p for p in re.split(r"[,\s]+", s.strip()) if p != ""]
        return cls(parts)

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return f"Scales({self.values})"


if __name__ == "__main__":


    # create output folder if not exists
    working_directory = os.getcwd()
    folder_name = os.path.join(working_directory, "wwwroot")
    folder_name = os.path.join(folder_name, "output_grain")
    os.makedirs(folder_name, exist_ok=True)

    parser = argparse.ArgumentParser(description="Add multiscale grain to an image (expects BGR input). The script converts to HLS, adds grain and saves output.")
    parser.add_argument("input_path", help="Input image path")
    # Example usages:
    # --scale 1 0.2 0.4
    # --scale "1, 0.2, 0.4"
    # Also accepts a single comma/space-separated string for backward compatibility.
    parser.add_argument("--scale", nargs='+', default="1 0.2 0.4", help="Scales for multiscale noise: eg. '1 0.2 0.4' or '1,0.2,0.4'")
    parser.add_argument("--intensity", type=float, default=0.4, help="Blend intensity for the grain")
    parser.add_argument("--grain_amplitude", type=float, default=0.18, help="Standard deviation for gaussian noise at each scale")

    args = parser.parse_args()

    image_path = args.input_path
    output_path = folder_name + '/' + os.path.basename(image_path).split('.')[0] + "_" + "-".join(args.scale) + "_" + str(args.intensity) + "_" + str(args.grain_amplitude) + ".png"

    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No input file was found: {image_path}")

    # convert to HLS as function expects HLS input
    img_hls = cv.cvtColor(img, cv.COLOR_BGR2HLS_FULL)

    # parse scales: support both a list of values (unquoted usage) and a single string
    if isinstance(args.scale, str):
        scales_obj = Scales.from_string(args.scale)
    else:
        # argparse will produce a list when nargs='+' is used; ensure floats
        scales_obj = Scales(args.scale)
        
    multiscale_grainy_img = add_multiscale_grain(img_hls, scales=scales_obj, intensity=args.intensity, grain_amplitude=args.grain_amplitude)

    # convert back to BGR for saving
    saved = cv.cvtColor(multiscale_grainy_img, cv.COLOR_HLS2BGR_FULL)

    ok = cv.imwrite(output_path, saved)
    if not ok:
        raise IOError(f"Failed to save file: {output_path}")

    result = {
        "filename": os.path.basename(output_path),
    }
    print(json.dumps(result))
