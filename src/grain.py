import argparse
import os
import json
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
    cv.imwrite("media/tests/grain/esa_multiscale_noise.jpg", cv.cvtColor((total_noise * 255).astype(np.uint8), cv.COLOR_HLS2BGR_FULL))

    chanel_scale = np.array([0.2, 0.6, 2.3], dtype=np.float32).reshape(1, 1, 3) ##hardcoded but it makes the most sense, different values do not look good
    total_noise *= chanel_scale


    blended = cv.addWeighted(image, 1.0, total_noise, intensity, 0)
    blended = np.clip(blended, 0, 1)

    return (blended * 255).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add multiscale grain to an image (expects BGR input). The script converts to HLS, adds grain and saves output.")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", nargs="?", default="media/tests/grain/esa_multiscale_grain.jpg", help="Output image path")
    parser.add_argument("--scales", type=float, nargs="+", default=[1, 0.2, 0.4], help="Scales for multiscale noise (space-separated floats)")
    parser.add_argument("--intensity", type=float, default=0.4, help="Blend intensity for the grain")
    parser.add_argument("--grain_amplitude", type=float, default=0.18, help="Standard deviation for gaussian noise at each scale")

    args = parser.parse_args()

    img = cv.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Nie znaleziono obrazu: {args.input}")

    # convert to HLS as function expects HLS input
    img_hls = cv.cvtColor(img, cv.COLOR_BGR2HLS_FULL)

    multiscale_grainy_img = add_multiscale_grain(img_hls, scales=tuple(args.scales), intensity=args.intensity, grain_amplitude=args.grain_amplitude)

    # convert back to BGR for saving
    saved = cv.cvtColor(multiscale_grainy_img, cv.COLOR_HLS2BGR_FULL)

    abs_output = os.path.abspath(args.output)
    out_dir = os.path.dirname(abs_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ok = cv.imwrite(abs_output, saved)
    if not ok:
        raise IOError(f"Nie udało się zapisać pliku: {abs_output}")

    result = {
        "filename": os.path.basename(abs_output),
    }
    print(json.dumps(result))
