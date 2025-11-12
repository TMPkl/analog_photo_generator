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

    chanel_scale = np.array([0.2, 0.6, 2.3], dtype=np.float32).reshape(1, 1, 3)
    total_noise *= chanel_scale


    blended = cv.addWeighted(image, 1.0, total_noise, intensity, 0)
    blended = np.clip(blended, 0, 1)

    return (blended * 255).astype(np.uint8)


if __name__ == "__main__":
    img = cv.imread("media/tests/pipline/final_output.jpg")
    if img is None:
        raise FileNotFoundError("Nie znaleziono obrazu: media/tests/esa.jpg")

    img_hls = cv.cvtColor(img, cv.COLOR_BGR2HLS_FULL)


    multiscale_grainy_img = add_multiscale_grain(img_hls)
    cv.imwrite("media/tests/grain/esa_multiscale_grain.jpg", cv.cvtColor(multiscale_grainy_img, cv.COLOR_HLS2BGR_FULL))

    print(" \n")
