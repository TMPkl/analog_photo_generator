from lib.myFilm import BGR2XYZ, XYZ2BGR
import cv2 as cv
import numpy as np


def create_noise_texture():
    grain_size = -1.2 #[-1.2 mam ale do wystestowania]
    grain_strenght = 0.02 # need to find good range of values
    max_grain_value = 5

    img = cv.imread('media/tests/c.jpg', cv.IMREAD_GRAYSCALE)

    grain_random = np.random.normal(128, 2, img.shape) # mapa szumu, srednio 128 szum jest std=8+ grain_strenght
    grain_random_n = cv.normalize(grain_random, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8) # normalizujemy do 0-255
    cv.imwrite('media/tests/grain/grain_map_1.jpg', grain_random_n)

    kernel = np.array([[1, 1, 1],
                       [1, -8+grain_size, 1],
                       [1, 1, 1]], dtype=np.float16)
    


    grain_map = cv.filter2D(grain_random_n, -1, kernel)

    grain_map = cv.normalize(grain_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)        
    
    out = np.where(grain_map == 0, 127, grain_map * grain_random*grain_strenght).astype(np.uint8)
    out = cv.GaussianBlur(out, (3, 3), 0)

    cv.imwrite('media/tests/grain/photo_grain.jpg', out)
    print(np.mean(out),np.std(out))
    return out

if __name__ == "__main__":
    bgr_color = [100, 150, 200]  # Example BGR color
    print(f"Original BGR color: {bgr_color}")

    xyz_color = BGR2XYZ(bgr_color)
    print(f"Converted to XYZ color space: {xyz_color}")

    converted_bgr = XYZ2BGR(xyz_color)
    print(f"Converted back to BGR color space: {converted_bgr.tolist()}")

    print("------------------------------------------")

    create_noise_texture()

    # img = img.astype(np.float32) / 255.0

    # # Tworzenie szumu (ziarna)
    # noise = np.random.normal(0, 0.05, img.shape)  # std=0.05 -> siła ziarna

    # # Dodanie szumu i przycięcie zakresu do 0–1
    # grainy = np.clip(img + noise, 0, 1)

    # # Zapis do pliku
    # cv.imwrite('media/tests/grain/photo_grain.jpg', (grainy * 255).astype(np.uint8))