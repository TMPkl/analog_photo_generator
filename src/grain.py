from lib.myFilm import BGR2XYZ, XYZ2BGR
import cv2 as cv
import numpy as np


if __name__ == "__main__":
    bgr_color = [100, 150, 200]  # Example BGR color
    print(f"Original BGR color: {bgr_color}")

    xyz_color = BGR2XYZ(bgr_color)
    print(f"Converted to XYZ color space: {xyz_color}")

    converted_bgr = XYZ2BGR(xyz_color)
    print(f"Converted back to BGR color space: {converted_bgr.tolist()}")

    print("------------------------------------------")

    grain_size = -0.4 #[-0.4-0.15]
    grain_strenght = 0.0 # need to find good range of values
    max_grain_value = 5

    img = cv.imread('media/tests/c.jpg', cv.IMREAD_GRAYSCALE)

    grain_map = np.random.normal(128, 8 + grain_strenght, img.shape) # mapa szumu, srednio 128 szum jest std=8+ grain_strenght
    grain_map = cv.normalize(grain_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8) # normalizujemy do 0-255
    cv.imwrite('media/tests/grain/grain_map_1.jpg', grain_map)
    grain_map = cv.GaussianBlur(grain_map, (3, 3), 0)


    kernel = np.array([[1, 1, 1],
                       [1, -8+grain_size, 1],
                       [1, 1, 1]], dtype=np.float16)
    
    grain_map = cv.filter2D(grain_map, -1, kernel)



    grain_map = cv.normalize(grain_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)    
    cv.imwrite('media/tests/grain/grain_map.jpg', grain_map)

    grain_map = cv.normalize(grain_map, None, -max_grain_value, max_grain_value, cv.NORM_MINMAX)

    # img = img.astype(np.float32) / 255.0

    # # Tworzenie szumu (ziarna)
    # noise = np.random.normal(0, 0.05, img.shape)  # std=0.05 -> siła ziarna

    # # Dodanie szumu i przycięcie zakresu do 0–1
    # grainy = np.clip(img + noise, 0, 1)

    # # Zapis do pliku
    # cv.imwrite('media/tests/grain/photo_grain.jpg', (grainy * 255).astype(np.uint8))