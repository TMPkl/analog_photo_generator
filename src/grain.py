from lib.myFilm import *
import cv2 as cv
import numpy as np


def create_noise_img(grain_size=-1.2, grain_strength=0.02, save_steps=False):
    img = cv.imread('media/tests/r.png', cv.IMREAD_GRAYSCALE)  # testowy obraz
    img = cv.resize(img, (img.shape[1]*4, img.shape[0]*4))  

    # losowy szum (średnia 128, odchylenie 2)
    grain_random = np.random.normal(128, 2, img.shape)
    grain_random_n = cv.normalize(grain_random, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    if save_steps:
        cv.imwrite('media/tests/grain/grain_map_1.jpg', grain_random_n)

    # filtr do ziarna
    kernel = np.array([[1, 1, 1],
                       [1, -8 + grain_size, 1],
                       [1, 1, 1]], dtype=np.float32)
    
    grain_map = cv.filter2D(grain_random_n, -1, kernel)
    grain_map = cv.normalize(grain_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # połączenie map i rozmycie
    out = np.where(grain_map == 0, 127, grain_map * grain_random * grain_strength).astype(np.float32)
    out = cv.GaussianBlur(out, (3, 3), 0)

    if save_steps:
        cv.imwrite('media/tests/grain/photo_grain.jpg', np.uint8(out))

    # --- Normalizacja do [-0.5, 0.5] ze średnią 0 ---
    mean_val = np.mean(out)
    out -= mean_val
    max_abs = np.max(np.abs(out))
    if max_abs != 0:
        out /= (2 * max_abs)  # teraz ~[-0.5, 0.5]

    return out.astype(np.float32)


def desaturate_img(img, factor=0.5):
    """
    Zmniejsza nasycenie kolorów w obrazie.
    factor = 1.0 -> brak zmian
    factor = 0.0 -> czarno-białe
    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv.split(hsv)

    s *= factor
    s = np.clip(s, 0, 255)

    hsv = cv.merge([h, s, v]).astype(np.uint8)
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


def three_channel_grain_img():
    """
    Tworzy 3-kanałowy szum (BGR), złożony z trzech niezależnych masek.
    """
    noise_xyz = cv.merge([
        create_noise_img(),
        create_noise_img(),
        create_noise_img()
    ])
    return XYZ2BGR(noise_xyz)


def apply_grain_mask(img, img_noise, grain_strength=0.05):
    """
    Nakłada maskę szumów ([-0.5, 0.5]) na obraz w sposób:
        pixel_out = pixel + noise * pixel * grain_strength
    """
    img = img.astype(np.float32)
    img_noise = cv.resize(img_noise, (img.shape[1], img.shape[0]))  # <- poprawka resize
    out = img + img_noise * img * grain_strength
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


if __name__ == "__main__":
    img = cv.imread('media/tests/esa.jpg', cv.IMREAD_GRAYSCALE)
    mask = create_noise_img()
    img_grain = apply_grain_mask(img, mask, grain_strength=0.25)

    cv.imwrite('media/tests/grain/photo_with_grain.jpg', img_grain)
    print("✅ zapisano obraz z ziarnem: media/tests/grain/photo_with_grain.jpg")
