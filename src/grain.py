from lib.myFilm import BGR2XYZ, XYZ2BGR
import cv2 as cv
import numpy as np


def create_noise_img(grain_size=-1.2, grain_strenght=0.02, save_steps =False):
    img = cv.imread('media/tests/esa.jpg', cv.IMREAD_GRAYSCALE) ### for test purposes only, need the dimensions of img

    grain_random = np.random.normal(128, 2, img.shape) # mapa szumu, srednio 128 szum jest std=8+ grain_strenght
    grain_random_n = cv.normalize(grain_random, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8) # normalizujemy do 0-255
    if save_steps:
        cv.imwrite('media/tests/grain/grain_map_1.jpg', grain_random_n)

    kernel = np.array([[1, 1, 1],
                       [1, -8+grain_size, 1],
                       [1, 1, 1]], dtype=np.float16)
    


    grain_map = cv.filter2D(grain_random_n, -1, kernel)

    grain_map = cv.normalize(grain_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)        
    
    out = np.where(grain_map == 0, 127, grain_map * grain_random*grain_strenght).astype(np.uint8)
    out = cv.GaussianBlur(out, (3, 3), 0)
    if save_steps:
        cv.imwrite('media/tests/grain/photo_grain.jpg', out)

    out = cv.normalize(out, None, 0,1,cv.NORM_MINMAX).astype(np.float32)
    return out


def normalize_to_target_color(img, target_color=(128, 128, 128)):
    img = img.astype(np.float32)
    mean_color = np.mean(img, axis=(0, 1), keepdims=True)

    img_out = img - mean_color + np.array(target_color, dtype=np.float32)
    img_out = np.clip(img_out, 0, 255)
    return img_out.astype(np.uint8)

def desaturate_img(img, factor=0.5): ##input BGR image
    """
    Zmniejsza nasycenie kolorów w obrazie.
    factor = 1.0 -> brak zmian
    factor = 0.0 -> całkowita desaturacja (czarno-białe)
    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv.split(hsv)

    s *= factor  # zmniejsz nasycenie
    s = np.clip(s, 0, 255)

    hsv = cv.merge([h, s, v]).astype(np.uint8)
    out = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return out

def three_channel_grain_img():
    out = XYZ2BGR(cv.merge([create_noise_img(), create_noise_img(), create_noise_img()]))
    return out

if __name__ == "__main__":
    #img = cv.imread('media/tests/c.jpg')
    #img = BGR2XYZ(img)

    noise = three_channel_grain_img()
    cv.imwrite('media/tests/grain/photo_grain_3ch_raw.jpg', noise)
    
    noise = normalize_to_target_color(noise, target_color=(128,128,128))
    cv.imwrite('media/tests/grain/photo_grain_3ch_normalized.jpg', noise)

    noise = cv.GaussianBlur(noise, (15, 15), 15)
    cv.imwrite('media/tests/grain/photo_grain_3ch_blurred.jpg', noise)

    noise = desaturate_img(noise, factor=0.25)
    cv.imwrite('media/tests/grain/photo_grain_3ch_desaturated.jpg', noise)
    print("------------------------------------------")

    esa_orginal = cv.imread('media/tests/esa.jpg')
    esa_with_grain = cv.addWeighted(esa_orginal.astype(np.float32), 1.0, noise.astype(np.float32), 0.3, 0)
    cv.imwrite('media/tests/grain/esa_with_grain.jpg', esa_with_grain.astype(np.uint8))


    create_noise_img()

    # img = img.astype(np.float32) / 255.0

    # # Tworzenie szumu (ziarna)
    # noise = np.random.normal(0, 0.05, img.shape)  # std=0.05 -> siła ziarna

    # # Dodanie szumu i przycięcie zakresu do 0–1
    # grainy = np.clip(img + noise, 0, 1)

    # # Zapis do pliku
    # cv.imwrite('media/tests/grain/photo_grain.jpg', (grainy * 255).astype(np.uint8))