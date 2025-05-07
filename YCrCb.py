import numpy as np


def RGB2YCbCr(im):
    x, y, _ = im.shape
    YCbCr = np.zeros_like(im, dtype=np.uint8)
    for i in range(x):
        for j in range(y):
            R, G, B = im[i][j] / 255.0
            Y = 16 + (65.481 * R + 128.553 * G + 24.966 * B)
            Cb = 128 + (-37.797 * R - 74.203 * G + 112.0 * B)
            Cr = 128 + (112.0 * R - 93.786 * G - 18.214 * B)
            YCbCr[i, j] = [Y, Cb, Cr]
    return YCbCr


def YCbCr2RGB(im):
    x, y, _ = im.shape
    RGB = np.zeros_like(im, dtype=np.uint8)
    for i in range(x):
        for j in range(y):
            Y, Cb, Cr = im[i][j].astype(np.float32)
            R = round(1.164 * (Y - 16) + 1.596 * (Cr - 128))
            G = round(1.164 * (Y - 16) - 0.392 * (Cb - 128) - 0.813 * (Cr - 128))
            B = round(1.164 * (Y - 16) + 2.017 * (Cb - 128))

            R = np.clip(R, 0, 255)
            G = np.clip(G, 0, 255)
            B = np.clip(B, 0, 255)

            RGB[i, j] = [R, G, B]
    return RGB
