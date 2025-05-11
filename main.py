from PIL import Image
import numpy as np
from YCrCb import YCbCr2RGB, RGB2YCbCr
from Down_Sampling import down_sampling, up_sampling
from Blocking import blocks, unblock
from DCT import DCT, iDCT
from Quantize import quant_matrix
from Zigzag import zigzag, back_zigzag
from Coder import encoding_blocks, decoding_blocks

Q_Y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

Q_C = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

pics = ["t", "Lenna", "Big_picture", "Big_picture_grayscale",
        "Big_picture_bw_no_dither", "Big_picture_bw_dithered"]
for pic in pics:
    # 1) Открытие изображения
    img = Image.open(f'test_data/{pic}.png')
    if img.mode in ('L', '1'):
        img = img.convert('RGB')
    img_val = np.array(img).astype(np.uint8)

    # 2) Переход в YCbCr
    YCbCr = RGB2YCbCr(img_val)

    # 3) Даунсемплинг цветовых каналов 4:2:0
    Y = YCbCr[:, :, 0]
    Cb, c_size = down_sampling(YCbCr[:, :, 1])
    Cr, c_size = down_sampling(YCbCr[:, :, 2])

    # 4) Блоки
    Y = blocks(Y, 8)
    Cb = blocks(Cb, 8)
    Cr = blocks(Cr, 8)

    # 5) ДКП
    Y = [Y[0]] + DCT(Y[1::])
    Cr = [Cr[0]] + DCT(Cr[1::])
    Cb = [Cb[0]] + DCT(Cb[1::])

    # 6) Подготовка матриц
    Q_Y = quant_matrix(Q_Y, 100)
    Q_C = quant_matrix(Q_C, 100)

    # 7) Квантование
    Y = [Y[0]] + [np.round(block / Q_Y) for block in Y[1::]]
    Cr = [Cr[0]] + [np.round(block / Q_C) for block in Cr[1::]]
    Cb = [Cb[0]] + [np.round(block / Q_C) for block in Cb[1::]]

    # 8) Зигзаг
    Y = [Y[0]] + zigzag(Y[1::])
    Cr = [Cr[0]] + zigzag(Cr[1::])
    Cb = [Cb[0]] + zigzag(Cb[1::])

    # 9) ACDC + HA
    t = [(512, 512),
         [100, 10, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
          0, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [110, 2, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
          0, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    Y = encoding_blocks(Y)
    # Cr = encoding_blocks(Cr, 1)
    # Cb = encoding_blocks(Cb, 1)

    # обратно
    Y = decoding_blocks(Y)
    a = Y
    for i in range(1, len(a)):
        for j in range(len(a[i])):
            if a[i][j] != t[i][j]:
                print(i, j, a[i][j], t[i][j])

    # Cr = decoding_blocks(Cr, 1)
    # Cb = decoding_blocks(Cb, 1)

    Y = [Y[0]] + back_zigzag(Y[1::])
    Cr = [Cr[0]] + back_zigzag(Cr[1::])
    Cb = [Cb[0]] + back_zigzag(Cb[1::])

    Y = [Y[0]] + [block * Q_Y for block in Y[1::]]
    Cr = [Cr[0]] + [block * Q_C for block in Cr[1::]]
    Cb = [Cb[0]] + [block * Q_C for block in Cb[1::]]

    Y = [Y[0]] + iDCT(Y[1::])
    Cr = [Cr[0]] + iDCT(Cr[1::])
    Cb = [Cb[0]] + iDCT(Cb[1::])

    Y = unblock(Y[1::], Y[0])
    Cb = unblock(Cb[1::], Cb[0])
    Cr = unblock(Cr[1::], Cr[0])

    Cb = up_sampling(Cb, c_size)
    Cr = up_sampling(Cr, c_size)

    ycc = np.dstack((Y, Cb, Cr))
    r = YCbCr2RGB(ycc)
    t = Image.fromarray(r)
    t.show()
