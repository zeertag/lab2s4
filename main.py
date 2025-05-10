from PIL import Image
import numpy as np
import os

from YCrCb import YCbCr2RGB, RGB2YCbCr
from Down_Sampling import down_sampling, up_sampling
from Blocking import blocks, unblock
from DCT import DCT, iDCT
from Quantize import scale_quant_matrix
from Zigzag import zigzag, back_zigzag
from DC import dif_dc, back_dc
# from ACDC import encode_blocks, decode_blocks
# from Ha import Huffman_JPEG_encode, Huffman_JPEG_decode
from pls_work import encode_blocks, decode_blocks

t = 0


def jpeg_encoding(quality, path):
    global t
    '''матрицы квантования'''
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

    # 1) открываю изображение
    img = Image.open(path)
    if img.mode in ('L', '1'):
        img = img.convert('RGB')
    img_val = np.array(img).astype(np.uint8)

    # 2) Перевожу его из формата RGB в YCrCb
    YCbCr_val = RGB2YCbCr(img_val)

    # 3) Даунсемплинг по каналам (4:2:0)
    Y = YCbCr_val[:, :, 0]
    Cr = down_sampling(YCbCr_val[:, :, 1])
    Cb = down_sampling(YCbCr_val[:, :, 2])

    # 4) Чтение по блокам
    Y = blocks(Y, 8)
    Cr = blocks(Cr, 8)
    Cb = blocks(Cb, 8)

    # 5) 2D DCT-II
    Y = [Y[0]] + DCT(Y[1::])
    Cr = [Cr[0]] + DCT(Cr[1::])
    Cb = [Cb[0]] + DCT(Cb[1::])

    # 6) Изменения матрицы квантования
    Q_Y = scale_quant_matrix(Q_Y, quality)
    Q_C = scale_quant_matrix(Q_C, quality)

    # 7) Квантование и обратное преобразование матрицы DCT по заданной матрице квантования
    Y = [Y[0]] + [np.round(block / Q_Y).astype(np.int16) for block in Y[1::]]
    Cr = [Cr[0]] + [np.round(block / Q_C).astype(np.int16) for block in Cr[1::]]
    Cb = [Cb[0]] + [np.round(block / Q_C).astype(np.int16) for block in Cb[1::]]

    # 8) Зигзаг обход
    Y = [Y[0]] + zigzag(Y[1::])
    Cr = [Cr[0]] + zigzag(Cr[1::])
    Cb = [Cb[0]] + zigzag(Cb[1::])

    # 9) Разностное кодирование DC коэф
    Y = [Y[0]] + dif_dc(Y[1::])
    Cr = [Cr[0]] + dif_dc(Cr[1::])
    Cb = [Cb[0]] + dif_dc(Cb[1::])
    t = Cr

    Y = encode_blocks(Y)
    Cr = encode_blocks(Cr, 1)
    Cb = encode_blocks(Cb, 1)
    # # 10 - 11) Переменное кодирование разностей DC и AC коэф + RLE кодирование AC коэффициентов
    # Y = [Y[0]] + encode_blocks(Y[1::])
    # Cr = [Cr[0]] + encode_blocks(Cr[1::])
    # Cb = [Cb[0]] + encode_blocks(Cb[1::])
    #
    # # 12) Кодирование разностей DC коэффициентов и последовательностей Run/Size по таблице кодов Хаффмана
    # Y = Huffman_JPEG_encode(Y)
    # Cr = Huffman_JPEG_encode(Cr)
    # Cb = Huffman_JPEG_encode(Cb)

    len_Y = len(Y)
    len_Cr = len(Cr)
    len_Cb = len(Cb)

    jpeg_data = (len_Y.to_bytes(4, 'big') + len_Cr.to_bytes(4, 'big') +
                 len_Cb.to_bytes(4, 'big') + Y + Cr + Cb)

    with open("Results/out", "wb") as f:
        f.write(jpeg_data)

    return jpeg_data


def jpeg_decode(quality, path="Results/out"):
    global t
    '''матрицы квантования'''
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

    with open(f"{path}", "rb") as file:
        data = file.read()
    len_Y = int.from_bytes(data[:4], 'big')
    len_Cr = int.from_bytes(data[4:8], 'big')

    Y = data[12:12 + len_Y]
    Cr = data[12 + len_Y:12 + len_Y + len_Cr]
    Cb = data[12 + len_Y + len_Cr::]

    # Хаффман
    Y = decode_blocks(Y)
    Cr = decode_blocks(Cr, 1)
    # for i in range(len(Cr)):
    #     if Cr[i] != t[i]:
    #         print(i, Cr[i], t[i])
    Cb = decode_blocks(Cb, 1)
    # Y = Huffman_JPEG_decode(Y)
    # Cr = Huffman_JPEG_decode(Cr)
    # Cb = Huffman_JPEG_decode(Cb)
    #
    # # ACDC
    # Y = [Y[0]] + decode_blocks(Y[1::]).tolist()
    # Cr = [Cr[0]] + decode_blocks(Cr[1::]).tolist()
    # Cb = [Cb[0]] + decode_blocks(Cb[1::]).tolist()

    # Разностное кодирование
    Y = [Y[0]] + back_dc(Y[1::])
    Cr = [Cr[0]] + back_dc(Cr[1::])
    Cb = [Cb[0]] + back_dc(Cb[1::])

    # Зигзаг
    Y = [Y[0]] + back_zigzag(Y[1::])
    Cr = [Cr[0]] + back_zigzag(Cr[1::])
    Cb = [Cb[0]] + back_zigzag(Cb[1::])

    # Квантования
    Q_Y = scale_quant_matrix(Q_Y, quality)
    Q_C = scale_quant_matrix(Q_C, quality)

    Y = [Y[0]] + [block * Q_Y for block in Y[1::]]
    Cr = [Cr[0]] + [block * Q_C for block in Cr[1::]]
    Cb = [Cb[0]] + [block * Q_C for block in Cb[1::]]

    # 2D DCT-II
    Y = [Y[0]] + iDCT(Y[1::])
    Cr = [Cr[0]] + iDCT(Cr[1::])
    Cb = [Cb[0]] + iDCT(Cb[1::])

    # блоки
    Y = unblock(Y[1::], Y[0])
    Cr = unblock(Cr[1::], Cr[0])
    Cb = unblock(Cb[1::], Cb[0])

    # апсемплинг
    Cr = up_sampling(Cr)
    Cb = up_sampling(Cb)

    YCrCb = np.dstack((Y, Cr, Cb))

    RGB = YCbCr2RGB(YCrCb)

    img = Image.fromarray(RGB)
    return img


# pics = ["Lenna", "Big_picture", "Big_picture_grayscale",
#         "Big_picture_bw_no_dither", "Big_picture_bw_dithered"]

pics = ["Lenna", "Big_picture", "Big_picture_grayscale",
        "Big_picture_bw_no_dither", "Big_picture_bw_dithered"]
for pic in pics:
    for quality in range(100, 101, 20):
        compressed = jpeg_encoding(quality, f"test_data/{pic}.png")
        decompressed = jpeg_decode(quality)
        dithered_path = os.path.join("Results", f"{pic}_{quality}.png")
        decompressed.save(dithered_path)
        print(pic, quality)
