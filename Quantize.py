import numpy as np


def scale_quant_matrix(Q, quality):
    if quality < 1:
        quality = 1
    elif quality > 100:
        quality = 100

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - quality * 2

    scaled_matrix = ((Q * scale + 50) / 100).astype(np.uint8)
    scaled_matrix[scaled_matrix == 0] = 1
    return scaled_matrix
