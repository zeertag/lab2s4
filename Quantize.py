import numpy as np


def quant_matrix(qmatrix, quality):
    if quality <= 0:
        quality = 0.01
    elif quality > 100:
        quality = 100

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - quality * 2

    scaled_matrix = ((qmatrix * scale + 50) / 100).astype(np.uint8)
    scaled_matrix[scaled_matrix == 0] = 1
    return scaled_matrix
