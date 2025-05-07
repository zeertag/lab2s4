import numpy as np


def create_dct_matrix(N):
    matrix = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            factor = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
            matrix[k, n] = factor * np.cos((np.pi * (2 * n + 1) * k) / (2 * N))
    return matrix


def DCT(data):
    transformed = []
    dct_mat = create_dct_matrix(data[0].shape[0])
    for block in data:
        dct_block = dct_mat @ block @ dct_mat.T
        transformed.append(dct_block)
    return transformed


def iDCT(data):
    transformed = []
    dct_mat = create_dct_matrix(data[0].shape[0])
    for block in data:
        dct_block = dct_mat.T @ block @ dct_mat
        transformed.append(dct_block)
    return transformed