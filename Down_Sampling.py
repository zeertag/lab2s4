import numpy as np


def down_sampling(chanel, f=2):
    h, w = chanel.shape
    down_sampled = np.zeros((h // f, w // f))

    for i in range(0, h, f):
        for j in range(0, w, f):
            block = chanel[i:i + f, j:j + f]
            down_sampled[i // f, j // f] = np.mean(block)

    return down_sampled.astype(np.uint8)


def up_sampling(chanel, f=2):
    h, w = chanel.shape
    up_sampled = np.zeros((h * f, w * f), dtype=chanel.dtype)

    for i in range(h):
        for j in range(w):
            up_sampled[i * f:(i + 1) * f, j * f:(j + 1) * f] = chanel[i, j]

    return up_sampled
