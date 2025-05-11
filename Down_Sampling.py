import numpy as np


def down_sampling(channel):
    original_shape = channel.shape
    h, w = original_shape

    if h % 2 != 0:
        channel = np.vstack((channel, channel[-1:, :]))
    if w % 2 != 0:
        channel = np.hstack((channel, channel[:, -1:]))

    down_sampled = np.zeros((channel.shape[0] // 2, channel.shape[1] // 2), dtype=np.uint8)

    for i in range(0, channel.shape[0], 2):
        for j in range(0, channel.shape[1], 2):
            block = channel[i:i + 2, j:j + 2]
            down_sampled[i // 2, j // 2] = np.mean(block)

    return down_sampled, original_shape


def up_sampling(chanel, original_shape):
    h, w = chanel.shape
    up_sampled = np.zeros((h * 2, w * 2), dtype=chanel.dtype)

    for i in range(h):
        for j in range(w):
            up_sampled[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = chanel[i, j]

    return up_sampled[:original_shape[0], :original_shape[1]]
