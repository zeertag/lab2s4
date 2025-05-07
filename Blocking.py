import numpy as np


def blocks(M, N):
    x, y = M.shape
    if x % N != 0 or y % N != 0:
        new_x, new_y = x, y
        if x % N != 0:
            new_x = N * (x // N + 1)
        if y % N != 0:
            new_y = N * (y // N + 1)
        new_M = np.zeros((new_x, new_y), dtype=M.dtype)
        new_M[:x, :y] = M
        M = new_M
        x, y = M.shape

    blocked = [M.shape]
    for i in range(0, x, N):
        for j in range(0, y, N):
            block = M[i:i + N, j:j + N]
            blocked.append(block)

    return blocked


def unblock(blocked, original_shape):
    x, y = original_shape
    N = blocked[0].shape[0]  # размер блока (обычно 8)

    # Расчёт размера с паддингом (если он был)
    new_x = N * ((x + N - 1) // N)
    new_y = N * ((y + N - 1) // N)

    M_padded = np.zeros((new_x, new_y), dtype=blocked[0].dtype)

    idx = 0
    for i in range(0, new_x, N):
        for j in range(0, new_y, N):
            if idx >= len(blocked):
                break
            M_padded[i:i + N, j:j + N] = blocked[idx]
            idx += 1

    return M_padded[:x, :y]
