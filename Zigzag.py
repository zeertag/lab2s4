import numpy as np


def zigzag(M):
    transform = []
    for block in M:
        line = []
        N = block.shape[0]
        for d in range(2 * N - 1):
            for i in range(d + 1):
                j = d - i
                if i < N and j < N:
                    if d % 2 == 0:
                        line.append(int(block[j, i]))
                    else:
                        line.append(int(block[i, j]))
        transform.append(line)
    return transform


def back_zigzag(M):
    blocks = []
    for line in M:
        N = int(len(line) ** 0.5)
        block = np.zeros((N, N))
        idx = 0
        for d in range(2 * N - 1):
            for i in range(d + 1):
                j = d - i
                if i < N and j < N:
                    if d % 2 == 0:
                        block[j, i] = line[idx]
                    else:
                        block[i, j] = line[idx]
                    idx += 1
        blocks.append(block)
    return blocks
