def dif_dc(M):
    for i in range(len(M) - 1, 0, -1):
        if i > 0:
            M[i][0] = M[i][0] - M[i - 1][0]
    return M


def back_dc(M):
    for i in range(len(M)):
        if i > 0:
            M[i][0] = M[i][0] + M[i - 1][0]
    return M
