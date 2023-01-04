import numpy as np


def compute_M(x_0, x_i, x_j):
    M = np.outer(x_i, x_i) - np.outer(x_j, x_j) + np.outer(x_j - x_i, x_0) + np.outer(x_0, x_j - x_i) 
    return M


def exhaustive_search(A_opt, x_0, data):
    cur = np.inf

    for i in range(len(data) - 1):
        for j in range(i + 1, len(data)):
            M = compute_M(x_0, data[i], data[j])
            obj = np.abs(np.sum(np.multiply(A_opt, M))) / (np.sum(np.multiply(M, M)))
            if obj < cur:
                cur = obj
                res = (i, j)

    return res


if __name__ == '__main__':
    A = np.random.rand(2, 2)
    x_0 = np.array([1, 1])
    data = np.random.rand(100, 2)

    res = exhaustive_search(A, x_0, data)
    print(res)
