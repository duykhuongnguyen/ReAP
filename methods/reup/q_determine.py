import numpy as np

from methods.reup.chebysev import chebysev_center, sdp_cost


def compute_M(x_0, x_i, x_j):
    M = np.outer(x_i, x_i) - np.outer(x_j, x_j) + np.outer(x_j - x_i, x_0) + np.outer(x_0, x_j - x_i)
    return M


def exhaustive_search(A_opt, x_0, data):
    cur = np.inf

    for i in range(len(data) - 1):
        for j in range(i + 1, len(data)):
            M_ij = compute_M(x_0, data[i], data[j])
            obj = np.abs(np.sum(np.multiply(A_opt, M_ij))) / (np.sum(np.multiply(M_ij, M_ij)))
            if obj < cur:
                cur = obj
                res = M_ij

    return res


def find_q(x_0, data, T, epsilon):
    """Find the set of constraints after T questions

    Parameters:
        x_0: input instance
        data: training data
        T: number of questions
        epsilon: parameter

    Returns:
    """
    d = x_0.shape[0]
    P = []

    for i in range(T):
        init = True if i == 0 else False
        radius, A_opt = chebysev_center(d, P, epsilon, init)
        M_ij = exhaustive_search(A_opt, x_0, data)
        P.append(M_ij)

    return P


if __name__ == '__main__':
    A = np.random.rand(2, 2)
    x_0 = np.array([1, 1])
    x_init = np.array([0, 0])
    data = np.random.rand(100, 2)

    P = find_q(x_0, data, 5, epsilon=1e-3)
    print(P)

    sdp = sdp_cost(x_init, x_0, P, 1e-3)
    print(sdp)
