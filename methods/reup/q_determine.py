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
            obj = np.abs(np.sum(np.multiply(A_opt, M_ij))) / np.linalg.norm(M_ij)
            if obj < cur:
                cur = obj
                res = M_ij
    
    return res


def similar_cost_heuristics(A_opt, x_0, data, A_0, epsilon, prev):
    l = data.shape[0]
    s = np.zeros(l)
    d = {}
    for i in range(l):
        s[i] = (data[i] - x_0).T @ A_opt @ (data[i] - x_0)
        d[i] = s[i]
    
    d_sorted = dict(sorted(d.items(), key=lambda item: item[1]))
    d_list = list(d_sorted.keys())

    cur = np.inf
    d_obj = {}
    for i in range(l - 1):
        M_ij = compute_M(x_0, data[d_list[i]], data[d_list[i + 1]])
        obj = np.abs(np.sum(np.multiply(A_opt, M_ij))) / np.linalg.norm(M_ij) 
        # d_obj[(d_list[i], d_list[i + 1])] = obj
        d_obj[obj] = (d_list[i], d_list[i + 1])

    d_obj_sorted = dict(sorted(d_obj.items()))
    for value in d_obj_sorted:
        if d_obj_sorted[value] not in prev:
            M_ij = compute_M(x_0, data[d_obj_sorted[value][0]], data[d_obj_sorted[value][1]])
            obj = np.sum(np.multiply(A_0, M_ij))
            res = M_ij if obj <= epsilon else -M_ij
            return res, d_obj_sorted[value]
        # if obj < cur: 
        #     cur = obj                    
        #     res = M_ij
        #     obj = np.sum(np.multiply(A_0, M_ij))
        #     res = M_ij if obj <= epsilon else -M_ij
    
    # return res


def find_q(x_0, data, T, A_0, epsilon):
    """Find the set of constraints after T questions

    Parameters:
        x_0: input instance
        data: training data
        T: number of questions
        epsilon: parameter

    Returns:
        P: feasible set
    """
    d = x_0.shape[0]
    P = []
    prev = []

    for i in range(T):
        init = True if i == 0 else False
        radius, A_opt = chebysev_center(d, P, epsilon, init)
        print(radius)
        # M_ij = exhaustive_search(A_opt, x_0, data)
        M_ij, pair = similar_cost_heuristics(A_opt, x_0, data, A_0, epsilon, prev)
        prev.append(pair)
        prev.append((pair[1], pair[0]))
        P.append(M_ij)

    return P


if __name__ == '__main__':
    A = np.random.rand(2, 2)
    x_0 = np.array([1, 1])
    x_init = np.array([0, 0])
    data = np.random.rand(100, 2)

    P = find_q(x_0, data, 10, epsilon=1e-3)

    sdp = sdp_cost(x_init, x_0, P, 1e-3)
