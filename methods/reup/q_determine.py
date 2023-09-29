import numpy as np
import scipy
import heapq

import torch

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
                res = M_ij, (data[i], data[j])
    
    return res


def exhaustive_search_k(A_opt, x_0, data, k):
    obj_l = []
    for i in range(len(data) - 1):
        for j in range(i + 1, len(data)):
            M_ij = compute_M(x_0, data[i], data[j])
            obj = np.abs(np.sum(np.multiply(A_opt, M_ij))) / np.linalg.norm(M_ij)
            obj_l.append(obj)

    obj_l.sort()
    print(obj_l[:k])


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
        d_obj[obj] = (d_list[i], d_list[i + 1])

    d_obj_sorted = dict(sorted(d_obj.items()))
    for value in d_obj_sorted:
        if d_obj_sorted[value] not in prev:
            M_ij = compute_M(x_0, data[d_obj_sorted[value][0]], data[d_obj_sorted[value][1]])
            obj = np.sum(np.multiply(A_0, M_ij))
            res = M_ij if obj <= epsilon else -M_ij
            return res, d_obj_sorted[value], (data[d_obj_sorted[value][0]], data[d_obj_sorted[value][1]])


def similar_cost_heuristics_k(A_opt, x_0, data, A_0, epsilon, prev, k=3):
    l = data.shape[0]
    s = np.zeros(l)
    d = {}
    for i in range(l):
        s[i] = (data[i] - x_0).T @ A_opt @ (data[i] - x_0)
        d[i] = s[i]

    d_sorted = dict(sorted(d.items(), key=lambda item: item[1]))
    d_list = list(d_sorted.keys())

    cur_max_l  = []
    for i in range(l - k):
        cur_sum_max, cur_max = -np.inf, (np.inf, i, [i + m for m in range(k)])
        for j in range(k):
            cur, cur_sum = i + j, 0
            for l in range(k):
                if i + l != cur:
                    M_ij = compute_M(x_0, data[d_list[cur]], data[d_list[i + l]])
                    obj = np.abs(np.sum(np.multiply(A_opt, M_ij))) / np.linalg.norm(M_ij)
                    cur_sum += obj
                
            if cur_sum > cur_sum_max:
                cur_sum_max = cur_sum
                cur_max = (cur_sum, i + j, [i + m for m in range(k)])
        cur_max_l.append(cur_max)

    min_l = sorted(cur_max_l)
    for i in range(len(min_l)):
        cur, cur_l = min_l[i][1], min_l[i][2]
        if cur_l not in prev:
            cost_l = [(data[d_list[i]] - x_0).T @ A_0 @ (data[d_list[i]] - x_0) for i in cur_l]
            min_idx = cur_l[np.argmin(cost_l)]
            M_ij_l = []
            for i in cur_l:
                if i != min_idx:
                    M_ij = compute_M(x_0, data[d_list[min_idx]], data[d_list[i]])
                    obj = np.sum(np.multiply(A_0, M_ij))
                    res = M_ij if obj <= epsilon else -M_ij
                    M_ij_l.append(res)

            return M_ij_l, cur_l


def similar_cost_heuristics_kpairs(A_opt, x_0, data, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        l = data.shape[0]
        s = np.zeros(l)
        d = {}
        for i in range(l):
            s[i] = (data[i] - x_0).T @ A_opt @ (data[i] - x_0)
            d[i] = s[i]
        
        d_sorted = dict(sorted(d.items(), key=lambda item: item[1]))
        d_list = list(d_sorted.keys())

        s = np.sort(s)
        heap = []
        for i in range(1, len(s) - 1):
            M_ij = compute_M(x_0, data[d_list[0]], data[d_list[i + 1]])
        
        for i in range(len(s) - 1):
            M_ij = compute_M(x_0, data[d_list[i]], data[d_list[i + 1]])
            obj = np.abs(np.sum(np.multiply(A_opt, M_ij))) / np.linalg.norm(M_ij)
            heap.append((obj, i, i + 1))

        heapq.heapify(heap)

        for _ in range(k):
            d, root, nei = heapq.heappop(heap)
            if nei + 1 < len(s):
                M_ij = compute_M(x_0, data[d_list[nei + 1]], data[d_list[root]])
                obj = np.abs(np.sum(np.multiply(A_opt, M_ij))) / np.linalg.norm(M_ij)
                heapq.heappush(heap, (obj, root, nei + 1))
        
        return d


def mean_rank(data, x_0, A_0, A_opt, top_k):
    l = data.shape[0]
    min_rank = sum([i for i in range(top_k)])
    max_rank = sum([l - 1 - i for i in range(top_k)])
    
    d_0 = {}
    # s = np.zeros(l)
    s = []
    for i in range(l):
        # s[i] = (data[i] - x_0).T @ A_0 @ (data[i] - x_0)
        s.append(((data[i] - x_0).T @ A_0 @ (data[i] - x_0), i))
        d_0[s[i]] = i

    d_rank = {}
    # s = np.sort(s)
    s.sort()
    for i in range(l):
        d_rank[s[i][1]] = i

    d_opt = {}
    for i in range(l):
        d_opt[i] = (data[i] - x_0).T @ A_opt @ (data[i] - x_0)
    d_opt = dict(sorted(d_opt.items(), key=lambda item: item[1]))

    keys = list(d_opt.keys())
    rank = 0
    
    for i in keys[:top_k]:
        rank += d_rank[i]
    
    return (rank - min_rank) / max_rank


def question_correction_gd(A_opt, A_0, x_0, x_i_opt, x_j_opt, alpha, max_iter, epsilon):
    A_opt = torch.tensor(A_opt, requires_grad=False)
    A_0 = torch.tensor(A_0, requires_grad=False)
    x_0 = torch.tensor(x_0, requires_grad=False)
    x_i = torch.tensor(x_i_opt, requires_grad=True)
    x_j = torch.tensor(x_j_opt, requires_grad=True)

    optimizer = torch.optim.SGD([x_i, x_j], lr=0.01, momentum=0.9)
    
    min_ = np.inf
    for i in range(max_iter):
        # Compute objective
        M_ij = torch.outer(x_i, x_i) - torch.outer(x_j, x_j) + torch.outer(x_j - x_i, x_0) + torch.outer(x_0, x_j - x_i)
        obj = (torch.sum((torch.multiply(M_ij, A_opt))) / torch.norm(M_ij)) ** 2
        if i == 0:
            print(torch.sqrt(obj))

        # Optimizer
        obj.backward()
        optimizer.step()

        if obj < min_:
            min_ = obj
            M_opt = M_ij
            pair = (x_i, x_j)
    
    obj = torch.sum(torch.multiply(A_0, M_ij))
    M_ij = M_ij if obj <= epsilon else -M_ij
    # M_ij = M_ij / torch.norm(M_ij)
    
    return M_ij.detach().numpy()
         

def question_correction_gd_k(A_opt, A_0, x_0, x_i_opt, x_j_opt, alpha, max_iter, epsilon):
    A_opt = torch.tensor(A_opt, requires_grad=False)
    A_0 = torch.tensor(A_0, requires_grad=False)
    x_0 = torch.tensor(x_0, requires_grad=False)
    x_i = torch.tensor(x_i_opt, requires_grad=True)
    x_j = torch.tensor(x_j_opt, requires_grad=True)

    optimizer = torch.optim.SGD([x_i, x_j], lr=0.01,
momentum=0.9)

    min_ = np.inf
    for i in range(max_iter):
        # Compute objective
        M_ij = torch.outer(x_i, x_i) - torch.outer(x_j, x_j) + torch.outer(x_j - x_i, x_0) + torch.outer(
_0, x_j - x_i)
        obj = (torch.sum((torch.multiply(M_ij, A_opt))) / torch.norm(M_ij)) ** 2

        # Optimizer
        obj.backward()
        optimizer.step()

        if obj < min_:
            min_ = obj
            M_opt = M_ij
            pair = (x_i, x_j)
    
    obj = torch.sum(torch.multiply(A_0, M_ij))
    M_ij = M_ij if obj <= epsilon else -M_ij
    # M_ij = M_ij / torch.norm(M_ij)

    return M_ij.detach().numpy()


def find_q(x_0, data, T, A_0, epsilon, cost_correction):
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
    P, prev, rank_l = [], [], []
    A_opt_l = []

    for i in range(T):
        # Solve chebysev
        init = True if i == 0 else False
        radius, A_opt = chebysev_center(d, P, epsilon, init)
        A_opt_l.append(A_opt)
        rank = mean_rank(data, x_0, A_0, A_opt, 5)
        rank_l.append(rank)

        # 1 question
        M_ij, pair, pair_data = similar_cost_heuristics(A_opt, x_0, data, A_0, epsilon, prev)
        prev.append(pair)
        P.append(M_ij)
        
        # k questions
        #M_ij_l, pairs = similar_cost_heuristics_k(A_opt, x_0, data, A_0, epsilon, prev, k=4)
        #prev.append(pairs)
        #P += M_ij_l
    
    if T != 0:
        radius, A_opt = chebysev_center(d, P, epsilon, False)
    else:
        A_opt = 0.5 * np.identity(d)

    return P, A_opt, rank_l


if __name__ == '__main__':
    A = np.random.rand(2, 2)
    A = A @ A.T
    x_0 = np.random.rand(2)
    x_init = np.array([0, 0])
    data = np.random.rand(100, 2)

    P = find_q(x_0, data, 10, A, epsilon=1e-3, cost_correction=True)
