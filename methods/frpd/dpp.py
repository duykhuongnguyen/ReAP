import time
import numpy as np
import torch
import dppy
import math
import matplotlib.pyplot as plt
import networkx as nx

from libs.frpd.quad import line_search
from libs.frpd.flow import opt_flow
from utils.funcs import shortest_path
 
 
def map_inference_dpp_greedy(kernel_matrix, max_length, epsilon=1E-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        # start_x_time = time.time()
        elements = kernel_matrix[selected_item, :]
        # print("elements: ", time.time() - start_x_time)
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        # if di2s[selected_item] < epsilon:
            # break
        selected_items.append(selected_item)
 
    S = np.sort(np.array(selected_items))
    # print(S)
    # print(np.linalg.det(kernel_matrix[S.reshape(-1, 1), S.reshape(1, -1)]), kernel_matrix[S.reshape(-1, 1), S.reshape(1, -1)])
    # det_L = np.linalg.det(kernel_matrix + np.identity(item_size))
    return S, np.linalg.det(kernel_matrix[S.reshape(-1, 1), S.reshape(1, -1)]) 
 

def map_inference_dpp_sw(kernel_matrix, window_size, max_length, epsilon=1E-10):
    """
    Sliding window version of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param window_size: positive int
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    v = np.zeros((max_length, max_length))
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    window_left_index = 0
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[window_left_index:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        v[k, window_left_index:k] = ci_optimal
        v[k, k] = di_optimal
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[window_left_index:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        if len(selected_items) >= window_size:
            window_left_index += 1
            for ind in range(window_left_index, k + 1):
                t = math.sqrt(v[ind, ind] ** 2 + v[ind, window_left_index - 1] ** 2)
                c = t / v[ind, ind]
                s = v[ind, window_left_index - 1] / v[ind, ind]
                v[ind, ind] = t
                v[ind + 1:k + 1, ind] += s * v[ind + 1:k + 1, window_left_index - 1]
                v[ind + 1:k + 1, ind] /= c
                v[ind + 1:k + 1, window_left_index - 1] *= c
                v[ind + 1:k + 1, window_left_index - 1] -= s * v[ind + 1:k + 1, ind]
                cis[ind, :] += s * cis[window_left_index - 1, :]
                cis[ind, :] /= c
                cis[window_left_index - 1, :] *= c
                cis[window_left_index - 1, :] -= s * cis[ind, :]
            di2s += np.square(cis[window_left_index - 1, :])
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


def map_inference_dpp_local_search(L, k, verbose=False):
    start_time = time.time()
    greedy_sol, greedy_prob = map_inference_dpp_greedy(L, k)
    greedy_time = time.time() - start_time

    cur_sol = greedy_sol.copy()
    cur_prob = greedy_prob

    N = L.shape[0]
    all_idx = np.array(range(N))
    ns_idx = np.setdiff1d(all_idx, cur_sol)

    it = 0
    while True:
        best_neighbors = None
        best_neighbors_prob = cur_prob
        si, sj = -1, -1
        localopt = True

        for i in range(len(cur_sol)):
            for v in ns_idx:
                neighbor = cur_sol.copy()
                neighbor[i] = v
                L_S = L[neighbor.reshape(-1, 1), neighbor.reshape(1, -1)]
                prob = np.linalg.det(L_S)
                if prob > best_neighbors_prob:
                    best_neighbors = neighbor
                    si, sj = cur_sol[i], v
                    best_neighbors_prob = prob
                    localopt = False

        if verbose:
            print("Iter ", it)
            print("Best neighbor: ({}, {})".format(si, sj))
            print("Best neighbor prob: ", best_neighbors_prob)

        if not localopt:
            cur_sol = best_neighbors
            cur_prob = best_neighbors_prob
            ns_idx = np.setdiff1d(all_idx, cur_sol)
        else:
            break
        it += 1

    ls_time = time.time() - start_time
    return cur_sol, cur_prob, ls_time, greedy_sol, greedy_prob, greedy_time


def map_inference_dpp_local_search_2(L, k, verbose=False):
    start_time = time.time()
    greedy_sol, greedy_prob = map_inference_dpp_greedy(L, k)
    greedy_time = time.time() - start_time

    if verbose:
        print("Prob: ", greedy_prob)

    cur_sol = greedy_sol.copy()
    cur_prob = greedy_prob
    obj_greedy = greedy_prob

    N = L.shape[0]
    all_idx = np.array(range(N))
    ns_idx = np.setdiff1d(all_idx, cur_sol)

    # L = np.arange(100).reshape(10, 10)
    L_S = L[cur_sol[:, np.newaxis], cur_sol]
    it = 0

    while True:
        start_iter_time = time.time()

        idx = np.array(range(len(cur_sol)))
        best_removal_idx = 0
        best_removal_prob = 0

        for i in range(len(cur_sol)):
            # cur_sol[i], cur_sol[-1] = cur_sol[-1], cur_sol[i]
            idx[i], idx[-1] = idx[-1], idx[i]
            L_Se = L_S[idx[:-1, np.newaxis], idx[:-1]]
            prob = np.linalg.det(L_Se)

            if prob > best_removal_prob:
                best_removal_idx = i
                best_removal_prob = prob
        obj_loc = best_removal_prob

        brid = best_removal_idx
        br = cur_sol[brid]

        best_neighbors = cur_sol.copy()
        best_add = -1
        best_neighbors_prob = cur_prob
        localopt = True

        for v in ns_idx:
            cur_sol[brid] = v
            L_S[brid, :] = L[v, cur_sol]
            L_S[:, brid] = L[cur_sol, v]
            prob = np.linalg.det(L_S)

            if prob > best_neighbors_prob:
                best_neighbors_prob = prob
                best_add = v
                localopt = False

        if verbose:
            print("Iter {}:".format(it))
            print("remove item: ", br)
            print("add item: ", best_add)
            print("best_neighbors_prob: ", best_neighbors_prob)

        if not localopt:
            cur_sol[brid] = best_add
            cur_prob = best_neighbors_prob
            L_S[brid, :] = L[best_add, cur_sol]
            L_S[:, brid] = L[cur_sol, best_add]
            ns_idx = np.setdiff1d(all_idx, cur_sol)
        else:
            cur_sol = best_neighbors
            cur_prob = best_neighbors_prob
            break
        it += 1

    ls_time = time.time() - start_time
    return cur_sol, obj_loc, ls_time, greedy_sol, greedy_prob, greedy_time
    # return cur_sol, cur_prob, ls_time, greedy_sol, greedy_prob, greedy_time


def map_inference_dpp_brute_force(L, k):
    start_time = time.time()
    N = L.shape[0]
    # I = np.identity(N)
    best_prob = 0
    best_s = np.array([])

    for subset in itertools.combinations(list(range(N)), k):
        S = np.array(subset, dtype=int).reshape(-1, 1)

        L_S = L[S, S.T]
        P_Ls = np.linalg.det(L_S)
        if best_prob < P_Ls:
            best_prob = P_Ls
            best_s = S

    return best_s.reshape(-1), best_prob, time.time() - start_time
 

def generate_recourse(x0, model, random_state, params=dict()):
    data = params['train_data']
    labels = params['labels']
    k = params['k']
    X = data[labels == 1]

    theta = params['frpd_params']['theta']
    kernel_width = params['frpd_params']['kernel']    
    interpolate = params['frpd_params']['interpolate']
    greedy = params['frpd_params']['greedy']
    n_neighbors = params['frpd_params']['n_neighbors']
    tau = params['frpd_params']['tau']
    graph_preprocess = params['graph_pre']
    graph_elem = params['graph']

    if graph_preprocess: 
        candidate_counterfactuals_star = []
        data_ = graph_elem["data"]
        labels_ = graph_elem["labels"]

        y_positive_indeces = np.where(labels_ == 1)

        adj_matrix = graph_elem['adj']
        weighted_adj_matrix = graph_elem['weighted_adj']
        G = nx.from_numpy_matrix(weighted_adj_matrix)

        distances, min_distance = shortest_path(adj_matrix, 0)
        distances_w, min_distance_w = shortest_path(weighted_adj_matrix, 0)

        candidate_min_distances = [min_distance + i for i in range(20)]
        min_distance_indeces = np.array([0])                                                
        for min_dist in candidate_min_distances:
            min_distance_indeces = np.c_[                                                           min_distance_indeces, np.array(np.where(distances == min_dist))               ]
        min_distance_indeces = np.delete(min_distance_indeces, 0)                           
        indeces_counterfactuals = np.intersect1d(
            np.array(y_positive_indeces), np.array(min_distance_indeces)
        )

        for i in range(indeces_counterfactuals.shape[0]):
            candidate_counterfactuals_star.append(indeces_counterfactuals[i])

        map_idx = {}
        for i, idx in enumerate(candidate_counterfactuals_star):
            map_idx[i] = idx
        # quad =  Solver(model, data_[candidate_counterfactuals_star], labels_[candidate_counterfactuals_star], theta, kernel_width)                                                    
        X = data_[candidate_counterfactuals_star]
        A = (X - x0).T / np.linalg.norm(X - x0, axis=1)                  
        S = A.T @ A                                                      
        d = np.linalg.norm(X - x0, axis=1)                               
        D = np.exp(-d ** 2 / kernel_width ** 2) * np.identity(d.shape[0])
        L = theta * S + (1 - theta) * D                                  
                                                                         
        res = map_inference_dpp_local_search_2(L, k)                     
        idx = res[3] if greedy else res[0]                               
        plans = X[idx, :]                                            
                                                                                            
        min_dist, paths = [], []
        for i in range(len(idx)):
            min_dist.append(map_idx[idx[i]])
            paths.append(nx.shortest_path(G, source=0, target=map_idx[idx[i]]))
        report = dict(feasible=True)                                                        
        return plans, distances_w[min_dist], paths, report

    report = dict(feasible=True)

    A = (X - x0).T / np.linalg.norm(X - x0, axis=1)
    S = A.T @ A
    d = np.linalg.norm(X - x0, axis=1)
    D = np.exp(-d ** 2 / kernel_width ** 2) * np.identity(d.shape[0])
    L = theta * S + (1 - theta) * D

    res = map_inference_dpp_local_search_2(L, k)
    idx = res[3] if greedy else res[0]
    X_diverse = X[idx, :]

    if interpolate == 'linear':
        recourse_set = []
        for i in range(k):
            best_x_b = line_search(model, x0, X_diverse[i], x0, p=2)
            recourse_set.append(best_x_b)
    
    elif interpolate == 'flow':
        recourse_set = opt_flow(x0, data, model, k, n_neighbors, tau, X_diverse)[0]

    plans = np.array(recourse_set)

    return plans, report


def dpp_recourse(x0, X, M, gamma=0.5, sigma=2.):
    """dpp recourse.
        map inference for a dpp kernel
            L = gamma * S + (1 - gamma) * exp(-d**2/sigma**2)
 
    Parameters
    ----------
    x0 :
        x0
    X : 
        positive samples
    M: int
        number of items
    gamma :
        weight 
    sigma :
        kernel width
    """
    A = (X - x0).T / np.linalg.norm(X - x0, axis=1)
    S = A.T @ A
    d = np.linalg.norm(X - x0, axis=1)
    D = np.exp(- d**2/sigma**2) * np.identity(d.shape[0])
    L = gamma * S + (1 - gamma) * D
 
    # selected_items = map_inference_dpp_sw(L, 1, M)
    cur_sol, cur_prob, ls_time, greedy_sol, greedy_prob, greedy_time = map_inference_dpp_local_search_2(L, M, verbose=False)
    # print(cur_sol, cur_prob, ls_time, greedy_sol, greedy_prob, greedy_time)
    # cur_sol, cur_prob, ls_time, greedy_sol, greedy_prob, greedy_time = map_inference_dpp_local_search_2(L, M, verbose=False)

    return cur_sol, cur_prob, ls_time, greedy_sol, greedy_prob, greedy_time
 
 
if __name__ == '__main__':
    d = 2
    M = 100
 
    np.random.seed(42)
    np.set_printoptions(suppress=True)
    x0 = np.array([.1, .2])
    X = np.random.randn(M, d)
 
    print("X = ", X)
    gamma = 1
    sigma = 10.
    slt_idx = dpp_recourse(x0, X, 4, gamma=gamma, sigma=sigma)
    mask = np.zeros(X.shape[0], dtype=bool)
    mask[slt_idx] = True
 
    print(mask)
    fig, ax = plt.subplots()
 
    selected_points = X[mask, :]
    print(~mask)
    nonselected_points = X[~mask, :]
    ax.scatter(selected_points[:, 0], selected_points[:, 1], marker='o', color='red')
    ax.scatter(nonselected_points[:, 0], nonselected_points[:, 1], marker='o', color='green')
    ax.scatter(x0[0], x0[1], marker='*', color='blue')
 
    for e in selected_points:
        plt.plot([x0[0], e[0]], [x0[1], e[1]])
 
    ax.set_title(f"$\\gamma = {gamma}, \\sigma = {sigma}$")
    plt.show()
