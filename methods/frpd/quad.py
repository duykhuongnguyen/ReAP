import time
import math
import numpy as np
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors
import networkx as nx

import gurobipy as grb

from libs.frpd.flow import opt_flow
from utils.funcs import shortest_path


class Solver(object):
    """ Class for optimization problem """

    def __init__(self, model, data, labels, theta=0.5, kernel_width=1):
        self.model = model
        self.data = data
        self.labels = labels
        self.theta = theta
        self.h = kernel_width

        self.data_hat = self.data[self.labels == 1]
        self.N, self.dim = self.data_hat.shape

    def quad_greedy(self, kernel_matrix, max_length, epsilon=1E-10):
        item_size = kernel_matrix.shape[0]
        cis = np.zeros((max_length, item_size))
        di2s = np.copy(np.diag(kernel_matrix))
        selected_items = list()
        selected_item = np.argmin(di2s)
        selected_items.append(selected_item)
        while len(selected_items) < max_length:
            di2s = np.array([np.inf for j in range(kernel_matrix.shape[0])])
            for i in range(kernel_matrix.shape[0]):
                selected_ = np.array(selected_items + [i])
                di2s[i] = np.sum(kernel_matrix[selected_.reshape(-1, 1), selected_.reshape(1, -1)])
            # k = len(selected_items) - 1
            # ci_optimal = cis[:k, selected_item]
            # di_optimal = np.sqrt(di2s[selected_item])
            # elements = kernel_matrix[selected_item, :]
            # eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
            # cis[k, :] = eis
            # di2s -= np.square(eis)
            # di2s[selected_item] = np.inf
            # selected_item = np.argmin(di2s)
            # if di2s[selected_item] < epsilon:
                # break
            selected_item = np.argmin(di2s)
            selected_items.append(selected_item)

        S = np.sort(np.array(selected_items))
        # print(S)
        # print(np.linalg.det(kernel_matrix[S.reshape(-1, 1), S.reshape(1, -1)]), kernel_matrix[S.reshape(-1, 1), S.reshape(1, -1)])
        # det_L = np.linalg.det(kernel_matrix + np.identity(item_size))
        return S, np.sum(kernel_matrix[S.reshape(-1, 1), S.reshape(1, -1)])

    def quad(self, S, d, k, cost_diverse=True):
        """ Solve quadratic program with gurobi
        """
        dim = len(d)

        # Model initialization
        model = grb.Model("qcp")
        model.params.NonConvex = 2
        model.setParam('OutputFlag', False)
        model.params.threads = 64

        # Variables
        z = model.addMVar(dim, vtype=grb.GRB.BINARY, name="z")
        z_sub = model.addMVar(1, vtype=grb.GRB.CONTINUOUS, name="zsub")

        # Set objectives
        if cost_diverse:
            obj = (1 - self.theta) * d @ z + self.theta * z_sub
        else:
            obj = z_sub
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Constraints
        if cost_diverse:
            model.addConstr(z.sum() == k)
        else:
            for i in range(0, k * k, k):
                model.addConstr(z[i:i+k].sum() == 1)
        model.addConstr(z_sub == z @ S @ z)

        # Optimize
        model.optimize()

        z_opt = np.zeros(dim)

        for i in range(dim):
            z_opt[i] = z[i].x

        return z_opt

    def compute_matrix(self, x0, data):
        N, dim = data.shape

        A = np.zeros((dim, N))
        d = np.zeros(N)

        A = (data - x0).T / np.linalg.norm(data - x0, axis=1)
        d = np.linalg.norm(data - x0, axis=1)
        S = np.dot(A.T, A)

        return A, S, d

    def find_eig(self, matrix):
        w, v = eigh(matrix)
        sum_eig = sum(w ** 2)
        cur_sum = sum_eig

        for i in range(len(w) - 1, -1, -1):
            cur_sum -= w[i] ** 2
            if cur_sum / sum_eig < 1e-9:
                return np.flip(w[i:len(w)]), np.flip(v[:, i:len(w)], axis=1)

    def best_response(self, w, v, d, k, max_iter=100, period=80):
        z = np.zeros(self.N)
        z_p = np.zeros((period, self.N))

        m_dim = len(w)
        gamma = np.zeros(m_dim)

        for i in range(max_iter):
            gamma = -self.theta * (np.multiply(w, np.dot(v.T, z)))
            # for j in range(m_dim):
                # gamma[j] = -(self.theta * w[j] * np.dot(v[:, j], z))

            gamma_identity = (1 - self.theta) * d - 2 * np.dot(v, gamma)
            idx = (gamma_identity).argsort()[:k]
            z = np.zeros(self.N)
            z[idx] = 1

            if i > max_iter - period:
                z_p[i - max_iter + period, :] = z

        return z_p

    def dp(self, w, v, d, k, step_size=0.1, max_iter=100, period=80):
        z = np.zeros(self.N)
        z_p = np.zeros((period, self.N))

        m_dim = len(w)
        gamma = np.zeros(m_dim)

        for i in range(max_iter):
            kappa = step_size / np.sqrt(i + 1)

            gamma_add = np.zeros(m_dim)
            for j in range(m_dim):
                gamma_add[j] = kappa * ((-2 * gamma[j]) / (self.theta * w[j]) - 2 * np.dot(v[:, j], z))

            gamma += gamma_add
            gamma_identity = (1 - self.theta) * d - 2 * np.dot(v, gamma)
            idx = (gamma_identity).argsort()[:k]
            z = np.zeros(self.N)
            z[idx] = 1

            if i > max_iter - period:
                z_p[i - max_iter + period, :] = z

        return z_p
    
    def solve(self, x0, k, period=20, best_response=True, return_time=False, return_obj=False):
        A, S, d = self.compute_matrix(x0, self.data[self.labels == 1])
        L = self.theta * S + (1 - self.theta) * (d * np.identity(d.shape[0]))
        S_ = np.copy(S)
        w, v = self.find_eig(S)

        # Best response and dp
        start_time = time.time()
        if best_response:
            z_p = self.best_response(w, v, d, k=k, period=period)
        else:
            z_p = self.dp(w, v, d, k=k, step_size=1, period=period)
        
        z_prev = np.zeros(len(d))
        for i in range(period):
            z_prev = np.logical_or(z_prev, z_p[i, :])
        
        t = time.time() - start_time

        idx_l = np.where(z_prev == 1)[0]

        data = self.data[self.labels == 1][np.where(z_prev == 1)[0]]
        A, S, d = self.compute_matrix(x0, data)
        z = self.quad(S, d, k)

        idx = idx_l[(-z).argsort()[:k]]
        X_diverse = self.data[self.labels == 1][idx]
        X_other = self.data[self.labels == 1][np.where(z_prev == 0)[0]]

        if return_time: 
            return t
        if return_obj:
            sum_obj = np.sum(L[idx.reshape(-1, 1), idx.reshape(1, -1)])
            return sum_obj

        return idx, X_diverse, X_other

    def generate_recourse(self, x0, k, period=20, best_response=True, interpolate='linear', n_neighbors=5, tau=0.5):
        idx, X_diverse, X_other = self.solve(x0, k, period, best_response)

        if interpolate == 'linear':
            recourse_set = []

            for i in range(X_diverse.shape[0]):
                best_x_b = line_search(self.model, x0, X_diverse[i], x0, p=2)  
                recourse_set.append(best_x_b)
 
        elif interpolate == 'flow':
            recourse_set = opt_flow(x0, self.data, self.model, k, n_neighbors, tau, X_diverse)[0]

        elif interpolate == 'graph':
            recourse_set = X_diverse

        recourse_set = np.array(recourse_set)

        if interpolate == 'graph':
            return recourse_set, idx

        return recourse_set, X_diverse, X_other


def line_search(model, x0, x1, x2, p=2):
    best_x_b = None
    best_dist = np.inf

    lambd_list = np.linspace(0, 1, 100)
    for lambd in lambd_list:
        x_b = (1 - lambd) * x1 + lambd * x2
        label = model.predict(x_b)
        if label == 1:
            dist = np.linalg.norm(x0 - x_b, ord=p)
            if dist < best_dist:
                best_x_b = x_b
                best_dist = dist
    
    return best_x_b


def generate_recourse(x0, model, random_state, params=dict()):
    data = params['train_data']
    labels = params['labels']
    X = data[labels == 1]
    k = params['k']

    theta = params['frpd_params']['theta']
    kernel_width = params['frpd_params']['kernel']
    period = params['frpd_params']['period']
    best_response = params['frpd_params']['response']
    interpolation = params['frpd_params']['interpolate']
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
            min_distance_indeces = np.c_[
                min_distance_indeces, np.array(np.where(distances == min_dist))
            ]
        min_distance_indeces = np.delete(min_distance_indeces, 0)
        indeces_counterfactuals = np.intersect1d(
            np.array(y_positive_indeces), np.array(min_distance_indeces)
        )
        for i in range(indeces_counterfactuals.shape[0]):
            candidate_counterfactuals_star.append(indeces_counterfactuals[i])
        
        map_idx = {}
        for i, idx in enumerate(candidate_counterfactuals_star):
            map_idx[i] = idx
        quad =  Solver(model, data_[candidate_counterfactuals_star], labels_[candidate_counterfactuals_star], theta, kernel_width)
        plans, idx = quad.generate_recourse(x0, k, period, best_response, interpolation, n_neighbors, tau)[:2]

        min_dist, paths = [], []
        for i in range(len(idx)):
            min_dist.append(map_idx[idx[i]])
            paths.append(nx.shortest_path(G, source=0, target=map_idx[idx[i]]))
        report = dict(feasible=True)
        return plans, distances_w[min_dist], paths, report

    quad =  Solver(model, data, labels, theta, kernel_width)
    plans = quad.generate_recourse(x0, k, period, best_response, interpolation, n_neighbors, tau)[0]
    report = dict(feasible=True)

    return plans, report


def quad_recourse(x0, k, model, data, labels, theta, kernel_width):
    quad =  Solver(model, data, labels, theta, kernel_width)
    sum_obj = quad.solve(x0, k, return_obj=True)

    return sum_obj


def quad_recourse_gurobi(x0, k, model, data, labels, theta, kernel_width):
    quad = Solver(model, data, labels, theta, kernel_width)
    A, S, d = quad.compute_matrix(x0, quad.data[quad.labels == 1])

    t = time.time()
    z = quad.quad(S, d, k)

    return time.time() - t

def quad_recourse_greedy(x0, k, model, data, labels, theta, kernel_width):  
    quad = Solver(model, data, labels, theta, kernel_width)
    A, S, d = quad.compute_matrix(x0, quad.data[quad.labels == 1])
    D = d * np.identity(d.shape[0])
    L = theta * S + (1 - theta) * D

    t = time.time()
    z, prob = quad.quad_greedy(L, k)

    return prob
