import numpy as np
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path
import networkx as nx


def mahalanobis_dist(x, y, A):
    return np.sqrt((x - y).T @ A @ (x - y))


def build_graph(data, A_opt, is_knn, n):
    def dist(x, y):
        return np.sqrt((x - y).T @ A_opt @ (x - y))

    if is_knn:
        nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree', metric=dist).fit(data)
        graph = nbrs.kneighbors_graph(data, mode="distance").toarray()
    else:
        graph = radius_neighbors_graph(data, radius=n, metric="pyfunc", func=dist, n_jobs=-1)

    return graph


def shortest_path_graph(adj, idx):
    G = nx.from_numpy_array(adj)
    dist_l, idx_l = [], []
    for i in idx:
        if nx.has_path(G, 0, i):
            dist_l.append(nx.shortest_path_length(G, 0, i, weight="weight"))
            idx_l.append(i)
    
    dist, min_idx = np.min(dist_l), idx_l[np.argmin(dist_l)]

    return dist, min_idx, nx.shortest_path(G, 0, min_idx, weight="weight")


def eval_cost(A, data, path):
    l = len(path)
    res = 0
    for i in range(l - 1):
        cost = np.sqrt((data[path[i + 1]] - data[path[i]]).T @ A @ (data[path[i + 1]] - data[path[i]]))
        res += cost

    return res
   

if __name__ == '__main__':
    data = np.random.rand(100, 2)
    A = np.random.rand(2, 2)
    A = A @ A.T

    graph = build_graph(data, A, True, 15)
    print(shortest_path_graph(graph))
