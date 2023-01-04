import numpy as np
import scipy as sp
from scipy.linalg import sqrtm, eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path
from sklearn.utils import check_random_state
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors, KernelDensity
from Levenshtein import distance as lev

import networkx as nx


def gelbrich_dist(mean_0, cov_0, mean_1, cov_1):
    t1 = np.linalg.norm(mean_0 - mean_1)
    t2 = np.trace(cov_0 + cov_1 - 2 *
                  sqrtm(sqrtm(cov_1) @ cov_0 @ sqrtm(cov_1)))
    return np.sqrt(t1 ** 2 + t2)


def check_symmetry(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def sqrtm_psd(A, check_finite=True):
    A = np.asarray(A)
    if len(A.shape) != 2:
        raise ValueError("Non-matrix input to matrix function.")
    w, v = eigh(A, check_finite=check_finite)
    w = np.maximum(w, 0)
    return (v * np.sqrt(w)).dot(v.conj().T)


def lp_dist(x, y, p=2):
    return np.linalg.norm(x - y, ord=p)


def l2_dist(x, y, p=2):
    return np.linalg.norm(x - y, ord=2)


def l1_dist(x, y, p=2):
    return np.linalg.norm(x - y, ord=1)


def uniform_ball(x, r, n, random_state=None):
    # muller method
    random_state = check_random_state(random_state)
    d = len(x)
    V_x = random_state.randn(n, d)
    V_x = V_x / np.linalg.norm(V_x, axis=1).reshape(-1, 1)
    V_x = V_x * (random_state.random(n) ** (1.0/d)).reshape(-1, 1)
    V_x = V_x * r + x
    return V_x


def normalize_exp(w, b):
    m = np.linalg.norm(np.hstack([w, b]), 2)
    return w/m, b/m


def compute_robustness(exps):
    w0, b0 = exps[0]
    w0, _ = normalize_exp(w0, b0)
    # e0 = np.hstack([w0, b0])
    ret = 0

    for w, b in exps[1:]:
        # e = np.hstack([w, b])
        w, _ = normalize_exp(w, b)
        ret = max(ret, np.linalg.norm(w0 - w, 2))
        # ret = max(ret, sp.spatial.distance.cosine(e, e0))

    return ret


def compute_fidelity_on_samples(exps, X, y):
    w, b = exps
    exp_pred = X @ w.T + b >= 0
    ret = np.sum(y == exp_pred) / len(y)
    return ret


def compute_fidelity(x, e, predict_fn, r_fid, num_samples=1000,
                     random_state=None, return_data=False):
    V_x = uniform_ball(x, r_fid, num_samples, random_state)
    y = np.argmax(predict_fn(V_x), axis=-1)
    w, b = e
    y_ls = V_x @ w.T + b >= 0
    ret = np.mean(y == y_ls)
    if return_data:
        return ret, V_x, y
    else:
        return ret


def compute_max_distance(x):
    max_dist = -np.inf
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            d = np.linalg.norm(x[i] - x[j], ord=2)
            max_dist = max(max_dist, d)

    return max_dist


def quadratic_divergence(covsa, cov):
    return np.trace((covsa - cov) @ (covsa - cov))


def bures_divergence(covsa, cov):
    covsa_sqrtm = sqrtm_psd(covsa)
    return np.trace(covsa + cov - 2 * sqrtm_psd(covsa_sqrtm @ cov @ covsa_sqrtm))


def fisher_rao_distance(covsa, cov):
    covsa_sqrtm_inv = np.linalg.inv(sqrtm_psd(covsa))
    return np.linalg.norm(sp.linalg.logm(covsa_sqrtm_inv @ cov @ covsa_sqrtm_inv))


distance_funcs = {
    "mpm": quadratic_divergence,
    "quad_rmpm": quadratic_divergence,
    "bw_rmpm": bures_divergence,
    "fr_rmpm": fisher_rao_distance,
    "logdet_rmpm": fisher_rao_distance,
    "l2": l2_dist,
}


def compute_max_shift(covsa, covs, metric='quad_rmpm'):
    max_dist = -np.inf
    for cov in covs:
        d = distance_funcs[metric](covsa, cov)
        max_dist = max(max_dist, d)
    return max_dist

def is_dominated(a, b):
    dominated = False
    for i in range(len(a)):
        if a[i] > b[i]:
            return False
        elif a[i] < b[i]:
            dominated = True    
    return dominated


def find_pareto(x, y, incline=True):
    a = list(zip(x, y))
    a = sorted(a, key=lambda x: (x[0], -x[1]))
    best = -1 if incline else 100
    pareto = []
    for e in a:
        if incline:
            if e[1] > best:
                pareto.append(e)
                best = e[1]
        else:
            if e[1] < best:
                pareto.append(e)
                best = e[1]

    return [e[0] for e in pareto], [e[1] for e in pareto]


def compute_validity(model, plans):
     out = model.predict(plans)
     return 1 if np.all(out == 1) else 0


def shortest_path(graph, index):
    """
    Uses dijkstras shortest path
    Parameters
    ----------
    graph: CSR matrix
    index: int
    Returns
    -------
    np.ndarray, float
    """
    distances = dijkstra(
        csgraph=graph, directed=False, indices=index, return_predecessors=False
    )
    distances[index] = np.inf  # avoid min. distance to be x^F itself
    min_distance = distances.min()
    return distances, min_distance


def compute_proximity(test_ins, plans, p=2):
    num_recourse, d = plans.shape
    ret = 0
    for i in range(num_recourse):
        ret += lp_dist(plans[i], test_ins, p)
    return ret / num_recourse


def compute_proximity_graph(adj, idx):
    graph = csr_matrix(adj)
    dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, indices=0, return_predecessors=True)
    # print(dist_matrix, idx, predecessors)
    return dist_matrix[idx]


def compute_proximity_graph_(adj, idx):
    G = nx.from_numpy_array(adj)
    for i in idx:
        print(nx.shortest_path_length(G, 0, i))
    return nx.shortest_path(G, 0, i)


def compute_diversity(cfs, dice_data, weights='inverse_mad', intercept_feature=True):
    num_cfs, d = cfs.shape

    if weights == 'inverse_mad':
        feature_weights_dict = {}
        normalized_mads = dice_data.get_valid_mads(normalized=True)
        for feature in normalized_mads:
            feature_weights_dict[feature] = round(
                1/normalized_mads[feature], 2)

        feature_weights = [1.0] if intercept_feature else []
        for feature in dice_data.ohe_encoded_feature_names:
            if feature in feature_weights_dict:
                feature_weights.append(feature_weights_dict[feature])
            else:
                feature_weights.append(1.0)
        feature_weights = np.array(feature_weights)

    elif isinstance(weights, np.ndarray):
        feature_weights = weights
    else:
        feature_weights = np.ones(d)

    ret = 0
    for i in range(num_cfs):
        for j in range(i+1, num_cfs):
            # ret += compute_dist(cfs[i], cfs[j], feature_weights)
            ret += lp_dist(cfs[i], cfs[j], 2)

    return ret / (num_cfs * (num_cfs-1) / 2)


def compute_dpp(cfs, method='inverse_dist', dist=lp_dist):
    """Computes the DPP of a matrix."""
    num_cfs, d = cfs.shape
    det_entries = np.ones((num_cfs, num_cfs))
    if method == "inverse_dist":
        for i in range(num_cfs):
            for j in range(num_cfs):
                det_entries[(i, j)] = 1.0 / \
                    (1.0 + dist(cfs[i], cfs[j]))
                if i == j:
                    det_entries[(i, j)] += 0.0001

    elif method == "exponential_dist":
        for i in range(num_cfs):
            for j in range(num_cfs):
                det_entries[(i, j)] = 1.0 / \
                    (np.exp(dist(cfs[i], cfs[j])))
                if i == j:
                    det_entries[(i, j)] += 0.0001

    diversity_loss = np.linalg.det(det_entries)
    return diversity_loss


def compute_distance_manifold(plans, train_data_1, k):
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(train_data_1)

    dist = np.zeros(k)
    for i in range(k):
        idx = knn.kneighbors(plans[i].reshape(1, -1), return_distance=False)
        dist[i] = np.linalg.norm(plans[i] - train_data_1[idx])

    return max(dist)


def compute_likelihood(plans, train_data_1, k, gamma=100.):
    sigma = np.identity(plans.shape[1])

    density = np.zeros(k)
    for i in range(k):
        s = 0
        for j in range(train_data_1.shape[0]):
            s += -np.dot(plans[i] - train_data_1[j], plans[i] - train_data_1[j]) / (gamma ** 2)
        density[i] = s

    return np.mean(density)


def compute_kde(plans, train_data_1, gamma=100.):
    kde = KernelDensity(kernel='gaussian').fit(train_data_1)
    log_density = kde.score_samples(plans)
    return np.mean(log_density)


def compute_pairwise_cosine(x0, plans, k):
    A = (plans - x0).T / np.linalg.norm(plans - x0, axis=1)
    S = np.dot(A.T, A)

    diversity = (np.sum(S) - k) / 2

    return diversity


### Diversity of paths
def compute_diversity_path(d, paths, weighted_matrix=None):
    K = len(paths)
    diverse = 0
    for i in range(len(paths) - 1):
        for j in range(i + 1, len(paths)):
            diverse += d(paths[i], paths[j], weighted_matrix)

    return 2 * diverse / ((K - 1) * K)


def hamming_distance(p1, p2, weighted_matrix=None):
    p1, p2 = set(p1), set(p2)
    return len(p1.union(p2)) - len(p1.intersection(p2))


def levenshtein_distance(p1, p2, weighted_matrix=None):
    return lev(p1, p2)


def jaccard(p1, p2, weighted_matrix):
    """Compute Jaccard coefficient between 2 paths

    Args:
        p1: set of nodes of path 1
        p2: set of nodes of path 2

    Returns:
        res: value of Jaccard coefficient
    """
    # Convert from list of nodes to edges
    edges1 = {(p1[i], p1[i + 1]) for i in range(len(p1) - 1)}
    edges2 = {(p2[i], p2[i + 1]) for i in range(len(p2) - 1)}

    # Jaccard
    union = edges1.union(edges2)
    intersection = edges1.intersection(edges2)

    s1, s2 = 0, 0
    for edge in intersection:
        s1 += weighted_matrix[edge]

    for edge in union:
        s2 += weighted_matrix[edge]

    return s1 / s2
