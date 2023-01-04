import copy
import numpy as np

from sklearn.utils import check_random_state
# import Dijkstra's shortest path algorithm
from scipy.sparse import csgraph, csr_matrix

# import graph building methods
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import networkx as nx
# from utils.visualization import visualize_explanations

def choose_random_subset(data, frac, index):
    """
    Choose a subset of data for computational efficiency

    Parameters
    ----------
    data : pd.DataFrame
    frac: float 0 < number =< 1
    fraction of data for which we compute the graph; if frac = 1, and data set large, then compute long
    index: int

    Returns
    -------
    pd.DataFrame
    """
    number_samples = np.int(np.rint(frac * data.shape[0]))
    list_to_choose = (
        np.arange(0, index).tolist()
        + np.arange(index + 1, data.shape[0]).tolist()
    )
    chosen_indeces = np.random.choice(
        list_to_choose,
        replace=True,
        size=number_samples,
    )
    chosen_indeces = [
        index
    ] + chosen_indeces.tolist()  # make sure sample under consideration included
    data = data[chosen_indeces]
    
    return data


def build_constraints(data, i, epsilon=0.5):
    """
    Parameters
    ----------
    data: pd.DataFrame
    i : int
        Position of immutable key
    keys_immutable: list[str]
    Immutable feature
    epsilon: int

    Returns
    -------
    np.ndarray, np.ndarray
    """
    immutable_constraint_matrix = np.outer(
            data[:, i] + epsilon,
            data[:, i] + epsilon,
    )
    immutable_constraint_matrix1 = immutable_constraint_matrix / ((1 + epsilon) ** 2)
    immutable_constraint_matrix1 = ((immutable_constraint_matrix1 == 1) * 1).astype(
        float
    )
    immutable_constraint_matrix2 = immutable_constraint_matrix / (epsilon ** 2)
    immutable_constraint_matrix2 = ((immutable_constraint_matrix2 == 1) * 1).astype(
        float
    )
    return immutable_constraint_matrix1, immutable_constraint_matrix2


def build_graph(data, immutable_constraint_matrix1, immutable_constraint_matrix2, is_knn, n):
    """

    Parameters
    ----------
    data: pd.DataFrame
    immutable_constraint_matrix1: np.ndarray
    immutable_constraint_matrix2: np.ndarray
    is_knn: bool
    n: int

    Returns
    -------
    CSR matrix
    """
    if is_knn:
        graph = kneighbors_graph(data, n_neighbors=n, n_jobs=-1)
    else:
        graph = radius_neighbors_graph(data, radius=n, n_jobs=-1)
    adjacency_matrix = graph.toarray()
    # adjacency_matrix = np.multiply(
    #     adjacency_matrix,
    #     immutable_constraint_matrix1,
    #     immutable_constraint_matrix2,
    # )  # element wise multiplication
    graph = csr_matrix(adjacency_matrix)
    return graph


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
    distances, predecessors = csgraph.dijkstra(
        csgraph=graph, directed=False, indices=index, return_predecessors=True
    )
    distances[index] = np.inf  # avoid min. distance to be x^F itself
    min_distance = distances.min()
    return distances, min_distance


def find_counterfactuals(candidates, data, immutable_constraint_matrix1, immutable_constraint_matrix2, index, n, y_positive_indeces, is_knn):
    """
    Steps 1 to 3 of the FACE algorithm

    Parameters
    ----------
    candidate_counterfactuals_star: list
    data: pd.DataFrame
    immutable_constraint_matrix1: np.ndarray
    immutable_constraint_matrix2: np.ndarray
    index: int
    n: int
    y_positive_indeces: int
    is_knn: bool

    Returns
    -------
    list
    """
    candidate_counterfactuals_star = copy.deepcopy(candidates)
    # STEP 1 -- BUILD NETWORK GRAPH
    graph = build_graph(
        data, immutable_constraint_matrix1, immutable_constraint_matrix2, is_knn, n
    )
    # STEP 2 -- APPLY SHORTEST PATH ALGORITHM  ## indeces=index (corresponds to x^F)
    distances, min_distance = shortest_path(graph, index)
    # STEP 3 -- FIND COUNTERFACTUALS
    # minimum distance candidate counterfactuals
    candidate_min_distances = [
        min_distance,
        min_distance + 1,
        min_distance + 2,
        min_distance + 3,
    ]
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
        candidate_counterfactuals_star.append(data[indeces_counterfactuals[i]])

    return candidate_counterfactuals_star


def graph_search(data, index, keys_immutables, model, n_neighbors=50, p_norm=2, mode="knn", frac=0.4, radius=0.25, K=3, graph_preprocess=False, graph=None):
    """
    :param data: df
    :param n_neighbors: int > 0; number of neighbors when constructing knn graph
    :param step: float > 0; step_size for growing spheres
    :param mode: str; either 'knn' or 'epsilon'
    :param model: classification model (either tf keras, pytorch or sklearn)
    :param p_norm: float=>1; denotes the norm (classical: 1 or 2)
    :param frac: float 0 < number =< 1; fraction of data for which we compute the graph; if frac = 1, and data set large, then compute long
    :param keys_immutable: list; list of input names that may not be searched over
    :param radius: float > 0; parameter for epsilon density graph
    :return: candidate_counterfactual_star: np array (min. cost counterfactual explanation)
    """
    # Choose a subset of data for computational efficiency
    data = choose_random_subset(data, frac, index)

    # ADD CONSTRAINTS by immutable inputs into adjacency matrix
    # if element in adjacency matrix 0, then it cannot be reached
    # this ensures that paths only take same sex / same race / ... etc. routes
    # for i in range(len(keys_immutable)):
    #     immutable_constraint_matrix1, immutable_constraint_matrix2 = build_constraints(
    #         data, i, keys_immutable
    #     )

    # POSITIVE PREDICTIONS
    y_predicted = model.predict_proba(data)
    y_predicted = np.argmax(y_predicted, axis=1)
    y_positive_indeces = np.where(y_predicted == 1)

    if mode == "knn":
        boundary = 3  # chosen in ad-hoc fashion
        median = n_neighbors
        is_knn = True

    elif mode == "epsilon":
        boundary = 0.10  # chosen in ad-hoc fashion
        median = radius
        is_knn = False

    else:
        raise ValueError("Only possible values for mode are knn and epsilon")

    neighbors_list = [
        median - boundary,
        median,
        median + boundary,
    ]

    # obtain candidate targets (CT); two conditions need to be met:
    # (1) CT needs to be predicted positively & (2) CT needs to have certain "density"
    # for knn I interpret 'certain density' as sufficient number of neighbours
    immutable_constraint_matrix1, immutable_constraint_matrix2 = None, None

    candidate_counterfactuals = []
    for n in neighbors_list:
        neighbor_candidates = find_counterfactuals(
                candidate_counterfactuals,
                data,
                immutable_constraint_matrix1,
                immutable_constraint_matrix2,
                index,
                n,
                y_positive_indeces,
                is_knn=is_knn,
                )
        candidate_counterfactuals += neighbor_candidates

    candidate_counterfactual_star = np.array(candidate_counterfactuals)
    candidate_counterfactual_star = np.unique(candidate_counterfactual_star, axis=0)

    # STEP 4 -- COMPUTE DISTANCES between x^F and candidate x^CF; else return NaN
    if candidate_counterfactual_star.size == 0:
        candidate_counterfactual_star = np.empty(
            data.values.shape[1],
        )
        candidate_counterfactual_star[:] = np.nan

        return candidate_counterfactual_star

    if p_norm == 1:
        c_dist = np.abs((data[index] - candidate_counterfactual_star)).sum(
            axis=1
        )
    elif p_norm == 2:
        c_dist = np.square((data[index] - candidate_counterfactuals)).sum(axis=1)
    else:
        raise ValueError("Distance not defined yet. Choose p_norm to be 1 or 2")

    # min_index = np.argmin(c_dist)
    min_index = np.argsort(c_dist)[:K]
    candidate_counterfactual_star = candidate_counterfactual_star[min_index]

    return candidate_counterfactual_star


def generate_recourse(x0, model, random_state, params=dict()):
    # Parameters
    mode = params['face_params']['mode']
    fraction = params['face_params']['fraction']
    train_data = params['train_data']
    graph_preprocess = params['graph_pre']
    graph_elem = params['graph']

    # Remove test instance from training instance and concat
    idx = np.all(x0 == train_data, axis=1)
    train_data = train_data[idx == False]
    train_data = np.concatenate([x0.reshape(1, -1), train_data])

    # Graph preprocess 
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
        
        data_star = data_[candidate_counterfactuals_star]
        distances_w_star = distances_w[candidate_counterfactuals_star]

        min_index = np.argsort(distances_w_star)[:params['k']]
        min_dist = distances_w_star[min_index]
        paths = []
        for i in range(len(min_index)):
            idx = np.all(data_star[min_index[i]] == data_, axis=1)
            idx = np.where(idx == True)[0][0]
            paths.append(nx.shortest_path(G, source=0, target=idx))
        candidate_counterfactual_star = data_star[min_index]

        return candidate_counterfactual_star, min_dist, paths, dict(feasible=True) 

    # Graph search
    cf = graph_search(train_data, 0, None, model, p_norm=1, mode=mode, frac=fraction, K=params['k'])
    report = dict(feasible=True)

    return cf, report
