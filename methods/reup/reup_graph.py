import numpy as np

from methods.reup.chebysev import chebysev_center
from methods.reup.q_determine import exhaustive_search, find_q
from methods.reup.graph import build_graph, shortest_path_graph, eval_cost


def generate_recourse(x0, model, random_state, params=dict()):
    # General parameters
    train_data = params['train_data']
    labels = params['labels']
    data = train_data[labels == 1]

    train_data = np.concatenate([x0.reshape(1, -1), train_data])

    pos_idx = np.where(labels == 1)[0] + 1

    cat_indices = params['cat_indices']

    # Graph parameters
    T = params['reup_params']['T']
    epsilon = params['reup_params']['eps']
    is_knn = params['reup_params']['knn']
    n = params['reup_params']['n']

    # Questions generation
    P, A_opt, mean_rank = find_q(x0, data, T, params['A'], epsilon, False)

    # Recourse generation
    # graph_0 = build_graph(train_data, params['A'], is_knn, n)
    graph_opt = build_graph(train_data, A_opt, is_knn, n)
    path = shortest_path_graph(graph_opt, pos_idx)[2]
    recourse = train_data[path[-1]]
    # graph_iden = build_graph(train_data, np.eye(train_data.shape[1]), is_knn, n)
    cost = eval_cost(params['A'], train_data, path)
    print(cost)
    feasible = True

    return recourse, cost, mean_rank, feasible
