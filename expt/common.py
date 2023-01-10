import numpy as np
import copy
import os
import torch
import joblib
import sklearn
from functools import partialmethod
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from collections import defaultdict, namedtuple

import dice_ml

from utils import helpers
from utils.data_transformer import DataTransformer
from utils.funcs import compute_max_distance, lp_dist, compute_validity, compute_proximity, compute_diversity, compute_distance_manifold, compute_dpp, compute_likelihood, compute_pairwise_cosine, compute_kde, compute_proximity_graph, compute_proximity_graph_, compute_diversity_path, hamming_distance, levenshtein_distance, jaccard, mahalanobis_dist

from classifiers import mlp, random_forest

from methods.face import face
from methods.dice import dice
from methods.gs import gs
from methods.reup import reup
from methods.wachter import wachter, wachter_reb


# Results = namedtuple("Results", ["l1_cost", "cur_vald", "fut_vald", "feasible"])
# Results = namedtuple("Results", ["valid", "l1_cost", "diversity", "dpp", "manifold_dist", "likelihood", "feasible"])
Results = namedtuple("Results", ["l1_cost", "valid", "feasible"])
Results_graph = namedtuple("Results_graph", ["valid", "l1_cost", "diversity", "dpp", "manifold_dist", "hamming",  "lev", "jac", "feasible"])


def to_numpy_array(lst):
    pad = len(max(lst, key=len))
    return np.array([i + [0]*(pad-len(i)) for i in lst])


def load_models(dname, cname, wdir):
    pdir = os.path.dirname(wdir)
    pdir = os.path.join(pdir, 'checkpoints')
    models = helpers.pload(f"{cname}_{dname}.pickle", pdir)
    return models


def calc_future_validity(x, shifted_models):
    preds = []
    for model in shifted_models:
        pred = model.predict(x)
        preds.append(pred)
    preds = np.array(preds)
    return np.mean(preds)


def enrich_training_data(num_samples, train_data, cat_indices, rng):
    rng = check_random_state(rng)
    cur_n, d = train_data.shape
    min_f_val = np.min(train_data, axis=0)
    max_f_val = np.max(train_data, axis=0)
    new_data = rng.uniform(min_f_val, max_f_val, (num_samples - cur_n, d))

    # new_data = rng.normal(0, 1, (num_samples - cur_n, d))
    # scaler = StandardScaler()
    # scaler.fit(train_data)
    # new_data = new_data * scaler.scale_ + scaler.mean_

    new_data[:, cat_indices] = new_data[:, cat_indices] >= 0.5

    new_data = np.vstack([train_data, new_data])
    return new_data


def to_mean_std(m, s, is_best):
    if is_best:
        return "\\textbf{" + "{:.2f}".format(m) + "}" + " $\pm$ {:.2f}".format(s)
    else:
        return "{:.2f} $\pm$ {:.2f}".format(m, s)


def _run_single_instance(idx, method, x0, model, seed, logger, params=dict()):
    torch.manual_seed(seed+2)
    np.random.seed(seed+1)
    random_state = check_random_state(seed)

    x_ar, feasible = method.generate_recourse(x0, model, random_state, params)

    # l1_cost = lp_dist(x0, x_ar, p=1)
    l1_cost = mahalanobis_dist(x_ar, x0, params['A'])
    valid = 1.0 if feasible else 0.0

    return Results(l1_cost, valid, feasible)


def _run_single_instance_plans(idx, method, x0, model, seed, logger, params=dict()):
    # logger.info("Generating recourse for instance : %d", idx)
    torch.manual_seed(seed+2)
    np.random.seed(seed+1)
    random_state = check_random_state(seed)

    df = params['dataframe']
    numerical = params['numerical']
    k = params['k']
    transformer = params['transformer']

    full_dice_data = dice_ml.Data(dataframe=df,
                              continuous_features=numerical,
                              outcome_name='label')
    plans, report = method.generate_recourse(x0, model, random_state, params)
    # print(idx, transformer.inverse_transform(x0.reshape(1, -1)))
    # print(transformer.inverse_transform(plans.reshape(3, -1)))

    valid = compute_validity(model, plans)
    l1_cost = compute_proximity(x0, plans, p=2)
    # diversity = compute_diversity(plans, transformer.data_interface)
    diversity = compute_pairwise_cosine(x0, plans, params['k'])
    manifold_dist = compute_distance_manifold(plans, params['train_data'], params['k'])
    dpp = compute_dpp(plans)
    # likelihood = compute_likelihood(plans, params['train_data'], params['k'])
    likelihood = compute_kde(plans, params['train_data'][params['labels'] == 1])

    return Results(valid, l1_cost, diversity, dpp, manifold_dist, likelihood, report['feasible'])


def _run_single_instance_plans_graph(idx, method, x0, graph, model, seed, logger, params=dict()):
    # logger.info("Generating recourse for instance : %d", idx)
    torch.manual_seed(seed+2)
    np.random.seed(seed+1)
    random_state = check_random_state(seed)

    df = params['dataframe']
    numerical = params['numerical']
    k = params['k']
    transformer = params['transformer']

    full_dice_data = dice_ml.Data(dataframe=df,
                              continuous_features=numerical,
                              outcome_name='label')
    params["graph"] = graph
    # print(graph["data"][0], x0)
    plans, dist, paths, report = method.generate_recourse(x0, model, random_state, params)

    valid = compute_validity(model, plans)
    # l1_cost = compute_proximity(x0, plans, p=2)
    l1_cost = np.mean(dist)
    diversity = compute_pairwise_cosine(x0, plans, params['k'])
    manifold_dist = compute_distance_manifold(plans, params['train_data'], params['k'])
    dpp = compute_dpp(plans)
    # likelihood = compute_kde(plans, params['train_data'][params['labels'] == 1])
    hamming = compute_diversity_path(hamming_distance, paths)
    lev = compute_diversity_path(levenshtein_distance, paths)
    jac = compute_diversity_path(jaccard, paths, weighted_matrix=graph['weighted_adj'])

    return Results_graph(valid, l1_cost, diversity, dpp, manifold_dist, hamming, lev, jac, report['feasible'])


method_name_map = {
    'face': "FACE",
    'dice': 'DiCE',
    'dice_ga': 'DICE_GA',
    'gs': "GS",
    'reup': "ReUP",
    'wachter': "Wachter",
}


dataset_name_map = {
    "synthesis": "Synthetic data",
    "german": "German",
    "sba": "SBA",
    "bank": "Bank",
    "student": "Student",
    "adult": "Adult",
    "compas": "Compas",
}

# metric_order = {'cost': -1, 'valid': 1, 'diversity': -1, 'dpp': 1, 'manifold_dist': -1, 'likelihood': 1}
metric_order = {'cost': -1, 'valid': 1}

metric_order_graph = {'cost': -1, 'valid': 1, 'diversity': -1, 'dpp': 1, 'hamming': 1, 'lev': 1, 'jac': -1}

method_map = {
    "face": face,
    "dice": dice,
    # "dice_ga": dice_ga,
    "gs": gs,
    "reup": reup,
    "wachter": wachter,
}


clf_map = {
    "net0": mlp.Net0,
    "mlp": mlp.Net0,
    "rf": random_forest.RandomForest,
}


train_func_map = {
    'net0': mlp.train,
    'mlp': mlp.train,
    'rf': random_forest.train,
}


synthetic_params = dict(num_samples=1000,
                        x_lim=(-2, 4), y_lim=(-2, 7),
                        f=lambda x, y: y >= 1 + x + 2*x**2 + x**3 - x**4,
                        random_state=42)


synthetic_params_mean_cov = dict(num_samples=1000, mean_0=None, cov_0=None, mean_1=None, cov_1=None, random_state=42)
