import numpy as np
import scipy
import copy
import os
import torch
import joblib
import sklearn
from functools import partialmethod
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from collections import defaultdict, namedtuple
from time import time

import dice_ml

from utils import helpers
from utils.data_transformer import DataTransformer
from utils.funcs import compute_max_distance, lp_dist, compute_validity, compute_proximity, compute_diversity, compute_distance_manifold, compute_dpp, compute_likelihood, compute_pairwise_cosine, compute_kde, compute_proximity_graph, compute_proximity_graph_, compute_diversity_path, hamming_distance, levenshtein_distance, jaccard, mahalanobis_dist

from classifiers import mlp, random_forest

from methods.face import face
from methods.dice import dice
from methods.reup import reup, reup_graph, reup_graph_iden, reup_graph_gt
from methods.wachter import wachter, wachter_gt


Results = namedtuple("Results", ["l1_cost", "valid", "rank", "feasible"])
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

    rank_l = []
    l1_cost = np.zeros(params['num_A'])
    t0 = time()
    if method == dice or method==wachter:
        x_ar, feasible = method.generate_recourse(x0, model, random_state, params)

    for i in range(params['num_A']):
        params['A'] = params['all_A'][i]

        if method == reup:
            x_ar, rank, feasible = method.generate_recourse(x0, model, random_state, params)
            rank_l.append(rank)
        elif method == reup_graph:
            x_ar, cost, rank, feasible = method.generate_recourse(x0, model, random_state, params)
            l1_cost[i] = cost
        elif method == reup_graph_iden or method == reup_graph_gt:
            x_ar, cost, feasible = method.generate_recourse(x0, model, random_state, params)
            l1_cost[i] = cost
        elif method != dice and method != wachter:
            x_ar, feasible = method.generate_recourse(x0, model, random_state, params)
        if method != reup_graph and method != reup_graph_iden:
            l1_cost[i] = mahalanobis_dist(x_ar, x0, params['A'])
    
    if params['reup_params']['rank']:
        rank_l = np.array(rank_l)
    l1_cost = np.sum(l1_cost)
    rank = np.mean(rank_l, axis=0)

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model = model.to("cpu")
    # x_ar = torch.from_numpy(x_ar).to("cpu")
    valid = 1.0 if model.predict(x_ar) else 0.0

    return Results(l1_cost, valid, rank, feasible)


method_name_map = {
    'face': "FACE",
    'dice': 'DiCE',
    'dice_ga': 'DICE_GA',
    'gs': "GS",
    'reup': "ReAP-K",
    'pair': "ReAP-2",
    'reup_graph': "ReUP",
    'reup_graph_iden': "ReUP($T=0$)",
    'reup_graph_gt': "ReUP($T$)",
    'wachter': "Wachter",
    'gt': "GT",
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

metric_order = {'cost': -1, 'valid': 1}

metric_order_graph = {'cost': -1, 'valid': 1, 'diversity': -1, 'dpp': 1, 'hamming': 1, 'lev': 1, 'jac': -1}

method_map = {
    "face": face,
    "dice": dice,
    "reup": reup,
    "reup_graph": reup_graph,
    "reup_graph_iden": reup_graph_iden,
    "reup_graph_gt": reup_graph_gt,
    "wachter": wachter,
    "gt": wachter_gt,
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
