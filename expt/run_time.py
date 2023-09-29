import os
import numpy as np
import pandas as pd
import copy
import joblib
import torch
import sklearn
import itertools
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter
from collections import defaultdict, namedtuple
from joblib import parallel_backend
from joblib.externals.loky import set_loky_pickler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler

import dice_ml

from utils import helpers
from utils.transformer import get_transformer
from utils.data_transformer import DataTransformer
from utils.funcs import compute_max_distance, lp_dist, find_pareto

from expt.common import synthetic_params, clf_map, method_map, method_name_map
from expt.common import dataset_name_map 
from expt.common import load_models, enrich_training_data


Results = namedtuple("Results", ["l1_cost", "cur_vald", "fut_vald", "feasible"])


def run_expt_run_time(ec, wdir, datasets, classifiers, methods,
               num_proc=4, plot_only=False, seed=None, logger=None,
               start_index=None, num_ins=None, rerun=True):
    logger.info("Running ept run time...")
    df, numerical = helpers.get_dataset("synthesis", params=synthetic_params)
    full_dice_data = dice_ml.Data(dataframe=df,
                    continuous_features=numerical,
                    outcome_name='label')
    transformer = DataTransformer(full_dice_data)

    y = df['label'].to_numpy()
    X_df = df.drop('label', axis=1)
    X = transformer.transform(X_df).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=42, stratify=y)

    d = X.shape[1]
    clf = clf_map["mlp"]
    model = load_models("synthesis", "mlp", "results/run_0/synthesis")

    y_pred = model.predict(X_test)
    uds_X, uds_y = X_test[y_pred == 0], y_test[y_pred == 0]
    uds_X, uds_y = uds_X[:100], uds_y[:100]

    ptv = "N"
    ptv_list = [100 * i for i in range(1, 50, 5)]
    
    time_greedy = []
    time_local  = []
    time_quad = []

    for value in ptv_list:
        _time_greedy = []
        _time_local = []
        _time_quad = []

        synthetic_params['num_samples'] = value
        df, numerical = helpers.get_dataset("synthesis", params=synthetic_params)
        full_dice_data = dice_ml.Data(dataframe=df,
                        continuous_features=numerical,
                        outcome_name='label')
        transformer = DataTransformer(full_dice_data)

        y = df['label'].to_numpy()
        X_df = df.drop('label', axis=1)
        X = transformer.transform(X_df).to_numpy()

        for i in range(len(uds_y)):
            cur_sol, cur_prob, ls_time, greedy_sol, greedy_prob, greedy_time = dpp_recourse(uds_X[i, :], X[y == 1], 3, gamma=0.5, sigma=2.)
            quad_time = quad_recourse(uds_X[i, :], 3, model, X, y, 0.5, 2.)
            _time_greedy.append(greedy_time)
            _time_local.append(ls_time)
            _time_quad.append(quad_time)

        _time_greedy = np.array(_time_greedy)
        _time_local = np.array(_time_local)
        _time_quad = np.array(_time_quad)

        time_greedy.append(np.mean(_time_greedy))
        time_local.append(np.mean(_time_local))
        time_quad.append(np.mean(_time_quad))

    res = {}
    res['greedy'] = time_greedy
    res['local'] = time_local
    res['quad'] = time_quad

    helpers.pdump(res, "time.pickle", "results/run_0/")

    # Matplotlib config
    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 18

    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize

    # Plot
    fig, ax = plt.subplots()

    ax.plot(ptv_list, time_greedy, label='Greedy', linewidth=5.0)
    ax.plot(ptv_list, time_local, label='Local search', linewidth=5.0)
    ax.plot(ptv_list, time_quad, label='Quad', linewidth=5.0)
    print(time_greedy, time_local, time_quad)

    ax.set(xlabel='$N$', ylabel='time')
    ax.grid()
    ax.legend(loc='lower right', frameon=False)
    plt.savefig('results/run_0/time.pdf', dpi=400)
    plt.tight_layout()

    plt.show()# Define shift params


    logger.info("Done ept run time.")
