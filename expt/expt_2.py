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
from expt.common import _run_single_instance, _run_single_instance_plans, to_numpy_array
from expt.common import load_models, enrich_training_data
from expt.expt_config import Expt2


Results = namedtuple("Results", ["l1_cost", "cur_vald", "fut_vald", "feasible"])

param_to_vary = {
    "reup": "T",
}


def run(ec, wdir, dname, cname, mname,
        num_proc, seed, logger, start_index=None, num_ins=None):
                # dname, cname, mname)
    print("Running dataset: %s, classifier: %s, method: %s..." %
                (dname, cname, mname))
    full_l = ["synthesis", "german"]
    df, numerical = helpers.get_dataset(dname, params=synthetic_params) if dname in full_l else helpers.get_full_dataset(dname, params=synthetic_params)
    full_dice_data = dice_ml.Data(dataframe=df,
                     continuous_features=numerical,
                     outcome_name='label')
    transformer = DataTransformer(full_dice_data)

    y = df['label'].to_numpy()
    X_df = df.drop('label', axis=1)
    X = transformer.transform(X_df).to_numpy()
    cat_indices = [len(numerical) + i for i in range(X.shape[1] - len(numerical))]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)

    d = X.shape[1]
    clf = clf_map[cname]
    model = load_models(dname, cname, wdir)
    y_labels = model.predict(X_train)
    y_pred = model.predict(X_test)               
    uds_X, uds_y = X_test[y_pred == 0], y_test[y_pred == 0]                                           

    ptv = param_to_vary[mname]
    method = method_map[mname]

    min_ptv = ec.params_to_vary[ptv]['min']
    max_ptv = ec.params_to_vary[ptv]['max']
    step_size = ec.params_to_vary[ptv]['step']
    ptv_list = np.arange(min_ptv, max_ptv+step_size, step_size)

    res = dict()
    res['ptv_name'] = ptv
    res['ptv_list'] = ptv_list
    res['cost'] = []
    res['feasible'] = []

    A = np.random.rand(X_train.shape[1], X_train.shape[1])                                                
    A = np.dot(A, A.T)                               

    for value in ptv_list:
        print("varying %s = %f" % (ptv, value))
        new_config = Expt2(ec.to_dict())

        if start_index is not None or num_ins is not None:
            num_ins = num_ins or 1
            start_index = start_index or 0
            uds_X = uds_X[start_index: start_index + num_ins]
            uds_y = uds_y[start_index: start_index + num_ins]
        else:
            uds_X, uds_y = uds_X[:ec.max_ins], uds_y[:ec.max_ins]

        params = dict(train_data=X_train,
                      labels=y_labels,
                      dataframe=df,
                      numerical=numerical,
                      config=new_config,
                      method_name=mname,
                      dataset_name=dname,
                      k=ec.k,
                      transformer=transformer,
                      cat_indices=cat_indices,
                      A=A,)

        params['dice_params'] = ec.dice_params
        params['reup_params'] = ec.reup_params

        if ptv == 'T':
            params['reup_params']['T'] = value
        
        rets = []
        for idx, x0 in enumerate(uds_X):
            ret = _run_single_instance(idx, method, x0, model, seed, logger, params)
            rets.append(ret)

        cost = []
        feasible = []

        for ret in rets:
            cost.append(ret.l1_cost)
            feasible.append(ret.feasible)

        res['cost'].append(np.array(cost))
        res['feasible'].append(np.array(feasible))

    helpers.pdump(res,
                  f'{cname}_{dname}_{mname}.pickle', wdir)

    logger.info("Done dataset: %s, classifier: %s, method: %s!",
                dname, cname, mname)


label_map = {
    'cost': 'Cost',
    'ptv_list': 'T',
}

def plot_2(ec, wdir, cname, dname, methods):
    def plot(methods, x_label, y_label, data):
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots()
        marker = reversed(['*', 'v', '^', 'o', (5, 1), (5, 0), '+', 's'])
        iter_marker = itertools.cycle(marker)

        for mname in methods:
            # X, y = find_pareto(data[mname][x_label], data[mname][y_label])
            X, y = data[mname][x_label], data[mname][y_label]
            ax.plot(X, y, marker=next(iter_marker),
                    label=method_name_map[mname], alpha=0.8)

        ax.set_ylabel(label_map[y_label])
        ax.set_xlabel(label_map[x_label])
        ax.set_yscale('log')
        ax.legend(prop={'size': 14})
        filepath = os.path.join(wdir, f"{cname}_{dname}_{x_label}_{y_label}.png")
        plt.savefig(filepath, dpi=400, bbox_inches='tight')

    data = defaultdict(dict)
    joint_feasible = None
    for mname in methods:
        res = helpers.pload(
            f'{cname}_{dname}_{mname}.pickle', wdir)
        for feasible in res['feasible']:
            if joint_feasible is None:
                joint_feasible = feasible
            # joint_feasible = np.logical_and(joint_feasible, feasible)

    for mname in methods:
        res = helpers.pload(
            f'{cname}_{dname}_{mname}.pickle', wdir)

        data[dname][mname] = {}
        data[mname]['ptv_name'] = res['ptv_name']
        data[mname]['ptv_list'] = res['ptv_list']
        data[mname]['cost'] = []

        for i in range(len(res['ptv_list'])):
            data[mname]['cost'].append(np.mean(res['cost'][i]))

    plot(methods, 'ptv_list', 'cost', data)


def plot_2_1(ec, wdir, cname, datasets, methods):
    def __plot(ax, data, dname, x_label, y_label):
        marker = reversed(['+', 'v', '^', 'o', (5, 0)])
        iter_marker = itertools.cycle(marker)

        for mname, o in data[dname].items():
            if mname == 'wachter':
                ax.scatter(data[dname][mname][x_label], data[dname][mname][y_label],
                           marker=(5, 1), label=method_name_map[mname], alpha=0.7, color='black', zorder=10)
            else:
                X, y = find_pareto(data[dname][mname][x_label], data[dname][mname][y_label])
                # X, y = data[dname][mname][x_label], data[dname][mname][y_label]
                ax.plot(X, y, marker=next(iter_marker),
                        label=method_name_map[mname], alpha=0.7, linewidth=2, markersize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_title(dataset_name_map[dname])

    data = defaultdict(dict)
    for dname in datasets:
        joint_feasible = None
        for mname in methods:
            res = helpers.pload(
                f'{cname}_{dname}_{mname}.pickle', wdir)
            for feasible in res['feasible']:
                if joint_feasible is None:
                    joint_feasible = feasible
                # joint_feasible = np.logical_and(joint_feasible, feasible)

        for mname in methods:
            res = helpers.pload(
                f'{cname}_{dname}_{mname}.pickle', wdir)

            # print(res)
            data[dname][mname] = {}
            data[dname][mname]['ptv_name'] = res['ptv_name']
            data[dname][mname]['ptv_list'] = res['ptv_list']
            data[dname][mname]['cost'] = []
            data[dname][mname]['diversity'] = []
            data[dname][mname]['dpp'] = []
            data[dname][mname]['manifold_dist'] = []

            for i in range(len(res['ptv_list'])):
                data[dname][mname]['cost'].append(np.mean(res['cost'][i]))
                data[dname][mname]['diversity'].append(np.mean(res['diversity'][i]))
                data[dname][mname]['dpp'].append(np.mean(res['dpp'][i]))
                data[dname][mname]['manifold_dist'].append(np.mean(res['manifold_dist'][i]))

    plt.style.use('seaborn-deep')
    plt.rcParams.update({'font.size': 24})
    num_ds = len(datasets)
    metrics = ['diversity', 'dpp']
    figsize_map = {5: (30, 5.5), 4: (30, 12), 3: (20, 5.5), 2: (10, 5.5), 1: (6, 5)}
    fig, axs = plt.subplots(len(metrics), num_ds, figsize=figsize_map[num_ds])
    if num_ds == 1:
        axs = axs.reshape(-1, 1)

    for i in range(num_ds):
        for j in range(len(metrics)):
            __plot(axs[j, i], data, datasets[i], 'cost', metrics[j])
            if i == 0:
                axs[j, i].set_ylabel(label_map[metrics[j]])
            if j == len(metrics) - 1:
                axs[j, i].set_xlabel(label_map['cost'])

    marker = reversed(['+', 'v', '^', 'o', (5, 0)])
    iter_marker = itertools.cycle(marker)
    ax = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    for mname in methods:
        if mname == 'wachter':
            ax.scatter([] , [], marker=(5, 1), label=method_name_map[mname], alpha=0.7, color='black')
        else:
            ax.plot([] , marker=next(iter_marker), label=method_name_map[mname], alpha=0.7)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.23 - .1 * (len(methods) > 5)),
              ncol=min(len(methods), 5), frameon=False)
    plt.tight_layout()
    joint_dname = ''.join([e[:2] for e in datasets])
    filepath = os.path.join(wdir, f"{cname}_{joint_dname}.pdf")
    plt.savefig(filepath, dpi=400, bbox_inches='tight')


def run_expt_2(ec, wdir, datasets, classifiers, methods,
               num_proc=4, plot_only=False, seed=None, logger=None,
               start_index=None, num_ins=None, rerun=True):
    logger.info("Running ept 2...")

    if datasets is None or len(datasets) == 0:
        datasets = ec.e2.all_datasets

    if classifiers is None or len(classifiers) == 0:
        classifiers = ec.e2.all_clfs

    if methods is None or len(methods) == 0:
        methods = ec.e2.all_methods


    if not plot_only:
        jobs_args = []

        for cname in classifiers:
            cmethods = copy.deepcopy(methods)
            if cname == 'rf' and 'wachter' in cmethods:
                cmethods.remove('wachter')

            for dname in datasets:
                for mname in cmethods:
                    filepath = os.path.join(wdir, f"{cname}_{dname}_{mname}.pickle")
                    if not os.path.exists(filepath) or rerun:
                        jobs_args.append((ec.e2, wdir, dname, cname, mname,
                            num_proc, seed, logger, start_index, num_ins))

        rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(run)(
            *jobs_args[i]) for i in range(len(jobs_args)))

    for cname in classifiers:
        cmethods = copy.deepcopy(methods)
        if cname == 'rf' and 'wachter' in cmethods:
            cmethods.remove('wachter')            
        for dname in datasets:
            plot_2(ec.e2, wdir, cname, dname, cmethods)
        # plot_2_1(ec.e2, wdir, cname, datasets, cmethods)

    logger.info("Done ept 2.")
