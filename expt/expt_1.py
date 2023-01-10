import os
import numpy as np
import pandas as pd
import joblib
import torch
import sklearn
import copy

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
from utils.funcs import compute_max_distance, lp_dist

from expt.common import synthetic_params, synthetic_params_mean_cov, clf_map, method_map
from expt.common import _run_single_instance, _run_single_instance_plans, _run_single_instance_plans_graph, to_mean_std
from expt.common import load_models, enrich_training_data
from expt.common import method_name_map, dataset_name_map, metric_order, metric_order_graph
from expt.expt_config import Expt1


def run(ec, wdir, dname, cname, mname,
        num_proc, seed, logger):
    print("Running dataset: %s, classifier: %s, method: %s..."
               % (dname, cname, mname))
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=42, stratify=y)
    # data_nodes, labels_nodes, adjacency_matrix = helpers.build_action_graph(X_train, y_train, 2, transformer, ["Age", "Personal status"])
    new_config = Expt1(ec.to_dict())
    
    # graph = helpers.pload(f"{dname}_{ec.n_neighbors}.pickle", "results/run_0/graph")
    # nodes = graph["data"]
    # node_labels = graph["labels"]
    # adj_matrix = graph["adj"]
    # weighted_adj_matrix = graph["weighted_adj"]

    d = X.shape[1]
    clf = clf_map[cname]
    model = load_models(dname, cname, wdir)
    method = method_map[mname]

    l1_cost = []
    valid = []
    feasible = []

    y_pred = model.predict(X_test)
    uds_X, uds_y = X_test[y_pred == 0], y_test[y_pred == 0]
    uds_X, uds_y = uds_X[:ec.max_ins], uds_y[:ec.max_ins]
    A = np.random.rand(X_train.shape[1], X_train.shape[1])
    A = np.dot(A, A.T)
    params = dict(train_data=X_train,
                  labels=y_train,
                  dataframe=df,
                  numerical=numerical,
                  config=new_config,
                  method_name=mname,
                  dataset_name=dname,
                  transformer=transformer,
                  cat_indices=cat_indices,
                  A=A,)
                  # transformer=transformer,
                  # graph_pre=ec.graph_pre)

    params['reup_params'] = ec.reup_params
    params['wachter_params'] = ec.wachter_params
    params['dice_params'] = ec.dice_params

    jobs_args = []

    for idx, x0 in enumerate(uds_X):
        jobs_args.append((idx, method, x0, model, seed, logger, params))

    rets = joblib.Parallel(n_jobs=min(num_proc, 8), prefer="threads")(joblib.delayed(_run_single_instance)(*jobs_args[i]) for i in range(len(jobs_args)))

    for ret in rets:
        l1_cost.append(ret.l1_cost)
        valid.append(ret.valid)
        feasible.append(ret.feasible)

    def to_numpy_array(lst):
        pad = len(max(lst, key=len))
        return np.array([i + [0]*(pad-len(i)) for i in lst])

    l1_cost = np.array(l1_cost)
    valid = np.array(valid)
    feasible = np.array(feasible)

    helpers.pdump((l1_cost, valid, feasible),
                  f'{cname}_{dname}_{mname}.pickle', wdir)

    logger.info("Done dataset: %s, classifier: %s, method: %s!",
                dname, cname, mname)

def plot_1(ec, wdir, cname, datasets, methods):
    res = defaultdict(list)
    res2 = defaultdict(list)

    for mname in methods:
        res2['method'].append(method_name_map[mname])

    for i, dname in enumerate(datasets):
        res['dataset'].extend([dataset_name_map[dname]] + [""] * (len(methods) - 1))

        joint_feasible = None
        for mname in methods:
            _, _, feasible = helpers.pload(
                f'{cname}_{dname}_{mname}.pickle', wdir)
            if joint_feasible is None:
                joint_feasible = np.ones_like(feasible)
            if '_ar' in mname:
                joint_feasible = np.logical_and(joint_feasible, feasible)

        temp = defaultdict(dict)

        for metric, order in metric_order.items():
            temp[metric]['best'] = -np.inf

        for mname in methods:
            l1_cost, valid, feasible = helpers.pload(
                f'{cname}_{dname}_{mname}.pickle', wdir)
            avg = {}
            avg['cost'] = l1_cost 
            avg['valid'] = valid 

            for metric, order in metric_order.items():
                m, s = np.mean(avg[metric]), np.std(avg[metric])
                temp[metric][mname] = (m, s)
                temp[metric]['best'] = max(temp[metric]['best'], m * order)

            temp['feasible'][mname] = np.mean(feasible)

        for mname in methods:
            res['method'].append(method_name_map[mname])
            for metric, order in metric_order.items():
                m, s = temp[metric][mname]
                is_best = (temp[metric]['best'] == m * order)
                res[metric].append(to_mean_std(m, s, is_best))
                res2[f"{metric}-{dname[:2]}"].append(to_mean_std(m, s, is_best))

            res[f'feasible'].append("{:.2f}".format(temp['feasible'][mname]))

    df = pd.DataFrame(res)
    filepath = os.path.join(wdir, f"{cname}{'_ar' if '_ar' in methods[0] else ''}.csv")
    df.to_csv(filepath, index=False, float_format='%.2f')

    df = pd.DataFrame(res2)
    filepath = os.path.join(wdir, f"{cname}_hor{'_ar' if '_ar' in methods[0] else ''}.csv")
    df.to_csv(filepath, index=False, float_format='%.2f')


def run_expt_1(ec, wdir, datasets, classifiers, methods,
               num_proc=4, plot_only=False, seed=None, logger=None, rerun=True):
    logger.info("Running ept 1...")

    if datasets is None or len(datasets) == 0:
        datasets = ec.e1.all_datasets

    if classifiers is None or len(classifiers) == 0:
        classifiers = ec.e1.all_clfs

    if methods is None or len(methods) == 0:
        methods = ec.e1.all_methods

    jobs_args = []
    if not plot_only:
        for cname in classifiers:
            cmethods = copy.deepcopy(methods)
            if cname == 'rf' and 'wachter' in cmethods:
                cmethods.remove('wachter')            

            for dname in datasets:
                for mname in cmethods:
                    filepath = os.path.join(wdir, f"{cname}_{dname}_{mname}.pickle")
                    if not os.path.exists(filepath) or rerun:
                        jobs_args.append((ec.e1, wdir, dname, cname, mname,
                            num_proc, seed, logger))

        rets = joblib.Parallel(n_jobs=num_proc)(joblib.delayed(run)(
            *jobs_args[i]) for i in range(len(jobs_args)))

    for cname in classifiers:
        cmethods = copy.deepcopy(methods)
        if cname == 'rf' and 'wachter' in cmethods:
            cmethods.remove('wachter')            

        plot_1(ec.e1, wdir, cname, datasets, cmethods)

    logger.info("Done ept 1.")
