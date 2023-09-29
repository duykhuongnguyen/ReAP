import logging
import numpy as np
import pandas as pd
import shutil
import os
import collections
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph

from utils.validation import check_random_state


def load_adult_income_dataset(filepath=None, only_train=False):
    """Loads adult income dataset from https://archive.ics.uci.edu/ml/datasets/Adult and prepares
       the data for data analysis based on https://rpubs.com/H_Zhu/235617
    :return adult_data: returns preprocessed adult income dataset.
    """
    #  column names from "https://archive.ics.uci.edu/ml/datasets/Adult"
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'income']

    if filepath is None:
        raw_data = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                                 delimiter=', ', dtype=str, invalid_raise=False)

        adult_data = pd.DataFrame(raw_data, columns=column_names)
    else:
        adult_data = pd.read_csv(filepath, names=column_names, dtype=str)

        df_obj = adult_data.select_dtypes(['object'])
        adult_data[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    # For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
    adult_data = adult_data.astype(
        {"age": np.int64, "educational-num": np.int64, "hours-per-week": np.int64})

    adult_data = adult_data.replace(
        {'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    adult_data = adult_data.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government',
                                     'Local-gov': 'Government'}})
    adult_data = adult_data.replace(
        {'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    adult_data = adult_data.replace(
        {'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'?': 'Other/Unknown'}})

    adult_data = adult_data.replace(
        {
            'occupation': {
                'Adm-clerical': 'White-Collar', 'Craft-repair': 'Blue-Collar',
                'Exec-managerial': 'White-Collar', 'Farming-fishing': 'Blue-Collar',
                'Handlers-cleaners': 'Blue-Collar',
                'Machine-op-inspct': 'Blue-Collar', 'Other-service': 'Service',
                'Priv-house-serv': 'Service',
                'Prof-specialty': 'Professional', 'Protective-serv': 'Service',
                'Tech-support': 'Service',
                'Transport-moving': 'Blue-Collar', 'Unknown': 'Other/Unknown',
                'Armed-Forces': 'Other/Unknown', '?': 'Other/Unknown'
            }
        }
    )

    adult_data = adult_data.replace({'marital-status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married',
                                                        'Married-spouse-absent': 'Married', 'Never-married': 'Single'}})

    adult_data = adult_data.replace({'race': {'Black': 'Other', 'Asian-Pac-Islander': 'Other',
                                              'Amer-Indian-Eskimo': 'Other'}})

    adult_data = adult_data[['age', 'workclass', 'education', 'marital-status', 'occupation',
                             'race', 'gender', 'hours-per-week', 'income']]

    adult_data = adult_data.replace({'income': {'<=50K': 0, '>50K': 1}})

    adult_data = adult_data.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                                   '11th': 'School', '10th': 'School', '7th-8th': 'School',
                                                   '9th': 'School', '12th': 'School', '5th-6th': 'School',
                                                   '1st-4th': 'School', 'Preschool': 'School'}})

    adult_data = adult_data.rename(
        columns={'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week'})

    if only_train:
        train, _ = train_test_split(adult_data, test_size=0.2, random_state=17)
        adult_data = train.reset_index(drop=True)

    # Remove the downloaded dataset
    if os.path.isdir('archive.ics.uci.edu'):
        entire_path = os.path.abspath('archive.ics.uci.edu')
        shutil.rmtree(entire_path)

    adult_data = adult_data.rename(columns={"income": "label"})
    continuous_features = ['age', 'hours_per_week']

    return adult_data, continuous_features


def gen_synthetic_data(num_samples=1000, mean_0=None, cov_0=None, mean_1=None, cov_1=None, random_state=None):
    random_state = check_random_state(random_state)

    if mean_0 is None or cov_0 is None or mean_1 is None or cov_1 is None:
        mean_0 = np.array([-2, -2])
        cov_0 = np.array([[0.5, 0], [0, 0.5]])
        mean_1 = np.array([2, 2])
        cov_1 = np.array([[0.5, 0], [0, 0.5]])
    
    num_class0 = random_state.binomial(n=num_samples, p=0.5)
    x_class0 = random_state.multivariate_normal(mean_0, cov_0, num_class0)
    x_class1 = random_state.multivariate_normal(
        mean_1, cov_1, num_samples-num_class0)
    data0 = np.hstack([x_class0, np.zeros((num_class0, 1))])
    data1 = np.hstack([x_class1, np.ones((num_samples-num_class0, 1))])
    raw_data = np.vstack([data0, data1])
    random_state.shuffle(raw_data)
    column_names = ['f' + str(i) for i in range(len(mean_0))] + ['label']
    df = pd.DataFrame(raw_data, columns=column_names)
    return df


def gen_synthetic_data_nl(num_samples=1000, x_lim=(-2, 4), y_lim=(-2, 7),
        f=lambda x, y: y >= 1 + x, random_state=42, add_noise=False):
# : + 2*x**2 + x**3 - x**4,
 #                          random_state=42, add_noise=False):
    random_state = check_random_state(random_state)
    std = 1.0
    x = random_state.uniform(x_lim[0], x_lim[1], num_samples)
    y = random_state.uniform(y_lim[0], y_lim[1], num_samples)
    noisy_y = y + random_state.normal(0, std, size=y.shape)
    label = f(x, noisy_y if add_noise else y).astype(np.int32)
    raw_data = {'f0': x, 'f1': y, 'label': label}
    df = pd.DataFrame(raw_data)
    return df


def get_dataset(dataset='synthesis', params=list()):
    if 'synthesis' in dataset:
        if isinstance(params, collections.Sequence):
            params.append('shift' in dataset)
            dataset = gen_synthetic_data_nl(*params)
        else:
            if 'shift' in dataset:
                params['add_noise'] = 'shift' in dataset
            dataset = gen_synthetic_data_nl(**params)
        numerical = list(dataset.columns)
        numerical.remove('label')
    elif 'adult' in dataset:
        # dataset, numerical = load_adult_income_dataset('./data/adult.csv')
        # print(numerical)
        joint_dataset = pd.read_csv('./data/adult.csv')
        numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'hours-per-week', 'capital-loss']
    elif 'german' in dataset:
        dataset = pd.read_csv('./data/corrected_german_small.csv'
                              if 'shift' in dataset else './data/german_small.csv')
        # numerical = ['Duration', 'Credit amount', 'Installment rate',
                     # 'Present residence', 'Age', 'Existing credits', 'Number people']
        numerical = ['Duration', 'Credit amount', 'Age']
    elif 'sba' in dataset:
        dataset = pd.read_csv('./data/sba_8905.csv'
                              if 'shift' in dataset else './data/sba_0614.csv')
        categorical = ['LowDoc', 'RevLineCr', 'NewExist',
                       'MIS_Status', 'UrbanRural', 'label']
        numerical = list(dataset.columns.difference(categorical))
    elif 'student' in dataset:
        dataset = pd.read_csv('./data/ms_student.csv'
                              if 'shift' in dataset else './data/gp_student.csv')
        numerical = ['Fedu', 'G1', 'G2', 'Medu', 'absences',
                     'age', 'freetime', 'goout', 'health', 'studytime']
    elif 'bank' in dataset:
        dataset = pd.read_csv('./data/bank_shift.csv'
                              if 'shift' in dataset else './data/bank.csv')
        numerical = ['age', 'balance', 'campaign', 'previous']
    elif 'compas' in dataset:
        dataset = pd.read_csv('./data/compas.csv')
        numerical = ['age', 'two_year_recid', 'priors_count', 'length_of_stay']
    elif 'gmc' in dataset:
        dataset = pd.read_csv('.data/gmc.csv')
        numerical = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
    elif 'heloc' in dataset:
        dataset = pd.read_csv('./data/heloc.csv')
        numerical = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'NumTotalTrades', 'NumTradesOpeninLast12M', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']
    else:
        raise ValueError("Unknown dataset")

    return dataset, numerical


def get_full_dataset(dataset='synthesis', params=list()):
    if 'synthesis' in dataset:
        joint_dataset = gen_synthetic_data(*params)
        numerical = list(joint_dataset.columns)
        numerical.remove('label')
    elif 'adult' in dataset:
        # joint_dataset, numerical = load_adult_income_dataset('./data/adult.csv')
        joint_dataset = pd.read_csv('./data/adult.csv')
        numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'hours-per-week', 'capital-loss']
    elif 'compas' in dataset:
        joint_dataset = pd.read_csv('./data/compas.csv')
        numerical = ['age', 'two_year_recid', 'priors_count', 'length_of_stay']
    elif 'german' in dataset:
        dataset = pd.read_csv('./data/german_small.csv')
        shift_dataset = pd.read_csv('./data/corrected_german_small.csv')
        joint_dataset = dataset.append(shift_dataset)
        # numerical = ['Duration', 'Credit amount', 'Installment rate',
                     # 'Present residence', 'Age', 'Existing credits', 'Number people']
        numerical = ['Duration', 'Credit amount', 'Age']
    elif 'sba' in dataset:
        dataset = pd.read_csv('./data/sba_0614.csv')
        shift_dataset = pd.read_csv('./data/sba_8905.csv')
        joint_dataset = dataset.append(shift_dataset)
        categorical = ['LowDoc', 'RevLineCr', 'NewExist',
                       'MIS_Status', 'UrbanRural', 'label']
        numerical = list(dataset.columns.difference(categorical))
    elif 'student' in dataset:
        dataset = pd.read_csv('./data/gp_student.csv')
        shift_dataset = pd.read_csv('./data/ms_student.csv')
        joint_dataset = dataset.append(shift_dataset)
        numerical = ['Fedu', 'G1', 'G2', 'Medu', 'absences',
                     'age', 'freetime', 'goout', 'health', 'studytime']
    elif 'bank' in dataset:
        dataset = pd.read_csv('./data/bank.csv')
        shift_dataset = pd.read_csv('./data/bank_shift.csv')
        joint_dataset = dataset.append(shift_dataset)
        numerical = ['age', 'balance', 'campaign', 'previous']
    elif 'gmc' in dataset:
        joint_dataset = pd.read_csv('./data/gmc.csv')
        numerical = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
    elif 'heloc' in dataset:
        joint_dataset = pd.read_csv('./data/heloc.csv')
        numerical = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'NumTotalTrades', 'NumTradesOpeninLast12M', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance','NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']
    else:
        raise ValueError("Unknown dataset")

    return joint_dataset, numerical


def build_action_graph(data, labels, is_knn, n_neighbors, transformer, immutable_l=[]):
    # nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    # distances, indices = nbrs.kneighbors(data)

    if is_knn:
        graph = kneighbors_graph(data, n_neighbors=n_neighbors, mode='distance', include_self=True, n_jobs=-1)
    else:
        graph = radius_neighbors_graph(data, radius=n_neighbors, mode='distance', include_self=True, n_jobs=-1)
    
    # graph = nbrs.kneighbors_graph(data)

    # Prune all the set nodes 1
    weighted_graph = graph.toarray()
    all_edges_ = np.transpose(np.nonzero(weighted_graph))
    
    all_edges = []
    # weighted_graph = np.zeros(graph.shape)
    for edge in all_edges_:
        if labels[edge[0]] == 1 and labels[edge[1]] == 1:
            continue
        all_edges.append((edge[0], edge[1]))
    # for i in range(graph.shape[0]):
    #     for j in range(1, indices.shape[1]):
    #         if (indices[i][j], i) not in all_edges:
    #             if labels[i] == 1 and labels[indices[i][j]] == 1:
    #                 continue
    #             all_edges.append((i, indices[i][j]))
                # weighted_graph[i][indices[i][j]] = distances[i][j]
    
    # Actionability pruning
    all_edges_act = all_edges
    # for feature in immutable_l:
    df = transformer.inverse_transform(data)
    for edge in all_edges:
        n1, n2 = edge
        # d1 = transformer.inverse_transform(data[n1].reshape(1, -1))
        # d2 = transformer.inverse_transform(data[n2].reshape(1, -1))
        check = df[immutable_l].iloc[n1].to_numpy() == df[immutable_l].iloc[n2].to_numpy()
        # if d1[feature][0] != d2[feature][0]:
        if not np.all(check == True):
            all_edges_act.remove(edge)
    print(len(all_edges_act))
    # Graph construction
    nodes = set()
    for edge in all_edges_act:
        nodes.add(edge[0])
        nodes.add(edge[1])
    nodes = sorted(list(nodes))
    
    nodes_map = {}
    for i in range(len(nodes)):
        nodes_map[nodes[i]] = i
    
    data_nodes = data[nodes]
    labels_nodes = labels[nodes]
    adjacency_matrix = np.zeros((len(nodes), len(nodes)))
    weighted_adjacency_matrix = np.zeros((len(nodes), len(nodes)))
    
    for edge in all_edges_act:
        n1, n2 = nodes_map[edge[0]], nodes_map[edge[1]]
        adjacency_matrix[n1][n2] = 1
        weighted_adjacency_matrix[n1][n2] = weighted_graph[edge[0]][edge[1]]

    return data_nodes, labels_nodes, adjacency_matrix, weighted_adjacency_matrix


def make_logger(name, log_dir):
    log_dir = log_dir or 'logs'
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'debug.log')
    handler = logging.FileHandler(log_file)
    formater = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formater)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formater)
    logger.addHandler(stream_handler)

    return logger


def pdump(x, name, outdir='.'):
    with open(os.path.join(outdir, name), mode='wb') as f:
        pickle.dump(x, f)


def pload(name, outdir='.'):
    with open(os.path.join(outdir, name), mode='rb') as f:
        return pickle.load(f)
