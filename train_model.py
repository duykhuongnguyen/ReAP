import argparse
import os
import joblib
import torch
import numpy as np
import yaml
import sklearn

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, \
    classification_report, confusion_matrix, roc_auc_score

import dice_ml

from utils.transformer import get_transformer
from utils.data_transformer import DataTransformer
from utils import helpers
from classifiers import mlp
from expt.common import clf_map, synthetic_params, train_func_map


arrival_data = {
    'synthesis': False,
    'german': False,
    'sba': False,
    'bank': False,
    'student': False,
}


def eval_performance(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)
    y_pred = np.argmax(y_prob, axis=-1)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob[:, 1])

    return accuracy, auc


def train_model(X_train, y_train, X_test, y_test, train_func, clf, d, lr, num_epoch, verbose, random_state):
    print("training classifier")
    torch.manual_seed(random_state)
    np.random.seed(random_state+1)
    model = clf(d)
    train_func(model, X_train, y_train, lr, num_epoch, verbose)
    acc, auc = eval_performance(model, X_test, y_test)
    return model, acc, auc


def train(clf_name, data_name, wdir, lr, num_epoch, seed=123, verbose=False, num_proc=1):
    arrival_ratio = 0.20
    train_shift_size = 0.8
    append_arrival = False 

    # transformer = get_transformer(data_name)
    full_l = ["synthesis", "german"]
    df, numerical = helpers.get_full_dataset(data_name, params=synthetic_params) if data_name not in full_l else helpers.get_dataset(data_name, params=synthetic_params)
    full_dice_data = dice_ml.Data(dataframe=df,
                         continuous_features=numerical,
                         outcome_name='label')
    transformer = DataTransformer(full_dice_data)

    y = df['label'].to_numpy()
    X = df.drop('label', axis=1)
    X = transformer.transform(X)
    X = X.to_numpy()

    d = X.shape[1]
    clf = clf_map[clf_name]
    train_func = train_func_map[clf_name]
    report = {}

    # Train present data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=42, stratify=y)

    model, acc, auc = train_model(X_train, y_train, X_test, y_test, train_func, clf, d, lr, num_epoch, verbose, seed)

    report['accuracy'] = acc
    report['auc'] = auc
    name = f"{clf_name}_{data_name}.pickle"
    helpers.pdump(model, name, wdir)
    print(report)
    print("Trained classifier: {} on current dataset: {}, and saved to {}".format(
        clf_name, data_name, os.path.join(wdir, name)))

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument('--clf', '-c', dest='clfs', nargs='*')
    parser.add_argument('--data', '-d', dest='datasets', nargs='*')
    parser.add_argument('--lr', '-lr', default=1e-3, type=float)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--num-proc', default=1, type=int)
    parser.add_argument('--run-id', default=0, type=int)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--seed', '-s', default=123, type=int)

    args = parser.parse_args()

    torch.set_printoptions(sci_mode=False)
    seed = 46
    torch.manual_seed(args.seed + 12)
    np.random.seed(args.seed + 11)
    np.set_printoptions(suppress=False)
    wdir = f"results/run_{args.run_id}/checkpoints"
    os.makedirs(wdir, exist_ok=True)

    report = {}
    for clf in args.clfs:
        clf_report = {}
        for data in args.datasets:
            print("training on dataset: ", data)
            data_report = train(clf, data, wdir,
                                args.lr, args.epoch,
                                args.seed, args.verbose, args.num_proc)
            clf_report[data] = data_report
        report[clf] = clf_report

    filepath = f"{wdir}/report.txt"
    with open(filepath, mode='w') as file:
        yaml.dump(report, file)
