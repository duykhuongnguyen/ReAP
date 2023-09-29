import os
import argparse
from expt.expt_config import ExptConfig
from expt.expt_1 import run_expt_1
from expt.expt_2 import run_expt_2
from expt.run_time import run_expt_run_time
from utils.helpers import make_logger
import logging
import matplotlib
import numpy as np
import torch
import random

logging.getLogger('matplotlib').setLevel(logging.ERROR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run experiments')
    parser.add_argument('--run-id', '-rid', default=0, type=int)
    parser.add_argument('--ept', '-e', dest='epts',
                        action='append', required=True)
    parser.add_argument('--config', '-cf', default=None, type=str)
    parser.add_argument('--datasets', '-d', dest='datasets', nargs='*')
    parser.add_argument('--methods', '-m', dest='methods', nargs='*')
    parser.add_argument('--classifiers', '-clf', dest='classifiers', nargs='*')
    parser.add_argument('--num-proc', '-np', default=1, type=int)
    parser.add_argument('--start-index', '-id', default=None, type=int)
    parser.add_argument('--num-ins', default=None, type=int)
    parser.add_argument('--update-config', '-uc', action='store_true')
    parser.add_argument('--plot-only', '-po', action='store_true')
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--seed', '-s', default=124, type=int)

    args = parser.parse_args()

    save_dir = f'results/run_{args.run_id}'
    config_path = os.path.join(save_dir, 'config.yml')

    os.makedirs(save_dir, exist_ok=True)

    np.random.seed(args.seed - 1)
    torch.manual_seed(args.seed - 2)
    random.seed(args.seed - 3)
    np.set_printoptions(suppress=True)

    ec = ExptConfig()
    if args.update_config or not os.path.isfile(config_path):
        ec.to_file(config_path, mode='merge_cls')
    else:
        ec.from_file(config_path)

    for ept in set(args.epts):
        ept_dir = os.path.join(save_dir, f'expt_{ept}')
        os.makedirs(ept_dir, exist_ok=True)
        logger = make_logger(f'expt_{ept}', ept_dir)

        if ept == "1":                                                                             
            run_expt_1(ec, ept_dir, datasets=args.datasets, methods=args.methods, rerun=args.rerun,
                       classifiers=args.classifiers, num_proc=args.num_proc,                       
                       plot_only=args.plot_only, seed=args.seed, logger=logger)                    
        if ept == '2':                                                                             
            run_expt_2(ec, ept_dir, datasets=args.datasets, methods=args.methods, rerun=args.rerun,
                       classifiers=args.classifiers, num_proc=args.num_proc,                       
                       plot_only=args.plot_only, seed=args.seed, logger=logger,                    
                       start_index=args.start_index, num_ins=args.num_ins)                         
        if ept == 'run_time':
            run_expt_run_time(None, None, None, None, None, logger=logger)
