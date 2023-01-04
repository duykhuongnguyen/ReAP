import numpy as np

from sklearn.utils import check_random_state

from libs.roar.linear_roar import LinearROAR
from libs.explainers.lime_wrapper import LimeWrapper
from libs.explainers.clime import CLime 

from utils.visualization import visualize_explanations


def generate_recourse(x0, model, random_state, params=dict()):
    rng = check_random_state(random_state)

    train_data = params['train_data']
    ec = params['config']
    cat_indices = params['cat_indices']
    perturb_radius = params['perturb_radius']

    delta_max = ec.roar_params['delta_max']


    # print("="*10, "\n")
    # print("delta_max: ", delta_max)

    explainer = CLime(train_data, class_names=['0', '1'],
                      discretize_continuous=False, random_state=rng)

    # print("x0: ", x0)
    w, b = explainer.explain_instance(x0, model.predict_proba,
                                      perturbation_std=perturb_radius * ec.max_distance,
                                      num_samples=ec.num_samples)
    # print("w, b: ", w, b)

    arg = LinearROAR(train_data, w, b, cat_indices, lambd=0.1, dist_type=1,
                     lr=0.01, delta_max=delta_max, max_iter=1000)
    x_ar = arg.fit_instance(x0, verbose=False)
    report = dict(feasible=arg.feasible)

    # print("x_0: ", x0)
    # print("x_ar: ", x_ar)
    # print("logit: x_0", np.dot(x0, w) + b)
    # print("logit: x_ar", np.dot(x_ar, w) + b)
    # print("delta_max: ", delta_max, "dist: ", np.linalg.norm(x_ar-x0, 1),
          # "predict_prob: ", model.predict_proba(x_ar)[1])
    # print(arg.feasible)
    # print("="*10, "\n")
    # raise ValueError
    # visualize_explanations(model, lines=[(w, b)], x_test=x0, mean_pos=x_ar,
                           # xlim=(-2, 4), ylim=(-4, 7), save=True)
    return x_ar, report


def search_lambda(model, X, y, params, logger):
    lbd_list = np.arange(0.01, 0.1, 0.01)
    logger.info('ROAR: Search best lambda')

    y_pred = model.predict(X)
    uds_X = X[y_pred == 0]
    max_ins = 50
    uds_X = uds_X[:max_ins]

    logger.info('ROAR: cross_validation size: %d', len(uds_X))

    feasible = []
    best_lbd = 0
    best_sum_f = 0

    for lbd in lbd_list:
        logger.info('ROAR: try with lambda = %.2f', lbd)
        params['lambda'] = lbd
        sum_f = 0
        for x0 in uds_X:
            x_ar, _ = generate_recourse(x0, model, 1, params)
            sum_f += model.predict(x_ar)

        logger.info('ROAR: number of valid instances = %d', sum_f)

        if sum_f >= best_sum_f:
            best_sum_f, best_lbd = sum_f, lbd


    logger.info('ROAR: best lambda = %f', best_lbd)
    return best_lbd
