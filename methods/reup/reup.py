from methods.reup.chebysev import chebysev_center
from methods.reup.q_determine import exhaustive_search, find_q
from methods.reup.gd import gd


def generate_recourse(x0, model, random_state, params=dict()):
    # General parameters
    train_data = params['train_data']
    labels = params['labels']
    data = train_data[labels == 1]

    cat_indices = params['cat_indices']

    # ReUP parameters
    T = params['reup_params']['T']
    epsilon = params['reup_params']['eps']

    lr = params['reup_params']['lr']
    lmbda = params['reup_params']['lmbda']
    
    # Questions generation
    P, A_opt, mean_rank = find_q(x0, data, T, params['A'], epsilon, True)

    # Recourse generation
    recourse, feasible = gd(model, x0, cat_indices, binary_cat_features=True, lr=lr, lambda_param=lmbda, y_target=[1], n_iter=1000, t_max_min=1000, norm=1, clamp=True, loss_type="MSE", P=P, epsilon=epsilon)

    return recourse, mean_rank, feasible
