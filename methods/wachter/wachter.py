from methods.wachter.gd import gd


def generate_recourse(x0, model, random_state, params=dict()):
    # General parameters
    cat_indices = params['cat_indices']

    # Wachter parameters
    lr = params['wachter_params']['lr']
    lmbda = params['wachter_params']['lmbda']
    
    # Recourse generation
    recourse, feasible = gd(model, x0, cat_indices, binary_cat_features=True, lr=lr, lambda_param=lmbda, y_target=[0, 1], n_iter=1000, t_max_min=1000, norm=1, clamp=True, loss_type="BCE")

    return recourse, feasible
