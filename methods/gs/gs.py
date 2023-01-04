import numpy as np
import pandas as pd
from numpy import linalg as LA


def hyper_sphere_coordindates(n_search_samples, instance, high, low, p_norm=2):

    # Implementation follows the Random Point Picking over a sphere
    # The algorithm's implementation follows: Pawelczyk, Broelemann & Kascneci (2020);
    # "Learning Counterfactual Explanations for Tabular Data" -- The Web Conference 2020 (WWW)
    # It ensures that points are sampled uniformly at random using insights from:
    # http://mathworld.wolfram.com/HyperspherePointPicking.html

    # This one implements the growing spheres method from
    # Thibaut Laugel et al (2018), "Comparison-based Inverse Classification for
    # Interpretability in Machine Learning" -- International Conference on Information Processing
    # and Management of Uncertainty in Knowledge-Based Systems (2018)

    """
    :param n_search_samples: int > 0
    :param instance: numpy input point array
    :param high: float>= 0, h>l; upper bound
    :param low: float>= 0, l<h; lower bound
    :param p: float>= 1; norm
    :return: candidate counterfactuals & distances
    """

    delta_instance = np.random.randn(n_search_samples, instance.shape[1])
    dist = np.random.rand(n_search_samples) * (high - low) + low  # length range [l, h)
    norm_p = LA.norm(delta_instance, ord=p_norm, axis=1)
    d_norm = np.divide(dist, norm_p).reshape(-1, 1)  # rescale/normalize factor
    delta_instance = np.multiply(delta_instance, d_norm)
    candidate_counterfactuals = instance + delta_instance

    return candidate_counterfactuals, dist


def growing_spheres_search(
    instance,
    keys_mutable,
    keys_immutable,
    continuous_cols,
    binary_cols,
    feature_order,
    model,
    n_search_samples=1000,
    p_norm=2,
    step=0.01,
    max_iter=1000,
    K=3,
):

    """
    :param instance: df
    :param step: float > 0; step_size for growing spheres
    :param n_search_samples: int > 0
    :param model: sklearn classifier object
    :param p_norm: float=>1; denotes the norm (classical: 1 or 2)
    :param max_iter: int > 0; maximum # iterations
    :param keys_mutable: list; list of input names we can search over
    :param keys_immutable: list; list of input names that may not be searched over
    :return:
    """  #

    # correct order of names
    keys_correct = feature_order
    # divide up keys
    keys_mutable_continuous = list(set(keys_mutable) - set(binary_cols))
    keys_mutable_binary = list(set(keys_mutable) - set(continuous_cols))

    # Divide data in 'mutable' and 'non-mutable'
    # In particular, divide data in 'mutable & binary' and 'mutable and continuous'
    instance_immutable_replicated = np.repeat(
        instance[keys_immutable].values.reshape(1, -1), n_search_samples, axis=0
    )
    instance_replicated = np.repeat(
        instance.values.reshape(1, -1), n_search_samples, axis=0
    )
    instance_mutable_replicated_continuous = np.repeat(
        instance[keys_mutable_continuous].values.reshape(1, -1),
        n_search_samples,
        axis=0,
    )
    # instance_mutable_replicated_binary = np.repeat(
    #     instance[keys_mutable_binary].values.reshape(1, -1), n_search_samples, axis=0
    # )

    # init step size for growing the sphere
    low = 0
    high = low + step

    # counter
    count = 0
    counter_step = 1

    # get predicted label of instance
    instance_label = np.argmax(model.predict_proba(instance.values))

    counterfactuals_found = False
    candidate_counterfactual_star = np.empty(
        instance_replicated.shape[1],
    )
    candidate_counterfactual_star[:] = np.nan
    while (not counterfactuals_found) and (count < max_iter):
        count = count + counter_step

        # STEP 1 -- SAMPLE POINTS on hyper sphere around instance
        candidate_counterfactuals_continuous, _ = hyper_sphere_coordindates(
            n_search_samples, instance_mutable_replicated_continuous, high, low, p_norm
        )

        # sample random points from Bernoulli distribution
        candidate_counterfactuals_binary = np.random.binomial(
            n=1, p=0.5, size=n_search_samples * len(keys_mutable_binary)
        ).reshape(n_search_samples, -1)

        # make sure inputs are in correct order
        candidate_counterfactuals = pd.DataFrame(
            np.c_[
                instance_immutable_replicated,
                candidate_counterfactuals_continuous,
                candidate_counterfactuals_binary,
            ]
        )
        candidate_counterfactuals.columns = (
            keys_immutable + keys_mutable_continuous + keys_mutable_binary
        )
        # enforce correct order
        candidate_counterfactuals = candidate_counterfactuals[keys_correct]

        # STEP 2 -- COMPUTE l_1 DISTANCES
        if p_norm == 1:
            distances = np.abs(
                (candidate_counterfactuals - instance_replicated)
            ).sum(axis=1)
        elif p_norm == 2:
            distances = np.square(
                (candidate_counterfactuals - instance_replicated)
            ).sum(axis=1)
        else:
            raise ValueError("Distance not defined yet")

        # counterfactual labels
        y_candidate_logits = model.predict_proba(candidate_counterfactuals.values)
        y_candidate = np.argmax(y_candidate_logits, axis=1)
        indeces = np.where(y_candidate != instance_label)
        candidate_counterfactuals = candidate_counterfactuals.values[indeces]
        candidate_dist = distances.values[indeces]
        
        if len(candidate_dist) >= K:  # certain candidates generated
            # min_index = np.argmin(candidate_dist)
            min_index = np.argsort(candidate_dist)[:K]
            candidate_counterfactual_star = candidate_counterfactuals[min_index]
            counterfactuals_found = True
    
        # no candidate found & push search range outside
        low = high
        high = low + step
    
    return candidate_counterfactual_star, counterfactuals_found


def generate_recourse(x0, model, random_state, params=dict()):
    df = params['dataframe']
    numerical = params['numerical']
    k = params['k']

    df = df.drop(columns=['label'])
    keys = df.columns

    x0_shape = x0.shape[0]
    mutables = numerical 
    feature_input_order = numerical + [i for i in range(x0_shape - len(numerical))]
    immutables = [i for i in range(x0_shape - len(numerical))]
    continuous = numerical
    categorical = [i for i in range(x0_shape - len(numerical))]

    df_x0 = pd.Series(x0, index=feature_input_order)

    cf_found = False
    step_init = 0.0001
    while not cf_found:
        cf, found = growing_spheres_search(
            df_x0,
            mutables,
            immutables,
            continuous,
            categorical,
            feature_input_order,
            model,
            k,
            step=step_init,
        )
        step_init *= 10
        if found:
            cf_found = True
    
    report = dict(feasible=True)
    
    return cf, report
