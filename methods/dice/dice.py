import pandas as pd

import dice_ml

from methods.dice.dice_wrapper import DicePyTorchWrapper


def generate_recourse(x0, model, random_state, params=dict()):
    df = params['dataframe']
    numerical = params['numerical']
    k = params['k']

    full_dice_data = dice_ml.Data(dataframe=df,
                                  continuous_features=numerical,
                                  outcome_name='label')
    dice_model = dice_ml.Model(
        model=model, backend='PYT')
    dice = DicePyTorchWrapper(full_dice_data, dice_model)      

    df = df.drop(columns=['label'])
    keys = df.columns
    # d = {}
    # print(keys, x0)
    # for i in range(len(x0)):
    #     d[keys[i]] = [x0[i]]
    # d = pd.DataFrame.from_dict(d)
    plans = dice.generate_counterfactuals(x0, total_CFs=k,
                                          desired_class="opposite",
                                          posthoc_sparsity_param=None,
                                          proximity_weight=params['dice_params']['proximity_weight'],
                                          diversity_weight=params['dice_params']['diversity_weight']) 
    
    report = dict(feasible=True)
    # print(plans.final_cfs_df_sparse)
    return plans.cf_examples_list[0].final_cfs_df_sparse, report
    # return plans.final_cfs_df.drop(columns=['label']).to_numpy(), report
