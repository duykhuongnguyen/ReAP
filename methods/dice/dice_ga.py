import pandas as pd

import dice_ml
from libs.dice.dice_genetic import DiceGenetic
from libs.dice.dice_wrapper import DicePyTorchWrapper


def generate_recourse(x0, model, random_state, params=dict()):
    df = params['dataframe']
    numerical = params['numerical']
    k = params['k']
    transformer = params['transformer']

    full_dice_data = dice_ml.Data(dataframe=df,
                                  continuous_features=numerical,
                                  outcome_name='label')
    dice_model = dice_ml.Model(
        model=model, backend='PYT')
    dice = DiceGenetic(full_dice_data, dice_model, x0)      

    df = df.drop(columns=['label'])
    keys = df.columns
    
    plans = dice._generate_counterfactuals(x0, total_CFs=k,
                                          desired_class="opposite",
                                          posthoc_sparsity_param=None,
                                          proximity_weight=params['dice_params']['proximity_weight'],
                                          diversity_weight=params['dice_params']['diversity_weight']) 
    
    report = dict(feasible=True)
    
    return plans.cf_examples_list[0].final_cfs_df_sparse, report
