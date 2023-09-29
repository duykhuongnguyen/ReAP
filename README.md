# ReAP

Code for submission: "Cost Adaptive Recourse Recommendation by Adaptive Preference Elicitation" (ReAP)

## Usage

1. Train MLP classifiers

```sh
python train_model.py --clf mlp --data synthesis german sba bank adult --num-proc 16
```

2. Run experiments

Gradient-based recourse (The saved files for experiment 1 could be used in experiment 2)

```sh                                             
python run_expt.py -e 1 --datasets synthesis german bank student -clf mlp --methods dice wachter reup -uc

python run_expt.py -e 2 --datasets synthesis german bank student -clf mlp --methods reup -uc
```
Graph-based recourse:

```sh                                             
python run_expt.py -e 1 --datasets synthesis german bank student -clf mlp --method reup_graph -uc 
```
