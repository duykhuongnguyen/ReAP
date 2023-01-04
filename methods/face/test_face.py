from carla import DataCatalog, MLModelCatalog
from carla.recourse_methods import Face


data_name = "adult"
dataset = DataCatalog(data_name)
factuals = dataset.raw.iloc[:10]

model = MLModelCatalog(dataset, "ann", backend="pytorch")

face = Face(model, {"mode": "knn", "fraction": 0.1})
counterfactuals = face.get_counterfactuals(factuals)
print("Factuals")
print(factuals.iloc[0])

print("Counterfactuals")
print(counterfactuals.iloc[0])
