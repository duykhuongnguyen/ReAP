import numpy as np

from sklearn.ensemble import RandomForestClassifier


class RandomForest(RandomForestClassifier):
    def __init__(self, n):
        super(RandomForest, self).__init__(n_estimators=20,
                                           max_depth=5,
                                           random_state=123)

    def predict_proba(self, inp):
        if len(inp.shape) == 1:
            inp = inp.reshape(1, -1)
        ret = super(RandomForest, self).predict_proba(inp)
        return ret.squeeze()

    def predict(self, inp):
        if len(inp.shape) == 1:
            inp = inp.reshape(1, -1)
        ret = super(RandomForest, self).predict_proba(inp)
        ret = np.argmax(ret, axis=-1)
        return ret.squeeze()


def train(model, X, y, *args, **kwargs):
    model.fit(X, y)
