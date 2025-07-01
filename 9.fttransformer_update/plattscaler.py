from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class PlattScaler(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = LogisticRegression(solver='lbfgs')

    def fit(self, logits, labels):
        logits = np.array(logits).reshape(-1, 1)
        self.model.fit(logits, labels)

    def predict_proba(self, logits):
        logits = np.array(logits).reshape(-1, 1)
        return self.model.predict_proba(logits)