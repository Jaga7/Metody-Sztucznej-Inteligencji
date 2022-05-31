from sklearn.datasets import make_classification
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.base import ClassifierMixin,BaseEstimator 
import numpy as np

class MySamplingClassifier(BaseEstimator , ClassifierMixin):
    def __init__(self, base_estimator,base_preprocessing=None):
        self.base_estimator = base_estimator
        if base_preprocessing:
            self.base_preprocessing=base_preprocessing
        else:
            self.base_preprocessing=base_estimator

    def fit(self, X, y):
         # czy X i y maja wlasciwy ksztalt
        X, y = check_X_y(X, y)
        # przechowanie unikalnych klas problemu
        self.classes_ = np.unique(y)
        # przechowujemy X i y
        self.X_, self.y_ = X, y
        if self.base_preprocessing==self.base_estimator:
            self.base_preprocessing.fit(self.X_, self.y_)
            self.base_estimator.fit(self.X_, self.y_)
        else:
            self.X_resampled, self.y_resampled = self.base_preprocessing.fit_resample(self.X_, self.y_)
            self.base_estimator.fit(self.X_resampled, self.y_resampled)

        return self

    def predict(self, X):
        check_is_fitted(self, 'classes_')

        X = check_array(X)

        class_pred = self.base_estimator.predict(X)
        return class_pred

