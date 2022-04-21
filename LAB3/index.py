

from matplotlib import pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn import clone, datasets
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from tabulate import tabulate


# zad. 2.1


class MojLosowy(BaseEstimator, ClassifierMixin):

    def __init__(self, metric='euclidean'):
        self.metric = metric

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes, licznoscKlas = np.unique(y, return_counts=True)
        self.X_, self.y_ = X, y
        self.prawdopodobienstwa = licznoscKlas/np.sum(licznoscKlas)

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        nwm = np.random.choice(
            self.classes, X.shape[0], p=self.prawdopodobienstwa)

        return nwm


X, y = datasets.make_classification(
    n_samples=100,
    n_informative=2,
    random_state=2024,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.2,
)

clf = MojLosowy()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("Zad. 2.1.")
print("Zwykly losowy:         %.3f" % accuracy_score(y_test, pred))


# zad. 2.2
class MyNearestNeighborClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, metric='euclidean'):
        self.metric = metric

    def fit(self, X, y):
        # czy X i y maja wlasciwy ksztalt
        X, y = check_X_y(X, y)
        # przechowanie unikalnych klas problemu
        self.classes, licznoscKlas = np.unique(y, return_counts=True)
        # przechowujemy X i y
        self.X_, self.y_ = X, y
        self.prawdopodobienstwa = licznoscKlas/np.sum(licznoscKlas)

        return self

    def predict(self, X):
        check_array(X)
        distancesToNeighbors = distance.cdist(self.X_, X, 'euclidean')
        distanceToNearestNeighbor = np.argmin(distancesToNeighbors, axis=0)
        return self.y_[distanceToNearestNeighbor]


X, y = datasets.make_classification(
    n_samples=100,
    n_informative=2,
    random_state=2024,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.2,
)

clfMKNN = MyNearestNeighborClassifier()
clfMKNN.fit(X_train, y_train)
predict = clfMKNN.predict(X_test)
score = accuracy_score(y_test, predict)
print(f"\nZad 2.2.\nMój:\t\t\t {score}")

clf = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
clf.fit(X_train, y_train)
predict = clf.predict(X_test)
score = accuracy_score(y_test, predict)
print(f"KNeighborsClassifier:\t {score}")


# zad. 2.3
clfs = {
    'RandomClassifier': MojLosowy(),
    'MykNN': KNeighborsClassifier(),
}

datasets = []
datasets.append([make_moons(), "moons"])
datasets.append([make_circles(), "circles"])
datasets.append([make_blobs(), "blobs"])


n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))

for data_id, dataset in enumerate(datasets):
    X, y = dataset[0]

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

print(f"\nZad.2.3. wyniki")

print(f"\nScores:\n{scores.shape}")

mean_scores = np.mean(scores, axis=2).T
std_scores = np.std(scores, axis=2).T
print("\nMean scores:\n", mean_scores)
print("\nStd values:\n", std_scores)

table = []

for data_id, dataset in enumerate(datasets):
    table.append([dataset[1], mean_scores[data_id, 0], std_scores[data_id, 0],
                 mean_scores[data_id, 1], std_scores[data_id, 1]])
table_headers = ['RandomClassifier mean',
                 'RandomClassifier std', 'MykNN mean', 'MykNN std']


print(tabulate(table, headers=table_headers, floatfmt=".2f"))
