from sklearn.ensemble import BaggingClassifier, BaseEnsemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, clone
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from scipy.stats import rankdata
from scipy.stats import ranksums
from tabulate import tabulate
from rse import RandomSubspaceEnsemble


class MyBaggingEnsemble(BaseEnsemble, ClassifierMixin):
    def __init__(self, random_state=None,hard_voting=True, useWeights=False,base_estimator = DecisionTreeClassifier()):
        self.base_estimator = base_estimator
        self.estimators = 5

        self.hard_voting=hard_voting
        self.useWeights = useWeights
        
        self.random_state = random_state
        # self.hard_voting = hard_voting
        np.random.seed(self.random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_samples = X.shape[0]
        self.max_samples=round((0.8*self.n_samples))
        
        self.classes = np.unique(y)
        self.bags = np.random.randint(0, self.n_samples, (self.estimators, self.max_samples))
        self.ensemble_ = []
        if(self.useWeights):
            self.weights = np.zeros(self.estimators)
            for i in range(self.estimators):
                self.ensemble_.append(clone(self.base_estimator).fit(X[self.bags[i],: ], y[self.bags[i]]))
                self.weights[i] = accuracy_score(self.ensemble_[i].predict(X), y)
        else:
            for i in range(self.estimators):
                self.ensemble_.append(clone(self.base_estimator).fit(X[self.bags[i],: ], y[self.bags[i]]))
        return self

    def predict(self, X):
        check_is_fitted(self, 'classes')

        X = check_array(X)

        if self.hard_voting:

            if self.useWeights:
                predict_table = []
                for i, member_clf in enumerate(self.ensemble_):
                    predict_table.append(member_clf.predict(X))

                predict_table = np.array(predict_table)

                class_pred = np.zeros(predict_table.T.shape[0], dtype=int)
                
                for i, row in enumerate(predict_table.T):
                    t = row*self.weights
                    
                    # if np.sum(t) >= (self.estimators/2):
                    #     class_pred[i] = 1
                    # else:
                    #     class_pred[i] = 0
                    
                    class_pred[i]=np.around(np.sum(t)/self.estimators)
                    
                    # class_pred[i] = int(class_pred[i])
                return self.classes[class_pred]
            else:

                predict_table = []
                for i, member_clf in enumerate(self.ensemble_):
                    predict_table.append(member_clf.predict(X))
                predict_table = np.array(predict_table)
                class_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predict_table.T)
                return self.classes[class_pred] 
        else:
            # Podejmowanie decyzji na podstawie wektorow wsparcia
            esm = self.ensemble_support_matrix(X)
            # Wyliczenie sredniej wartosci wsparcia
            average_support = np.mean(esm, axis=0)
            # Wskazanie etykiet
            prediction = np.argmax(average_support, axis=1)
            # Zwrocenie predykcji calego zespolu
            return self.classes[prediction]  
                

 
    def ensemble_support_matrix(self, X):
        # Wyliczenie macierzy wsparcia
        probas_ = []
        for i, member_clf in enumerate(self.ensemble_):
            probas_.append(member_clf.predict_proba(X))
            if self.useWeights:
                probas_[i] = probas_[i]*self.weights[i]
        return np.array(probas_)

          

dataset = 'cryotherapy'
dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

clfs = {
    'hard_no_weights': MyBaggingEnsemble(random_state=1420,useWeights=False,hard_voting=True),
    'hard_weights': MyBaggingEnsemble(random_state=1420,useWeights=True,hard_voting=True),
    'soft_no_weights': MyBaggingEnsemble(random_state=1420,useWeights=False,hard_voting=False),
    'soft_weights':MyBaggingEnsemble(random_state=1420,useWeights=True,hard_voting=False),
    'rse_hard':RandomSubspaceEnsemble(base_estimator=DecisionTreeClassifier(), n_estimators=5,hard_voting=True, random_state=1420),
    'rse_soft':RandomSubspaceEnsemble(base_estimator=DecisionTreeClassifier(), n_estimators=5, hard_voting=False, random_state=1420)
}

# Zad. 1.
print("\nZad.1.:\n")
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1420)
bagging = clfs['hard_no_weights']
cart = DecisionTreeClassifier(random_state=1420)
bagging_scores = []
cart_scores = []

for train, test in rskf.split(X, y):
    cart.fit(X[train], y[train])
    bagging.fit(X[train], y[train])
    y_bagging_pred = bagging.predict(X[test])
    y_cart_pred = cart.predict(X[test])
    bagging_scores.append(accuracy_score(y[test], y_bagging_pred))
    cart_scores.append(accuracy_score(y[test], y_cart_pred))
print("Bagging: %.3f " %  np.mean(bagging_scores))
print("CART: %.3f " %  np.mean(cart_scores))


# Zad.2. i Zad.3.
print("\nZad.2.")
datasets = ['australian', 'balance', 'breastcan', 'cryotherapy', 'diabetes',
            'digit', 'ecoli4', 'german', 'glass2', 'heart', 'ionosphere',
            'liver', 'monkthree', 'shuttle-c0-vs-c4', 'sonar', 'soybean',
            'vowel0', 'waveform', 'wisconsin', 'yeast3']
n_datasets = len(datasets)

scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

mean_scores = np.mean(scores, axis=2).T

ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)
print("\nRanks:\n", ranks)

mean_ranks = np.mean(ranks, axis=0)
print("\nMean ranks:\n", mean_ranks)

alfa = .05
w_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

headers = list(clfs.keys())
names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)


advantage = np.zeros((len(clfs), len(clfs)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nAdvantage:\n", advantage_table)

significance = np.zeros((len(clfs), len(clfs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
print("\nStatistically significantly better:\n", stat_better_table)

Zad. 3.
print("\n\nZad.3.:\n")

rse=RandomSubspaceEnsemble(base_estimator=DecisionTreeClassifier(), n_estimators=5, n_subspace_features=5, hard_voting=True, random_state=123)

