import math
import numpy as np
from sklearn import datasets
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score,f1_score,precision_score
from imblearn.metrics import geometric_mean_score,specificity_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from Hellinger import hellinger

clf= GaussianNB()

score_functions= {
    "chi2": chi2,
    "f_classif": f_classif,
    "hellinger": hellinger 
}

n_score_functions=len(score_functions)

datasets = ['cleveland_0_vs_4', 'segment0', 'winequality_red_3_vs_5', 'winequality_red_4', 'winequality_red_8_vs_6',
            'winequality_red_8_vs_6_7', 'winequality_white_3_9_vs_5', 'winequality_white_3_vs_7', 'winequality_white_9_vs_4','page_blocks_1_3_vs_4']

n_datasets = len(datasets)

metrics = {
    "accuracy":accuracy_score,
    "recall": recall_score,
    'precision': precision_score,
    'specificity': specificity_score,
    'f1': f1_score,
    'g-mean': geometric_mean_score,
    'bac': balanced_accuracy_score,
}

# metrics = {
#     "accuracy": accuracy_score,
#     "recall": recall_score,
# }
n_metrics=len(metrics)


n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

scores=np.zeros((n_score_functions,n_datasets,n_metrics, n_splits * n_repeats))

for data_id, dataset in enumerate(datasets):
    print("aaa",dataset," bbb ","datasets/%s.csv" % (dataset))
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=";")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for score_fun_id, score_fun_name in enumerate(score_functions):
            if(score_fun_name=="chi2"):
                select=make_pipeline(MinMaxScaler(), SelectKBest( score_func=chi2,k=5))
            else:
                select=SelectKBest(score_functions[score_fun_name],k=5)

            select.fit(X[train], y[train])

            X_test_transformed = select.transform(X[test])
            X_train_transformed = select.transform(X[train]) 
            clf.fit(X_train_transformed, y[train])  

            y_pred = clf.predict(X_test_transformed)

            # scores[score_fun_id, data_id, fold_id] = accuracy_score(y[test], y_pred)
            for metric_id, metric in enumerate(metrics):
                scores[score_fun_id,data_id, metric_id,fold_id] = metrics[metric](
                y[test], y_pred)

# Zapisanie wynikow
np.save('results', scores)




# for data_id, dataset in enumerate(datasets):
#     dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
#     X = dataset[:, :-1]
#     y = dataset[:, -1].astype(int)

#     for fold_id, (train, test) in enumerate(rskf.split(X, y)):
#         for clf_id, clf_name in enumerate(clfs):
#             clf = clone(clfs[clf_name])
#             clf.fit(X[train], y[train])
#             y_pred = clf.predict(X[test])
#             scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

# /////////////////////////////////////////////////////

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=.2,
# )
# # Bez feature selection
# print("Bez feature selection")
# print(X_train.shape)


# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(accuracy_score(y_test, y_pred),"\n")


# # Feature selection: f_classif (ANOVA)
# select = SelectKBest(k=2, score_func=f_classif)
# select.fit(X_train, y_train)
# print("Feature selection: f_classif (ANOVA)")

# print(select.transform(X_train).shape)
# X_test_f = select.transform(X_test)
# X_train_f = select.transform(X_train)


# clf.fit(X_train_f, y_train)
# y_pred = clf.predict(X_test_f)
# print(accuracy_score(y_test, y_pred),"\n")


# # Feature selection: chi-square
# select= make_pipeline(MinMaxScaler(), SelectKBest(k=2, score_func=chi2))
# select.fit(X_train, y_train)
# print("Feature selection: chi-square")

# print(select.transform(X_train).shape)
# X_test_chi = select.transform(X_test)
# X_train_chi = select.transform(X_train)


# clf.fit(X_train_chi, y_train)
# y_pred = clf.predict(X_test_chi)
# print(accuracy_score(y_test, y_pred),"\n")

# # Feature selection: hellinger
# select=  SelectKBest(k=2, score_func=hellinger)
# # select= make_pipeline(MinMaxScaler(), SelectKBest(k=2, score_func=chi2))
# select.fit(X_train, y_train)
# print("Feature selection: Hellinger distance")

# print(select.transform(X_train).shape)
# X_test_hellinger = select.transform(X_test)
# X_train_hellinger = select.transform(X_train)


# clf.fit(X_train_hellinger, y_train)
# y_pred = clf.predict(X_test_hellinger)
# print(accuracy_score(y_test, y_pred),"\n")