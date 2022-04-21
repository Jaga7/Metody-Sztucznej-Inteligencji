

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from random import random
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn import clone, datasets
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from tabulate import tabulate


X, y = datasets.make_classification(
    n_samples=500,
    random_state=1024
)
print(f"{np.shape(X)}")
print(f"{np.shape(X[1])}")


normal_samples = np.random.normal(0, 1, np.shape(X[1]))

print(f"\n{normal_samples}")

X *= normal_samples[None, :]


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=.2,
# )

clfs = {
    'GNB': GaussianNB(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=1234),
}

n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

scores = np.zeros((len(clfs), n_splits * n_repeats))

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for clf_id, clf_name in enumerate(clfs):
        clf = clone(clfs[clf_name])
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)


print(f"\nZad.3.1. wyniki")

print(f"\nScores shape:\n{scores.shape}")

print(f"\nScores:\n{scores}")

mean_scores = np.mean(scores, axis=1).T
std_scores = np.std(scores, axis=1).T
print("\nMean scores:\n", mean_scores)
print("\nStd values:\n", std_scores)

table = []

table.append([mean_scores[0], std_scores[0],
              mean_scores[1], std_scores[1],
             mean_scores[2], std_scores[2]])

table_headers = ['GNB mean',
                 'GNB std', 'kNN mean', 'kNN std', 'CART mean', 'CART std']


print(tabulate(table, headers=table_headers, floatfmt=".3f"))


# Zad. 3.2.
print(f"\n\nZad. 3.2.")
scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X)

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for clf_id, clf_name in enumerate(clfs):
        clf = clone(clfs[clf_name])
        X_train_scaled = scaler.fit_transform(X[train])
        print(f"{X_train_scaled}")
        clf.fit(X_train_scaled, y[train])
        # clf.fit(X_train_scaled[train], y[train])
        X_test_scaled = scaler.transform(X[test])
        y_pred = clf.predict(X_test_scaled)
        scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)


print(f"\nZad.3.2. wyniki")

print(f"\nScores shape:\n{scores.shape}")

print(f"\nScores:\n{scores}")

mean_scores = np.mean(scores, axis=1).T
std_scores = np.std(scores, axis=1).T
print("\nMean scores:\n", mean_scores)
print("\nStd values:\n", std_scores)

table2 = []

table2.append([mean_scores[0], std_scores[0],
              mean_scores[1], std_scores[1],
               mean_scores[2], std_scores[2]])

table2_headers = ['GNB mean',
                  'GNB std', 'kNN mean', 'kNN std', 'CART mean', 'CART std']


print(tabulate(table2, headers=table2_headers, floatfmt=".3f"))

# Zad.3.3.
print(f"\n\nZad. 3.3.")

pca = PCA(svd_solver='full')


for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for clf_id, clf_name in enumerate(clfs):
        clf = clone(clfs[clf_name])

        X_train_scaled = scaler.fit_transform(X[train])
        X_test_scaled = scaler.transform(X[test])

        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        accumulated_sum = np.cumsum(pca.explained_variance_ratio_)
        attributes_mask = accumulated_sum < .8
        X_pca_train = X_train_scaled[:, attributes_mask]
        X_pca_test = X_test_scaled[:, attributes_mask]

        clf.fit(X_pca_train, y[train])
        y_pred = clf.predict(X_pca_test)
        scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)

print(f"\nZad.3.3. wyniki")

print(f"\nScores shape:\n{scores.shape}")

print(f"\nScores:\n{scores}")

mean_scores = np.mean(scores, axis=1).T
std_scores = np.std(scores, axis=1).T
print("\nMean scores:\n", mean_scores)
print("\nStd values:\n", std_scores)

table3 = []

table3.append([mean_scores[0], std_scores[0],
              mean_scores[1], std_scores[1],
               mean_scores[2], std_scores[2]])

table3_headers = ['GNB mean',
                  'GNB std', 'kNN mean', 'kNN std', 'CART mean', 'CART std']


print(tabulate(table3, headers=table3_headers, floatfmt=".3f"))
