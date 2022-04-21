from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn import svm


x, y = datasets.make_classification(
    n_samples=400,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    weights=None,
    flip_y=0.08,
    class_sep=1.0,
    hypercube=True,
    shift=0.0,
    scale=1.0,
    shuffle=0,
    random_state=1
)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap="bwr", edgecolors='r')
plt.xlabel("$1")
plt.ylabel("$2")
plt.show()
plt.savefig('scatterplot.png')

dataset = np.concatenate((x, y[:, np.newaxis]), axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

model = GaussianNB()
model.fit(X_train, y_train)
prediction = model.predict_proba(X_test)
arcmax = np.argmax(prediction, axis=1)
print("Accuracy:",accuracy_score(y_test, arcmax))
print(prediction)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print(clf.score(X_test, y_test))
