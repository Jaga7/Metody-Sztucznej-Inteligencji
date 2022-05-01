import math
import numpy as np
from sklearn import datasets
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

clf= GaussianNB()
X, y = datasets.make_classification(
    n_samples=500,
)

# # normal_samples = np.random.normal(0, 1, np.shape(X[1]))

# # print(f"\n{normal_samples}")

# # X *= normal_samples[None, :]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.2,
)

# Bez feature selection
print("Bez feature selection")
print(X_train.shape)


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred),"\n")


# Feature selection: f_classif (ANOVA)
select = SelectKBest(k=2, score_func=f_classif)
select.fit(X_train, y_train)
print("Feature selection: f_classif (ANOVA)")

print(select.transform(X_train).shape)
X_test_f = select.transform(X_test)
X_train_f = select.transform(X_train)


clf.fit(X_train_f, y_train)
y_pred = clf.predict(X_test_f)
print(accuracy_score(y_test, y_pred),"\n")


# Feature selection: chi-square
select= make_pipeline(MinMaxScaler(), SelectKBest(k=2, score_func=chi2))
select.fit(X_train, y_train)
print("Feature selection: chi-square")

print(select.transform(X_train).shape)
X_test_chi = select.transform(X_test)
X_train_chi = select.transform(X_train)


clf.fit(X_train_chi, y_train)
y_pred = clf.predict(X_test_chi)
print(accuracy_score(y_test, y_pred),"\n")



# Feature selection: hellinger

def hellinger_explicit(p, q):
    """Hellinger distance between two discrete distributions.
       Same as original version but without list comprehension
    """


    list_of_squares = []
    for p_i, q_i in zip(p, q):

        # caluclate the square of the difference of ith distr elements
        s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2

        # append 
        list_of_squares.append(s)

    # calculate sum of squares
    sosq = sum(list_of_squares)    

    return sosq / math.sqrt(2)


select=  SelectKBest(k=2, score_func=hellinger_explicit)
# select= make_pipeline(MinMaxScaler(), SelectKBest(k=2, score_func=chi2))
select.fit(X_train, y_train)
print("Feature selection: Hellinger distance")

print(select.transform(X_train).shape)
X_test_hellinger = select.transform(X_test)
X_train_hellinger = select.transform(X_train)


clf.fit(X_train_hellinger, y_train)
y_pred = clf.predict(X_test_hellinger)
print(accuracy_score(y_test, y_pred),"\n")