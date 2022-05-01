import math
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


clf= GaussianNB()
X, y = datasets.make_classification(
    n_samples=500,
    n_features=12
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.2,
)

# print(X_train.shape)
# print(X.shape)
# print(np.histogram(X))


Xcale=np.histogram(X)


print("Całe X: ",np.histogram(X))

def hellinger_explicit(p, q):
    """Hellinger distance between two discrete distributions.
       Same as original version but without list comprehension
    """
    # p=np.histogram(p)
    # q=np.histogram(q)

    list_of_squares = []
    for p_i, q_i in zip(p, q):

        # caluclate the square of the difference of ith distr elements
        s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2

        # append 
        list_of_squares.append(s)

    # calculate sum of squares
    sosq = sum(list_of_squares)    

    return sosq / math.sqrt(2)


histCale, bin_edges_cale=Xcale
XClass1=np.histogram(X[0],bin_edges_cale)
XClass2=np.histogram(X[1],bin_edges_cale)

# print("Dla XClass1",XClass1)
# print("Dla XClass2",XClass2)

XClass1Hist,XClass1BinEdges=XClass1
XClass2Hist,XClass2BinEdges=XClass2
print(hellinger_explicit(XClass1Hist,XClass2Hist))