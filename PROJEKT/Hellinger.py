import math
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


clf= GaussianNB()
X, y = datasets.make_classification(
    n_samples=500,
    n_features=12,
    random_state=2140
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=.2,
)

# print(X_train.shape)
# print(X.shape)
# print(np.histogram(X))


Xcale=np.histogram(X)


# print("Całe X: ",np.histogram(X))

# def hellinger_explicit(p, q):
#     """Hellinger distance between two discrete distributions.
#        Same as original version but without list comprehension
#     """
#     # p=np.histogram(p)
#     # q=np.histogram(q)


#     list_of_squares = []
#     for p_i, q_i in zip(p, q,strict=True):

#         # caluclate the square of the difference of ith distr elements
#         s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2

#         # append 
#         list_of_squares.append(s)

#     # calculate sum of squares
#     sosq = sum(list_of_squares)    

#     return sosq / math.sqrt(2)


# def hellinger_explicit(X, y):
#     """Hellinger distance between two discrete distributions.
#        Same as original version but without list comprehension
#     """
#     # p=np.histogram(p)
#     # q=np.histogram(q)
    

#     # To poniżej to raczej ma być pętla i takie coś dla każdego atrybutu
#     XHist,XBinEdges=np.histogram(X)
#     XClass1=np.histogram(X[0],XBinEdges)
#     XClass2=np.histogram(X[1],XBinEdges)
#     XClass1Hist,XClass1BinEdges=XClass1
#     XClass2Hist,XClass2BinEdges=XClass2
#     XClass1NormalizedFrequencies=XClass1Hist/sum(XClass1Hist)
#     XClass2NormalizedFrequencies=XClass2Hist/sum(XClass2Hist)

#     list_of_squares = []
#     for p_i, q_i in zip(XClass1NormalizedFrequencies, XClass2NormalizedFrequencies,strict=True):

#         # caluclate the square of the difference of ith distr elements
#         s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2

#         # append 
#         list_of_squares.append(s)

#     # calculate sum of squares
#     sosq = sum(list_of_squares)    

#     return sosq / math.sqrt(2)


histCale, bin_edges_cale=Xcale
XAttribute1=np.histogram(X[:,0],bin_edges_cale)
XAttribute2=np.histogram(X[:,1],bin_edges_cale)

# print("Dla XAttribute1",XAttribute1)
# print("Dla XAttribute2",XAttribute2)

XAttribute1Hist,XAttribute1BinEdges=XAttribute1
XAttribute2Hist,XAttribute2BinEdges=XAttribute2
XAttribute1NormalizedFrequencies=XAttribute1Hist/sum(XAttribute1Hist)
XAttribute2NormalizedFrequencies=XAttribute2Hist/sum(XAttribute2Hist)
# print(XAttribute1Hist)
# print(sum(XAttribute1Hist))
# print(XAttribute1P)

# print(X[0].shape[0])

# print(X[:,0])

# for idxAttribute in range(0,X[0].shape[0]-1):
#     for idxSample, value in enumerate(X[:,0]):
#         print(idxSample, value)


# stworzenie 3-wymiarowej macierzy, klasy, atrybuty, wartości w binach
valuesOfAttributes = np.zeros((2, X[0].shape[0], len(bin_edges_cale)-1))



# Służy do zapisania wartości dla różnych klas i zrobienia z nich histogramów
valuesOfAttributesForClass0=[]
valuesOfAttributesForClass1=[]


#iterowanie przez wartości atrybutu o indeksie 0 = XAttribute1, zapisanie do tablicy i tu powinno się 2 histogramy z nich
for idxSample, value in enumerate(X[:,0]):
    if y[idxSample]==0:valuesOfAttributesForClass0.append(value)
    if y[idxSample]==1:valuesOfAttributesForClass1.append(value)
    if(idxSample==len(X[:,0])-1): 
        XClass0=np.histogram(valuesOfAttributesForClass0,bin_edges_cale)
        XClass1=np.histogram(valuesOfAttributesForClass1,bin_edges_cale)
        XClass0Hist,XClass0BinEdges=XClass0
        XClass1Hist,XClass1BinEdges=XClass1
        XClass0NormalizedFrequencies=XClass0Hist/sum(XClass0Hist)
        XClass1NormalizedFrequencies=XClass1Hist/sum(XClass1Hist)
        # zapisanie prawdopodobieństw rozkładów do macierzy
        valuesOfAttributes[0,0]=XClass0NormalizedFrequencies
        valuesOfAttributes[1,0]=XClass1NormalizedFrequencies


#iterowanie przez wartości w binach atrybutu o indeksie 0 = XAttribute1
for idxBin, value in enumerate(XAttribute1NormalizedFrequencies):
    if y[idxBin]==0:valuesOfAttributes[0,0,idxBin]
    # print(idxBin, value)


print(valuesOfAttributes[0,0])
print(valuesOfAttributes[1,0])
# print(hellinger_explicit(XClass1NormalizedFrequencies,XClass2NormalizedFrequencies))
