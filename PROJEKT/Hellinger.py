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

# bin_edges_cale chyba będzie trzeba inne dla każdego atrybutu
# Bierzemy bin_edges_cale (tu dla atrybutu o indeksie 0), wykorzystamy przy tworzeniu histogramów dla X+ i X-
Xcale=np.histogram(X[:,0])
histCale, bin_edges_cale=Xcale

# stworzenie 3-wymiarowej macierzy, klasy, atrybuty, wartości w binach
valuesOfAttributes = np.zeros((2, X[0].shape[0], len(bin_edges_cale)-1))


# Służy do zapisania wartości dla różnych klas i zrobienia z nich histogramów
valuesOfAttributesForClass0=[]
valuesOfAttributesForClass1=[]


#iterowanie przez wartości atrybutu o indeksie 0, zapisanie do tablicy i tu powinno się 2 histogramy z nich
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
        # zapisanie prawdopodobieństw występowań (w sensie normalized frequencies) dla rozkładów do macierzy
        valuesOfAttributes[0,0]=XClass0NormalizedFrequencies
        valuesOfAttributes[1,0]=XClass1NormalizedFrequencies




print(valuesOfAttributes[0,0])
print(valuesOfAttributes[1,0])
# print(hellinger_explicit(XClass1NormalizedFrequencies,XClass2NormalizedFrequencies))
