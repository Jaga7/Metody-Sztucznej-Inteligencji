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

segment0_dataset=np.genfromtxt('segment0.csv',delimiter=';')
print(segment0_dataset.shape)
print(segment0_dataset[2206,6])
X=segment0_dataset[:,0:19]
# print(X2)
# print(X)
# print(segment0_dataset.astype(float)[:,19:20])
y=segment0_dataset.astype(int)[:,19]
# print(y)

for idx, value in enumerate(X):
    for idx2, value2 in enumerate(value):
        if(math.isnan(value2)):
            print(idx)
            print("nanaa") 
            print(idx2)
            print(value2)

# print(y)
# X2 = segment0_dataset[:, :-1]
# y2 = segment0_dataset[:, -1].astype(int)
# print(X2)
# print(y2)


# Feature selection: hellinger

def hellinger(X, y):

    # Bierzemy bin_edges_cale, wykorzystamy przy tworzeniu histogramów dla X+ i X-
    Xcale=np.histogram(X)
    histCale, bin_edges_cale=Xcale

    # stworzenie 3-wymiarowej macierzy, klasy, atrybuty, wartości w binach
    valuesOfAttributes = np.zeros((2, X[0].shape[0], len(bin_edges_cale)-1))
    valuesOfAttributes = give_discretized_distributions(X, y)
    
    array_of_distances=[]

    array_of_squares = []

    # iterowanie przez rozkłady dla atrybutów
    for idxAttribute, distributionsForClasses in enumerate(valuesOfAttributes[:,]):
        array_of_squares = []
        # iterowanie przez biny, indeks 0 tylko po to żeby dostać ilość binów
        for wartośćWBinieDlaKlasy0, wartośćWBinieDlaKlasy1 in zip(distributionsForClasses[0], distributionsForClasses[1]):
            calculation = (math.sqrt(wartośćWBinieDlaKlasy0) - math.sqrt(wartośćWBinieDlaKlasy1)) ** 2
            array_of_squares.append(calculation)
            if(len(array_of_squares)==len(distributionsForClasses[0])):
                sum_of_squares = sum(array_of_squares)
                array_of_distances.append(sum_of_squares / math.sqrt(2))

    # we don't need to return p_values for SelectKBest method, so we return an array of zeros
    WRONGp_values=np.zeros(len(array_of_distances))

    return array_of_distances,WRONGp_values


def give_discretized_distributions(X, y):

    # Bierzemy bin_edges_cale, wykorzystamy przy tworzeniu histogramów dla X+ i X-
    Xcale=np.histogram(X)
    histCale, bin_edges_cale=Xcale

    # stworzenie 3-wymiarowej macierzy, atrybuty, klasy, wartości w binach
    valuesOfAttributes = np.zeros((X[0].shape[0],2,  len(bin_edges_cale)-1))

    # X[0] daje wartości atrybutów dla próbki o indeksie 0, bierzemy w celu przeiterowania przez ilość atrybutów
    for idxAttribute, uselessValue in enumerate(X[0]):
        # Służy do zapisania wartości dla różnych klas i zrobienia z nich histogramów
        valuesOfAttributesForClass0=[]
        valuesOfAttributesForClass1=[]

        #iterowanie przez wartości atrybutu o indeksie idxAttribute, zapisanie do tablicy i tu powinno się 2 histogramy z nich
        for idxSample, value in enumerate(X[:,idxAttribute]):
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
                valuesOfAttributes[idxAttribute,0]=XClass0NormalizedFrequencies
                valuesOfAttributes[idxAttribute,1]=XClass1NormalizedFrequencies

    return valuesOfAttributes

# próba czy działa 
dystanse=hellinger(X,y)
print(dystanse)

# dupa=give_discretized_distributions(X,y)
# print(dupa.shape)
# print(dupa[0,0])
# print(dupa[0,1])
# # iterowanie przez rozkłady dla atrybutów
# for idxAttribute, distributionsForClasses in enumerate(dupa[:,]):
#     # iterowanie przez biny, indeks 0 tylko po to żeby dostać ilość binów
#     for wartośćWBinieDlaKlasy0, wartośćWBinieDlaKlasy1 in zip(distributionsForClasses[0], distributionsForClasses[1]):
#         print(idxAttribute," a wartość w binie dla klasy 0: ", wartośćWBinieDlaKlasy0," a wartość w binie dla klasy 1: ", wartośćWBinieDlaKlasy1)
    


# #iterowanie przez wartości w binach atrybutu o indeksie 0 o klasie 0
# for idxBin, value in enumerate(XAttribute1NormalizedFrequencies):
#     if y[idxBin]==0:valuesOfAttributes[0,0,idxBin]
#     # print(idxBin, value)o


# print(valuesOfAttributes[0,0])
# print(valuesOfAttributes[1,0])

# print("CAŁA MACIERZ 3-WYMIAROWA: ",valuesOfAttributes.shape)

# print(hellinger_explicit(XClass1NormalizedFrequencies,XClass2NormalizedFrequencies))
