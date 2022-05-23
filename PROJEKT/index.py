import math
import numpy as np
from sklearn import datasets
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone

# n_splits = 5
# n_repeats = 2
# rskf = RepeatedStratifiedKFold(
#     n_splits=n_splits, n_repeats=n_repeats, random_state=42)


clf= GaussianNB()

# datasets = ['australian',]
datasets = ['cleveland_0_vs_4', 'segment0', 'winequality_red_3_vs_5', 'winequality_red_4', 'winequality_red_8_vs_6',
            'winequality_red_8_vs_6_7', 'winequality_white_3_9_vs_5', 'winequality_white_3_vs_7', 'winequality_white_9_vs_4']

n_datasets = len(datasets)

# metrics = {
#     "recall": recall,
#     'precision': precision,
#     'specificity': specificity,
#     'f1': f1_score,
#     'g-mean': geometric_mean_score_1,
#     'bac': balanced_accuracy_score,
# }

winequality_red_4_dataset=np.genfromtxt('winequality_red_4.csv',delimiter=';')
print(winequality_red_4_dataset.shape)
# print(winequality_red_4_dataset[2206,6])
X=winequality_red_4_dataset[:, :-1]
y=winequality_red_4_dataset[:, -1].astype(int)
  
print(X[4])
# segment0 dataset wczytywanie

# segment0_dataset=np.genfromtxt('segment0.csv',delimiter=';')
# print(segment0_dataset.shape)
# # print(segment0_dataset[2206,6])
# X=segment0_dataset[:,0:19]
# y=segment0_dataset.astype(int)[:,19]

# cleveland-0_vs_4 dataset wczytywanie

# cleveland_0_vs_4_dataset=np.genfromtxt('cleveland-0_vs_4.csv',delimiter=';')
# print(cleveland_0_vs_4_dataset.shape)
# X=cleveland_0_vs_4_dataset[:,0:13]
# # print(X2)
# print(X)
# y=cleveland_0_vs_4_dataset.astype(int)[:,13]
# print(y)

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

    # # Służy tylko do 
    # Xcale=np.histogram(X)
    # histCale, bin_edges_cale=Xcale

    # stworzenie 3-wymiarowej macierzy, atrybuty, klasy, wartości w binach
    # valuesOfAttributes = np.zeros((X[0].shape[0],2,  len(bin_edges_cale)-1))
    valuesOfAttributes = np.zeros((X[0].shape[0],2,  10))

    # X[0] daje wartości atrybutów dla próbki o indeksie 0, bierzemy w celu przeiterowania przez ilość atrybutów
    for idxAttribute, uselessValue in enumerate(X[0]):
        # Służy do zapisania wartości dla różnych klas i zrobienia z nich histogramów
        valuesOfAttributesForClass0=[]
        valuesOfAttributesForClass1=[]
        # Bierzemy bin_edges_cale, wykorzystamy przy tworzeniu histogramów dla X+ i X-
        histAtrybutu, bin_edges_atrybutu=np.histogram(X[:,idxAttribute])

        #iterowanie przez wartości atrybutu o indeksie idxAttribute, zapisanie do tablicy i tu powinno się 2 histogramy z nich
        for idxSample, value in enumerate(X[:,idxAttribute]):
            if y[idxSample]==0:valuesOfAttributesForClass0.append(value)
            if y[idxSample]==1:valuesOfAttributesForClass1.append(value)
            if(idxSample==len(X[:,0])-1): 
                XClass0=np.histogram(valuesOfAttributesForClass0,bin_edges_atrybutu)
                XClass1=np.histogram(valuesOfAttributesForClass1,bin_edges_atrybutu)
                XClass0Hist,XClass0BinEdges=XClass0
                XClass1Hist,XClass1BinEdges=XClass1
                XClass0NormalizedFrequencies=XClass0Hist/sum(XClass0Hist)
                XClass1NormalizedFrequencies=XClass1Hist/sum(XClass1Hist)
                # zapisanie prawdopodobieństw występowań (w sensie normalized frequencies) dla rozkładów do macierzy
                valuesOfAttributes[idxAttribute,0]=XClass0NormalizedFrequencies
                valuesOfAttributes[idxAttribute,1]=XClass1NormalizedFrequencies

    return valuesOfAttributes


select=  SelectKBest(k=2, score_func=hellinger)
# select= make_pipeline(MinMaxScaler(), SelectKBest(k=2, score_func=chi2))
select.fit(X_train, y_train)
print("Feature selection: Hellinger distance")

print(select.transform(X_train).shape)
X_test_hellinger = select.transform(X_test)
X_train_hellinger = select.transform(X_train)


clf.fit(X_train_hellinger, y_train)
y_pred = clf.predict(X_test_hellinger)
print(accuracy_score(y_test, y_pred),"\n")