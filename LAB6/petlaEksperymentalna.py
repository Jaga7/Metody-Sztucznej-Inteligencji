from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score,f1_score,precision_score
from imblearn.metrics import geometric_mean_score,specificity_score
from MySamplingClassifier import MySamplingClassifier


# X,y=datasets.make_classification(
#     n_samples=500,
#     n_features=4,
#     n_informative=2,
#     weights=[1/6,5/6],
#     flip_y=0,
#     random_state=1024
# )
# print(np.bincount(y))
    
datasets = {
    '1:5': datasets.make_classification(
    n_samples=500,
    n_features=4,
    n_informative=2,
    weights=[1/6,5/6],
    flip_y=0,
    random_state=1024
),
    '1:99': datasets.make_classification(
    n_samples=500,
    n_features=4,
    n_informative=2,
    weights=[0.01,0.99],
    flip_y=0,
    random_state=1024
),
    '1:9 5%': datasets.make_classification(
    n_samples=500,
    n_features=4,
    n_informative=2,
    weights=[0.1,0.9],
    flip_y=0.05,
    random_state=1024
),
    '1:5:5':datasets.make_classification(
    n_samples=500,
    n_clusters_per_class=1,
    n_features=4,
    n_classes=3,
    n_informative=2,
    weights=[1/11,5/11,5/11],
    random_state=1024
),
    '3 classes balanced':datasets.make_classification(
    n_samples=500,
    n_clusters_per_class=1,
    n_features=4,
    n_classes=3,
    n_informative=2,
    random_state=1024
),
}



# Wykorzystując funkcję `make_classification` opracuj pięć problemów o różnych parametrach:

# 1. Dychotomię o proporcji klas `1:5`.
# 2. Dychotomię o proporcji klas `1:99`.
# 3. Dychotomię o proporcji klas `1:9` i szumie etykiet na poziomie `5%`.
# 4. Problem z trzema klasami w proporcjach `1:5:5`.
# 5. Referencyjny zbiór zbalansowany z trzema klasami.

# Dla wszystkich problemów wygeneruj `500` wzorców oraz cztery atrybuty, w tym tylko dwa informatywne.

preprocs = {
    'ros': RandomOverSampler(random_state=1410),
    'rus': RandomUnderSampler(random_state=1410),
    'smote' : SMOTE(random_state=1410),
    'none': None,
}

metrics = {
    "accuracy":accuracy_score,
    "recall": recall_score,
    'precision': precision_score,
    'specificity': specificity_score,
    'f1': f1_score,
    'g-mean': geometric_mean_score,
    'bac': balanced_accuracy_score,
}


n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)

scores = np.zeros((len(preprocs),n_datasets, n_splits * n_repeats, len(metrics)))
for data_id, dataset in enumerate(datasets):
    # print(dataset)
     
    X,y=datasets[dataset]
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for preproc_id, preproc in enumerate(preprocs):
            # print(preproc)
            classifier =MySamplingClassifier(base_estimator=GaussianNB(),base_preprocessing=preprocs[preproc])
            if dataset=="1:99"and preproc=='smote':
                classifier =MySamplingClassifier(base_estimator=GaussianNB(),base_preprocessing=SMOTE(random_state=1410,k_neighbors=3))
            
            # if preprocs[preproc] == None:
            #     X_train, y_train = X[train], y[train]
            #     classifier.fit(X[train], y[train])
            # else:
            #     classifier.fit(X[train], y[train])
            #     # X_train, y_train = GaussianNB().fit(X[train], y[train])

            classifier.fit(X[train], y[train])
            y_pred = classifier.predict(X[test])

            for metric_id, metric in enumerate(metrics):
                # print(metric)
                if (dataset=='1:5:5'or dataset=='3 classes balanced') and (metric=="recall" or metric=="precision" or metric=="specificity" or metric=="f1"):
                    scores[preproc_id, data_id,fold_id, metric_id] = metrics[metric](y[test], y_pred,average='micro')
                    
                else:
                    scores[preproc_id, data_id,fold_id, metric_id] = metrics[metric](y[test], y_pred)

np.save('results', scores)

