import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import ttest_rel
from tabulate import tabulate

# score functions, datasety, metryki, foldy
scores = np.load("results.npy")
print(scores.shape)

mean_scores = np.mean(scores, axis=3)
print(mean_scores.shape)

score_functions= [
    "chi2",
    "f_classif",
    "hellinger"
]

datasets = ['cleveland_0_vs_4', 'segment0', 'winequality_red_3_vs_5', 'winequality_red_4', 'winequality_red_8_vs_6',
            'winequality_red_8_vs_6_7', 'winequality_white_3_9_vs_5', 'winequality_white_3_vs_7', 'winequality_white_9_vs_4','page_blocks_1_3_vs_4']

n_datasets = len(datasets)

metrics = [
    "accuracy",
    "recall" ,
    'precision' ,
    'specificity',
    'f1',
    'g-mean',
    'bac'
]

for data_id, dataset in enumerate(datasets):
    print("Dataset\n")
    for score_function_id, score_function in enumerate(score_functions):
        for metric_id, metric in enumerate(metrics):
            print("\n",datasets[data_id]," ",score_functions[score_function_id]," ",metrics[metric_id],mean_scores[score_function_id,data_id,metric_id])