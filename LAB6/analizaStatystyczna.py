import numpy as np
from scipy.stats import ttest_rel
from tabulate import tabulate

scores = np.load("results.npy")
print(scores.shape)
mean_scores = np.mean(scores, axis=2)
print(mean_scores.shape)


datasets = {
    '1:5',
    '1:99',
    '1:9 5%',
    '1:5:5',
    '3 classes balanced'
}

preprocs = {
    'ros',
    'rus',
    'smote' ,
    'none'
}

metrics = {
    "accuracy",
    "recall",
    'precision',
    'specificity',
    'f1',
    'g-mean',
    'bac',
}

alfa = .05
t_statistic_f1 = np.zeros((len(preprocs), len(preprocs)))
p_value_f1 = np.zeros((len(preprocs), len(preprocs)))


for data_id, dataset in enumerate(datasets):
    for metric_id, metric in enumerate(metrics):
        print("\nDATASET: ",dataset,"  METRIC: ",metric)
        for i in range(len(preprocs)):
            for j in range(len(preprocs)):
                t_statistic_f1[i, j], p_value_f1[i, j] = ttest_rel(mean_scores[i,data_id,metric_id], mean_scores[j,data_id,metric_id])
        # print("t-statistic:\n", t_statistic_f1, "\n\np-value:\n", p_value_f1)


        headers = ["ros", "rus", "SMOTE","None"]
        names_column = np.array([["ros"], ["rus"], ["SMOTE"],["None"]])
        t_statistic_table = np.concatenate((names_column, t_statistic_f1), axis=1)
        t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
        p_value_table = np.concatenate((names_column, p_value_f1), axis=1)
        p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
        # print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)


        advantage = np.zeros((len(preprocs), len(preprocs)))
        advantage[t_statistic_f1 > 0] = 1
        advantage_table = tabulate(np.concatenate(
            (names_column, advantage), axis=1), headers)
        # print("\n\nAdvantage:\n", advantage_table)


        significance = np.zeros((len(preprocs), len(preprocs)))
        significance[p_value_f1 <= alfa] = 1
        significance_table = tabulate(np.concatenate(
            (names_column, significance), axis=1), headers)
        # print("\n\nStatistical significance (alpha = 0.05):\n", significance_table)


        stat_better = significance * advantage
        stat_better_table = tabulate(np.concatenate(
            (names_column, stat_better), axis=1), headers)
        print("\n\nStatistically significantly better:\n", stat_better_table)

print(mean_scores[0,:,2])
print(mean_scores[1,:,2])
print(mean_scores[2,:,2])
print(mean_scores[3,:,2])