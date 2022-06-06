import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.stats import ttest_rel
from tabulate import tabulate

# score functions, datasety, metryki, foldy
scores = np.load("results.npy")
print(scores.shape)
# print(scores[0,0,3])
# print(scores[0,0,4])
# print(scores[0,0,5])

mean_scores = np.mean(scores, axis=3)
print(mean_scores.shape)
# mean_scores = np.mean(mean_scores, axis=1)
# print(mean_scores.shape)
# print("chi2",mean_scores[0,:])
# print("anova",mean_scores[1,:])
# print("hellinger",mean_scores[2,:])
# print("próba",mean_scores[2,4])



score_functions= {
    "chi2",
    "f_classif",
    "hellinger"
}

# datasets = ['cleveland_0_vs_4', 'segment0', 'winequality_red_3_vs_5', 'winequality_red_4', 'winequality_red_8_vs_6',
#             'winequality_red_8_vs_6_7', 'winequality_white_3_9_vs_5', 'winequality_white_3_vs_7', 'winequality_white_9_vs_4','page_blocks_1_3_vs_4']


# metrics = {
#     "accuracy":accuracy_score,
#     "recall": recall_score,
#     'precision': precision_score,
#     'specificity': specificity_score,
#     'f1': f1_score,
#     'g-mean': geometric_mean_score,
#     'bac': balanced_accuracy_score,
# }

alfa = .05
t_statistic_f1 = np.zeros((len(score_functions), len(score_functions)))
p_value_f1 = np.zeros((len(score_functions), len(score_functions)))

for i in range(len(score_functions)):
    for j in range(len(score_functions)):
        # Tu trzeba robić, mean_scores[i,0-9(datasety),4-6(metryki)]
        # 4: f1-score, 5: g-mean, 6: balanced accuracy
        t_statistic_f1[i, j], p_value_f1[i, j] = ttest_rel(mean_scores[i,0,4], mean_scores[j,0,4])
# print("t-statistic:\n", t_statistic_f1, "\n\np-value:\n", p_value_f1)


print("\nG-mean:\n")
headers = ["chi2", "ANOVA", "Hellinger"]
names_column = np.array([["chi2"], ["ANOVA"], ["Hellinger"]])
t_statistic_table = np.concatenate((names_column, t_statistic_f1), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value_f1), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
# print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)


advantage = np.zeros((len(score_functions), len(score_functions)))
advantage[t_statistic_f1 > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
# print("\n\nAdvantage:\n", advantage_table)


significance = np.zeros((len(score_functions), len(score_functions)))
significance[p_value_f1 <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
# print("\n\nStatistical significance (alpha = 0.05):\n", significance_table)


stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)
print("\n\nStatistically significantly better:\n", stat_better_table)