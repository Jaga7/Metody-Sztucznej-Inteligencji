import numpy as np
import matplotlib.pyplot as plt
from math import pi

scores = np.load("results.npy")
print(scores.shape)
scores = np.mean(scores, axis=3)
print(scores.shape)
mean_scores = np.mean(scores, axis=1).T
print(mean_scores.shape)

metrics = [
    "accuracy",
    "recall" ,
    'precision' ,
    'specificity',
    'f1',
    'g-mean',
    'bac'
]

score_functions= [
    "chi2",
    "f_classif",
    "hellinger"
]

N = mean_scores.shape[0]

# kat dla kazdej z osi
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# spider plot
ax = plt.subplot(111, polar=True)

# pierwsza os na gorze
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# po jednej osi na metryke
plt.xticks(angles[:-1], metrics)

# os y
ax.set_rlabel_position(0)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
color="grey", size=7)
plt.ylim(0,1)


# Dodajemy wlasciwe ploty dla kazdej z metod
for score_function_id, score_function in enumerate(score_functions):
    print("score_function_id",score_function_id,"score_function",score_function)
    values=mean_scores[:, score_function_id].tolist()
    values += values[:1]
    print(values)
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=score_function)

# Dodajemy legende
plt.legend(bbox_to_anchor=(1.15, -0.05), ncol=5)
# Zapisujemy wykres
plt.savefig("radar", dpi=200)