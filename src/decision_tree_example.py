import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

import csv

from util.tool import plot_learning_curve

with open('./data/handled_parameter.csv', 'r', encoding='utf-8-sig') as f:
    csv_reader = csv.reader(f)
    rows = [row for row in csv_reader]
    medical_parameter = rows
    f.close()

with open('./data/statistical_feature.csv', 'r', encoding='utf-8-sig') as f:
    csv_reader = csv.reader(f)
    rows = [row for row in csv_reader]
    statistical_parameter = rows
    f.close()

with open('./data/topological_feature.csv', 'r', encoding='utf-8-sig') as f:
    csv_reader = csv.reader(f)
    rows = [row for row in csv_reader]
    topological_parameter = rows
    f.close()

medical_parameter = np.array(medical_parameter).T

bci_value = np.array(medical_parameter[1], dtype=float)
booi_value = np.array(medical_parameter[2], dtype=float)
pvr_list = np.array(medical_parameter[-3], dtype=float)

chosen_samples = []
target = []
for i in range(len(bci_value)):
    if bci_value[i] > 100.:
        chosen_samples.append(i)
        target.append(0)

    else:
        chosen_samples.append(i)
        target.append(1)



statistical_parameter = np.array(statistical_parameter).T
topological_parameter = np.array(topological_parameter).T

feature_matrix = [statistical_parameter[1], statistical_parameter[2], statistical_parameter[3], statistical_parameter[4],
                  topological_parameter[1], topological_parameter[2], topological_parameter[3],
                  pvr_list]

feature_matrix = np.array(feature_matrix, dtype=float)
normalized_feature_matrix = []

for feature_line in feature_matrix:
    max_of_line = np.max(feature_line)
    normalized_feature_matrix.append(feature_line / max_of_line)

normalized_feature_matrix = np.array(normalized_feature_matrix, dtype=float).T

train_samples = []
for index in chosen_samples:
    train_samples.append(normalized_feature_matrix[index])


classifier = tree.DecisionTreeClassifier(max_depth=2, min_samples_split=16)
classifier.fit(train_samples, target)


fn=['delta_q','backswing','maxtime','trt', 'average degree', 'average path length', 'transitivity', 'pvr']
cn=['bci>100', 'bci<100']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(classifier,
               feature_names = fn,
               class_names=cn,
               filled = True);
fig.savefig('dt.png')