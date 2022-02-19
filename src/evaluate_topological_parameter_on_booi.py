import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
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

booi_value = np.array(medical_parameter[2], dtype=float)
pvr_list = np.array(medical_parameter[-3], dtype=float)

label = []
for sample_booi in booi_value:
    if sample_booi < 40:
        label.append(0)
    elif sample_booi > 40:
        label.append(1)
    else:
        print("error")

statistical_parameter = np.array(statistical_parameter).T

feature_matrix = [statistical_parameter[1], statistical_parameter[2], statistical_parameter[3], statistical_parameter[4], pvr_list]

feature_matrix = np.array(feature_matrix, dtype=float)
normalized_feature_matrix = []

for feature_line in feature_matrix:
    max_of_line = np.max(feature_line)
    normalized_feature_matrix.append(feature_line / max_of_line)

normalized_feature_matrix = np.array(normalized_feature_matrix, dtype=float).T

cv = KFold(n_splits=10, shuffle=True, random_state=123)
ax = plot_learning_curve(RandomForestClassifier(n_estimators=30, max_depth=3, min_samples_split=10, random_state=10), "Learning Curve", normalized_feature_matrix, label, ax=None, cv=cv)
plt.show()