import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

target = []
for booi in booi_value:
    if booi < 40:
        target.append(0)
    else:
        target.append(1)

statistical_parameter = np.array(statistical_parameter).T
topological_parameter = np.array(topological_parameter).T

feature_matrix = [statistical_parameter[1], statistical_parameter[2], statistical_parameter[3], statistical_parameter[4],
                  # topological_parameter[1], topological_parameter[2], topological_parameter[3],
                  pvr_list]

feature_matrix = np.array(feature_matrix, dtype=float)
normalized_feature_matrix = []

for feature_line in feature_matrix:
    max_of_line = np.max(feature_line)
    normalized_feature_matrix.append(feature_line / max_of_line)

normalized_feature_matrix = np.array(normalized_feature_matrix, dtype=float).T

classifier = XGBClassifier(base_score=0.6, booster='gblinear', eta=0.3)
# classifier = XGBClassifier(base_score=0.5, booster='dart', eta=0.3)
# classifier = RandomForestClassifier(n_estimators=50, max_depth=3, min_samples_split=10, random_state=10)
# classifier = LogisticRegression(random_state=0)

cv = KFold(n_splits=5, shuffle=True, random_state=100)
ax = plot_learning_curve(classifier, "dart-booi", normalized_feature_matrix, target, ax=None, cv=cv)
plt.show()