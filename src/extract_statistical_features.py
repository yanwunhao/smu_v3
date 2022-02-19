import numpy as np

import csv

from util.features import extract_backswing, extract_trt, extract_tt

with open('./data/handled_curve.csv', 'r', encoding='utf-8-sig') as f:
    csv_reader = csv.reader(f)
    rows = [row for row in csv_reader]
    curves = rows
    f.close()

# split sample number and curve data
curves = np.array(curves)
curves = curves.T
sample_number = curves[0]
curves = curves[1:].T

# change datatype
curves = np.array(curves, dtype=float)

normalized_curves = []

# fix the start and the end of each curve, then normalize it
for curve in curves:
    curve[0] = 0
    curve[-1] = 0
    max_value = np.max(curve)
    normalized_curve = curve / max_value
    normalized_curves.append(normalized_curve)

statistical_features_list = []

# extract features from normalized curves
for data in normalized_curves:
    AVG = round(np.average(data), 3)

    # extract delta_q by calculate delta_q=MAX-AVG
    delta_q = round(1.0-AVG, 3)

    # extract backswing and TRT
    backswing = extract_backswing(data, AVG)
    trt = extract_trt(data, AVG)
    tt = extract_tt(data, AVG)

    feature_vector = [delta_q, backswing, tt, trt]
    statistical_features_list.append(feature_vector)

# merge features and sample_num into one matrix
output_matrix = [sample_number]
statistical_features_list = np.array(statistical_features_list, dtype=float).T
for line in statistical_features_list:
    output_matrix.append(line)

output_matrix = np.array(output_matrix).T

with open('./data/statistical_feature.csv', 'w') as f:
    csv_writer = csv.writer(f)
    for line in output_matrix:
        csv_writer.writerow(line)
    f.close()
