import numpy as np
from ts2vg import NaturalVG, HorizontalVG

import csv

from util.features import extract_average_degree, extract_average_path_length, extract_transitivity

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

topological_features_list = []

for data in normalized_curves:
    # resample curve to a length of 20
    resampling_x = np.linspace(0, 200, 20)
    resampling_curve = np.interp(resampling_x, range(200), data)

    # make every element of curve become natural multiple of 0.05
    align_curve = []
    for element in resampling_curve:
        element = round(int(element/0.05) * 0.05, 2)
        align_curve.append(element)

    # visibility graph
    natural_g = NaturalVG()
    natural_g.build(align_curve)
    natural_edges = natural_g.edges

    horizontal_g = HorizontalVG()
    horizontal_g.build(align_curve)
    horizontal_edges = horizontal_g.edges

    average_degree = extract_average_degree(natural_edges, horizontal_edges)
    average_path_length = extract_average_path_length(natural_edges)
    transitivity = extract_transitivity(natural_edges)

    feature_vector = [average_degree, average_path_length, transitivity]
    topological_features_list.append(feature_vector)

output_matrix = [sample_number]
topological_features_list = np.array(topological_features_list, dtype=float).T
for line in topological_features_list:
    output_matrix.append(line)

output_matrix = np.array(output_matrix).T

with open('./data/topological_feature.csv', 'w') as f:
    csv_writer = csv.writer(f)
    for line in output_matrix:
        csv_writer.writerow(line)
    f.close()
