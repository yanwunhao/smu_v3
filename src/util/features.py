import numpy as np
import networkx as nx


def extract_backswing(curve, threshold):
    reversed_curve = np.array(curve, dtype=float).tolist()
    reversed_curve.reverse()
    count = 0
    for element in reversed_curve:
        if element < threshold:
            count = count + 1
            continue
        else:
            return round(count / len(curve), 3)
    return -1


def extract_maxtime(curve):
    max_value = np.max(curve)
    return max_value

def extract_tt(curve, threshold):
    count = 0
    for element in curve:
        if element < threshold:
            count = count + 1
        else:
            pass
    return round(count/len(curve), 3)


def extract_trt(curve, threshold):
    curve = np.array(curve, dtype=float)
    # convert raw curve to curve of significant figure 3
    curve_sf3 = []
    for element in curve:
        element_sf3 = round(element, 3)
        curve_sf3.append(element_sf3)

    threshold = threshold + 0.0001

    # count TRT
    count = 0
    for i in range(len(curve) - 1):
        if (curve_sf3[i] - threshold) * (curve_sf3[i + 1] - threshold) < 0.:
            count = count + 1
        else:
            continue

    return count


def extract_average_degree(natural_graph, horizontal_graph):
    return (len(natural_graph) * 2 - len(horizontal_graph)) / 20.


def extract_average_path_length(natural_graph):
    G = nx.Graph(natural_graph)
    average_path_length = nx.average_shortest_path_length(G)
    return average_path_length


def extract_transitivity(natural_graph):
    G = nx.Graph(natural_graph)
    transitivity = nx.transitivity(G)
    return transitivity

