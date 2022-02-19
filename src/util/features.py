import numpy as np


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
