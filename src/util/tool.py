def sort_ts(x_series, y_series):
    n = len(x_series)
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if x_series[j] > x_series[j+1]:
                x_series[j], x_series[j+1] = x_series[j+1], x_series[j]
                y_series[j], y_series[j+1] = y_series[j+1], y_series[j]
    return x_series, y_series
