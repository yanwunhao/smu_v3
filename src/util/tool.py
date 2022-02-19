def sort_ts(x_series, y_series):
    n = len(x_series)
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if x_series[j] > x_series[j+1]:
                x_series[j], x_series[j+1] = x_series[j+1], x_series[j]
                y_series[j], y_series[j+1] = y_series[j+1], y_series[j]
    return x_series, y_series


def plot_learning_curve(estimator, title, x, y, ax=None, ylim=None, cv=None, n_jobs=None):
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np

    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, shuffle=True, cv=cv, n_jobs=n_jobs)

    if ax is None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Score")
    ax.grid()
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training Score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="b", label="Test Score")
    ax.legend(loc="best")
    return ax