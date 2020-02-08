import numpy as np
import matplotlib.pylab as plt
import B_spline_curve


def residue_varance(y1, y2):
    if len(y1) == len(y2):
        y1 = np.array(y1)
        y2 = np.array(y2)
        return np.mean(np.square(y1 - y2))


def two_spline_curve(x, y):
    if len(x) != 3 or len(y) != 3:
        return []
    a0 = (x[0] + x[1]) / 2
    a1 = x[1] - x[0]
    a2 = (x[0] - 2 * x[1] + x[2]) / 2

    b0 = (y[0] + y[1]) / 2
    b1 = y[1] - y[0]
    b2 = (y[0] - 2 * y[1] + y[2]) / 2

    return (a0, a1, a2, b0, b1, b2)


def plot_2curve(args):
    plt_cx = []
    plt_cy = []
    curve_cx = []
    curve_cy = []
    for t in range(0, 10):
        t = float(t) / 10
        x_t = args[0] + args[1] * t + args[2] * t * t
        y_t = args[3] + args[4] * t + args[5] * t * t
        curve_cx.append(x_t)
        curve_cy.append(y_t)
        if int(x_t) - x_t == 0:
            plt_cx.append(int(x_t))
            plt_cy.append(y_t)
    # plt.plot(plt_cx,plt_cy)
    return plt_cx, plt_cy, curve_cx, curve_cy


#compress data
def batch_normalization(Line, ruler):
    result = []
    mean = np.mean(Line)
    var = np.var(Line)
    Line = (Line - mean) / (np.sqrt(var + 1e-8))
    periodic = int(len(Line) / ruler)
    remaining = len(Line) % ruler
    true_index = 0
    while true_index < len(Line):
        if remaining != 0:
            temp = np.mean(Line[true_index:true_index + periodic + 1])
            remaining = remaining - 1
            true_index = true_index + 1
        else:
            temp = np.mean(Line[true_index:true_index + periodic])
        result.append(temp)
        true_index = true_index + periodic
    # return Line
    return np.array(result)


def pca(X, k):  # k is the components you want
    # mean of each feature
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    # normalization
    norm_X = X - mean
    # scatter matrix
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    # Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(reverse=True)
    # select the top k eig_vec
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    # get new data
    data = np.dot(norm_X, np.transpose(feature))
    return data
