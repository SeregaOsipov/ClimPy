import numpy as np
from scipy import stats

__author__ = 'Sergey Osipov <Serega.Osipov@gmail.com>'


def get_cdf(f, radii):
    """
    Compute cdf: integral of f(x)*dx up to x
    :param f:
    :param radii:
    :return:
    """
    cdf = []
    for r in radii:
        ind = radii <= r
        cdf.append(np.trapz(f[ind], radii[ind]))
    return np.array(cdf)


def compute_r_bias_rmse(obs, model, in_log_space=False):
    '''
    in_log_space flag will compute statistics for log(x) instead of x
    '''

    true_values = obs
    predicted_values = model

    if in_log_space:
        true_values = np.log(obs)
        predicted_values = np.log(model)

    n = len(true_values)

    rmse = np.linalg.norm(true_values - predicted_values) / np.sqrt(n)
    bias = np.mean(predicted_values - true_values)
    pearsonr, p_value = stats.pearsonr(predicted_values, true_values)  # scipy

    return pearsonr, bias, rmse