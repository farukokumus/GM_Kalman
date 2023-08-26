# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#

import numpy as np
from numpy import ndim
from numpy.linalg import inv, cholesky, eig
from scipy import linalg
from scipy.stats import multivariate_normal
import itertools
from enum import Enum

from ime_fgs.utils import row_vec, col_vec


class SigmaPointScheme(Enum):
    SigmaPointsClassic = 1
    SigmaPointsReduced = 2
    GaussHermite = 3


def sigma_points_classic(mean, cov, alpha=0.9):
    # Construct 2n+1 Sigma Points
    # alpha =1 results in classical sigma point transform; everything else results in the scaled unscented transform
    # (the latter is recommended).
    mean = np.squeeze(mean)

    n = mean.size
    k = 3 / alpha / alpha - n

    # Calculate the Sigma points
    sigma_points = np.zeros((n, 2 * n + 1))

    sigma_points[:, 0] = mean
    for i in range(1, n + 1):
        sigma_points[:, i] = mean + alpha * (linalg.sqrtm((n + k) * cov))[:, i - 1]
    for i in range(n + 1, 2 * n + 1):
        sigma_points[:, i] = mean - alpha * (linalg.sqrtm((n + k) * cov))[:, i - n - 1]

    # Calculate the weight of each SP
    weights = np.zeros((2 * n + 1, 1))
    weights[0] = k / (n + k) / alpha / alpha + 1 - 1 / alpha / alpha
    weights[1:] = 1 / (2 * alpha * alpha * (n + k))
    weights = np.squeeze(weights)

    assert (np.isclose(1, sum(weights)))

    return sigma_points, weights


def sigma_points_reduced(mean, cov):
    # Construct n+1 Sigma Points
    n = mean.size

    sign = +1

    u_matrix = sign * np.array([[1 / np.sqrt(2), -1 / np.sqrt(2)]])

    weights = [1 / (n + 1)] * (n + 1)

    for k in range(2, n + 1):
        a = sign / np.sqrt(k * (k + 1))

        lowest_row_part1 = np.ones((1, k)) * a

        right_column_part1 = np.zeros((k - 1, 1))

        lower_right_element = -k * a

        right_column = np.concatenate((right_column_part1, np.atleast_2d(lower_right_element)))

        u_matrix = np.concatenate((u_matrix, lowest_row_part1))

        u_matrix = np.concatenate((u_matrix, right_column), axis=1)

    M = np.ones((1, n + 1)) * mean

    sigma_points = M + linalg.sqrtm((n + 1) * cov) @ u_matrix

    return sigma_points, weights


def quasi_random_riemann_sums(func, mean, cov, amount_of_points):
    try:
        import sobol_seq
    except ImportError:
        raise RuntimeError('Optional package sobol_seq must be installed for this operation.')

    n = mean.size

    def det_jacobi(y):
        z = np.cos(np.pi * y / 2) ** 2
        return ((np.pi / 2) ** n) / np.prod(z)

    def tangens_transform(f, x):
        return f(np.tan(np.pi * x / 2) + mean.flatten()) * det_jacobi(x)

    def e(x):
        return func(x) * multivariate_normal.pdf(x, mean=mean.flatten(), cov=cov)

    def t_e(x):
        return tangens_transform(e, x)

    s_n = amount_of_points
    s = sobol_seq.i4_sobol_generate(n, s_n)
    s = 2 * s - 1
    res_mean = 0
    for i in range(0, s_n):
        res_mean = t_e(s[i]) + res_mean
    res_mean = res_mean * (2 ** n) / s_n

    def v(x):
        return np.outer(func(x) - res_mean, func(x) - res_mean) * multivariate_normal.pdf(x,
                                                                                          mean=mean.flatten(),
                                                                                          cov=cov)

    def t_v(x):
        return tangens_transform(v, x)

    res_cov = 0
    for i in range(0, s_n):
        res_cov = t_v(s[i]) + res_cov
    res_cov = res_cov * (2 ** n) / s_n

    def cr_v(x):
        return np.outer(x - mean, func(x) - res_mean) * multivariate_normal.pdf(x, mean=mean.flatten(), cov=cov)

    def t_cr_v(x):
        return tangens_transform(cr_v, x)

    res_cr_var = 0
    for i in range(0, s_n):
        res_cr_var = t_cr_v(s[i]) + res_cr_var
        res_cr_var = res_cr_var * (2 ** n) / s_n

    return res_mean, res_cov, res_cr_var


def gauss_hermite(mean, cov, degree_of_exactness=10, rotation=False, sparseGrid=False, spectral=False):
    n = mean.size

    # cov = 0.5*(cov + cov.T) + 0.0001*np.eye(np.size(cov, 1))

    degree_arg = round((degree_of_exactness + 1) / 2)

    if sparseGrid is True:  # TODO: Implement Sparse Grid Gauss Hermite
        raise NotImplementedError('Sparse Grid Gauss Hermite not yet implemented')
    else:
        sigma_points_1D, weights_1D = np.polynomial.hermite.hermgauss(degree_arg)

        sigma_points = np.array(list(itertools.product(*(sigma_points_1D,) * n)))

        weights = np.prod(np.array(list(itertools.product(*(weights_1D,) * n))), 1) * (np.pi ** (-0.5 * n))

    if (rotation is True) and (sparseGrid is False):
        rotation_matrix = np.identity(n)
        basis_vectors = np.identity(n)
        reference_basis_vec = basis_vectors[..., 0]
        for i in range(1, n - 1):
            active_basis_vec = basis_vectors[..., i]
            A = np.outer(reference_basis_vec, reference_basis_vec) + np.outer(active_basis_vec, active_basis_vec)
            B = np.outer(reference_basis_vec, active_basis_vec) - np.outer(active_basis_vec, reference_basis_vec)
            rotation_matrix = rotation_matrix @ (np.identity(n) + (np.cos(np.pi / 4) - 1) * A + np.sin(np.pi / 4) * B)
    else:
        rotation_matrix = np.identity(n)

    if spectral is True:
        eigenvalues, eigenvectors = eig(cov)
        transform_matrix = eigenvectors @ (np.identity(n) * np.sqrt(eigenvalues))
        sigma_points = np.dot(rotation_matrix, sigma_points.T).T
        sigma_points = 2.0 ** 0.5 * np.dot(transform_matrix,
                                           sigma_points.T).T + np.array([mean.flatten().T, ] * sigma_points.shape[0])
    else:
        sigma_points = np.dot(rotation_matrix, sigma_points.T).T
        sigma_points = 2.0 ** 0.5 * np.dot(cholesky(cov),
                                           sigma_points.T).T + np.array([mean.flatten().T, ] * sigma_points.shape[0])

    # Sigma points appear to be arranged in a different way than the classic sigma point (unscented transformation)
    sigma_points = sigma_points.T

    return sigma_points, weights


def unscented_transform_gaussian(mean, cov, func, sigma_point_scheme=None, alpha=None, degree_of_exactness=7):
    if sigma_point_scheme is None:
        sigma_point_scheme = SigmaPointScheme.SigmaPointsClassic
    assert isinstance(sigma_point_scheme, SigmaPointScheme)

    if alpha is None:
        alpha = 0.99

    mean = col_vec(mean)
    cov = np.atleast_2d(cov)
    assert np.ndim(cov) == 2
    assert cov.shape[0] == cov.shape[1]

    N = mean.shape[0]

    # Calculate the Sigma Points
    if sigma_point_scheme == SigmaPointScheme.SigmaPointsClassic:
        (sigma_points, weights) = sigma_points_classic(mean, cov, alpha)
    elif sigma_point_scheme == SigmaPointScheme.SigmaPointsReduced:
        (sigma_points, weights) = sigma_points_reduced(mean, cov)
    elif sigma_point_scheme == SigmaPointScheme.GaussHermite:
        (sigma_points, weights) = gauss_hermite(mean, cov, degree_of_exactness=degree_of_exactness)
    else:
        raise NotImplementedError

    # Send the sigma points through func
    if N == 1:
        sigma_points_transformed = row_vec(np.concatenate(
            [col_vec(func(sigma_points[:, point_idx])) for point_idx in range(0, sigma_points.shape[1])], axis=0))
    else:
        sigma_points_transformed = np.concatenate(
            [col_vec(func(sigma_points[:, point_idx])) for point_idx in range(0, sigma_points.shape[1])], axis=1)

    # transform mean
    res_mean = np.zeros_like(sigma_points_transformed[:, [0]], dtype=float)
    for ii in range(0, sigma_points.shape[1]):
        res_mean += weights[ii] * sigma_points_transformed[:, [ii]]

    # transform variance
    res_cov = np.zeros([sigma_points_transformed.shape[0], sigma_points_transformed.shape[0]], dtype=float)
    for ii in range(0, sigma_points.shape[1]):
        res_cov += weights[ii] * col_vec(sigma_points_transformed[:, [ii]] - res_mean) @ \
            row_vec(sigma_points_transformed[:, [ii]] - res_mean)

    # calculate cross-covariance
    cross_cov = np.zeros([sigma_points.shape[0], sigma_points_transformed.shape[0]], dtype=float)

    # Wrong implementation as of 2018/06/25
    # Fixed by Christian Hoffmann
    # np.zeros_like(cov, dtype=float)
    # for ii in range(0, sigma_points.shape[1]):
    #     cross_cov += weights[ii] * col_vec(sigma_points[:, [ii]] - res_mean) @ \
    #                row_vec(sigma_points_transformed[:, [ii]] - res_mean)
    for ii in range(0, sigma_points.shape[1]):
        cross_cov += weights[ii] * col_vec(sigma_points[:, [ii]] - mean) @ \
            row_vec(sigma_points_transformed[:, [ii]] - res_mean)

    # These asserts don't make sense
    # assert ndim(res_cov) == ndim(cross_cov) == ndim(res_mean) == 2
    # assert res_cov.shape == cross_cov.shape == (res_mean.shape[0], res_mean.shape[0])

    return res_mean, res_cov, cross_cov


def backwards_unscented(l_in_msg, r_out_msg, r_in_msg, cr_var):
    '''calculates backwards message of the unscented Node according to the RTS smoother'''

    l_in_msg_w = l_in_msg.convert(type(r_in_msg))
    r_out_msg_w = r_out_msg.convert(type(r_in_msg))

    x_r_w = l_in_msg_w.info
    y_r_w = r_out_msg_w.info
    y_l_w = r_in_msg.info

    x_r_m = l_in_msg_w.weighted_mean
    y_r_m = r_out_msg_w.weighted_mean
    y_l_m = r_in_msg.weighted_mean

    # calculate x_l_w
    # old
    # a = x_r_w*cr_var*y_r_w
    # x_l_w = -a*inv(-(y_r_w+y_r_w*inv(y_l_w)*y_r_w)+(a.transpose())*((l_in_msg.cov).transpose())*a)@(a.transpose())

    # new
    a = x_r_w @ cr_var
    w_y_tilde = y_r_w @ inv(y_r_w + y_l_w) @ y_l_w
    x_l_w = -a @ inv(inv(-w_y_tilde) + cr_var.transpose() @ x_r_w @ cr_var) @ (a.transpose())

    # calculate x_l_m
    # old
    # x_l_m = (x_r_w+x_l_w)@cr_var@(y_r_w@inv(y_r_w+y_l_w)@(y_r_m+y_l_m)-y_r_m)+x_l_w@l_in_msg.mean

    # new
    x_l_m = x_l_w @ l_in_msg.mean - (x_r_w + x_l_w) @ cr_var @ (y_l_w @ inv(y_r_w + y_l_w) @ (y_r_m + y_l_m) - y_l_m)

    weighted_mean = x_l_m
    info = x_l_w

    return weighted_mean, info
