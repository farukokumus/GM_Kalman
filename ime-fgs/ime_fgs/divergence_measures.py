# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
import types

import scipy.stats as spst
import scipy.special as spsp
from ime_fgs.utils import nullspace
from numpy import linspace, ndim, trapz, repeat, mean, ediff1d, zeros, ones, exp, sqrt, pi, atleast_2d, shape, diag, \
    concatenate, eye, sign, where
from numpy.linalg import norm, cholesky, inv


def compute_moments(p, x=linspace(-100, 100, 10000), max_order=2):
    """
    Function to numerically compute moments from 0th to specified order of a given probability distribution.

    :param p: Vector containing values of the probability density function p(x).
    :param x: Corresponding vector of associated values of the random variable.
    :param max_order: Maximum order of the moments to return.
    :return: The class with docstring inheritance applied.
    """

    # todo take a look at the R implementation
    assert ndim(p) == 1
    assert ndim(x) == 1
    assert max_order >= 2

    if p.shape[0] == 1:
        p = p.T

    if x.shape[0] == 1:
        x = x.T

    assert p.shape == x.shape

    N = len(x)
    dx = repeat(mean(ediff1d(x)), N)
    # pos    = np.dstack((x1, x2))

    moment = repeat(0.0, max_order + 1)

    moment[0] = trapz(p * dx)
    moment[1] = trapz(p * x * dx)

    for n in range(2, max_order + 1):
        moment[n] = trapz(p * ((x - moment[1]) ** n) * dx)

    return moment


def KL_projection_gauss(p, x=linspace(-100, 100, 10000)):
    """
    Function to numerically project a given probability distribution onto a Gaussian by minimizing the Kullback-Leibler
    divergence.

    :param p: Vector containing values of the probability density function p(x).
    :param x: Corresponding vector of associated values of the random variable.
    :param max_order: Maximum order of the moments to return.
    :return: A Gaussian distribution q with minimal Kullback-Leibler divergence w.r.t. p.
    """

    p_moments = compute_moments(p, x)
    q = spst.multivariate_normal(mean=p_moments[1], cov=p_moments[2], allow_singular=False)

    return q


def mode_matching_gauss(p, x=linspace(-100, 100, 10000)):
    """
    Function to numerically project a given probability distribution onto a Gaussian by minimizing the inverse
    Kullback-Leibler divergence.

    :param p: Vector containing values of the probability density function p(x).
    :param x: Corresponding vector of associated values of the random variable.
    :return: A Gaussian distribution q with minimal Kullback-Leibler divergence w.r.t. p.
    """

    mode_value = max(p)
    mode = where(p == mode_value)

    cov = 9
    # whats the right cov to match the mode

    q = spst.multivariate_normal(mean=x[mode[0][0]], cov=cov, allow_singular=False)

    return q


def mode_matched_mean_cov_of_doubly_truncated_gaussian(mean, cov, hyperplane_normal, hyperplane_upper_bound,
                                                       hyperplane_lower_bound):
    """
    Function to project a given truncated Gaussian distribution onto a Gaussian by matching modes.

    For x as the independent variable, the hyperplane is defined by
        trunc(x) := [[hyperplane_normal.T @ x - hyperplane_lower_bound >= 0]]
        trunc(x) := [[hyperplane_normal.T @ x - hyperplane_upper_bound <= 0]]
    where [[ . ]] is the indicator function.
    The (oneside) truncated Gaussian is therefore given by
        p_T(x) = trunc(x) * p(x)

    The algorithm follows the technical note by Marc Toussaint:
    Marc Toussaint (October 16, 2009): Technical Note:
    Computing moments of a truncated Gaussian for EP in high-dimensions.
    Machine Learning & Robotics group, TU Berlin

    It proceeds by rotating the coordinate basis, such that the constraint is aligned with the
    first axis. It hence first standardizes the Gaussian, then rotates it for alignment with the
    first axis. The truncated moments are then computed based on formulae for the 1D case.
    Finally the truncated mean and covariance are transformed to revert the rotation.

    :param mean: Mean vector of the Gaussian distribution to be truncated.
    :param cov: Covariance matrix of the Gaussian distribution to be truncated.
    :param hyperplane_upper_bound: Offset describing the upper bound of the truncation.
    :param hyperplane_lower_bound: Offset describing the lower bound of the truncation.
    :return: Mean and covariance of a Gaussian distribution q with minimal Kullback-Leibler divergence w.r.t. the
    truncated Gaussian.
    """

    # # Define hyperplane
    # # hyperplane_normal = np.array([1, 0])
    #
    # M = cholesky( atleast_2d( cov + eye( shape(cov)[0] )*1e-12 ) )
    # hyperplane_normal = atleast_2d(hyperplane_normal)
    # mean = atleast_2d(mean)
    #
    # # If the mean of the Gaussian to be truncated is too far off, numerical issues will occur due to the normalization
    # # This is fixed here by adding a small epsilon, which will act as a limit on the "certainty" of clipping.
    # # Consequently the resulting variance of the truncated Gaussian is bounded from below by a small quantity
    # zu_unnorm = (hyperplane_normal.T @ (mean) - hyperplane_upper_bound)
    # # if zu_unnorm >= 1e1:
    # #     zu_unnorm = 1e1
    #
    # zl_unnorm = (hyperplane_normal.T @ (mean) - hyperplane_lower_bound)
    # # if zl_unnorm <= -1e1:
    # #     zl_unnorm = -1e1
    #
    # # Normalized hyperplane offset
    # zl = zl_unnorm / norm( M @ hyperplane_normal )
    # zu = zu_unnorm / norm( M @ hyperplane_normal )
    #
    # # Normalized and rotated hyperplane normal
    # v = (M @ hyperplane_normal) / norm( M @ hyperplane_normal )
    #
    # # Normalization constant
    # n1 = 1.0 / 2.0 * (1 + spsp.erf( zu / sqrt( 2.0 ) ))
    # n2 = 1.0 / 2.0 * (1 + spsp.erf( zl / sqrt( 2.0 ) ))
    #
    # n = n1 - n2
    # if n == 0.0:
    #     # Violations of upper bound
    #     zu_viol = sign( zu_unnorm ) > 0.0
    #     # Violations of lower bound
    #     zl_viol = sign( zl_unnorm ) < 0.0
    #
    #     m = zeros( shape( mean ) )
    #     m[zu_viol] = -zu[zu_viol] # atleast_2d(hyperplane_upper_bound)[zu_viol]
    #     m[zl_viol] = -zl[zl_viol] # atleast_2d(hyperplane_lower_bound)[zu_viol]
    #
    #     l = 1e-12
    # else:
    #
    #     # Moment matched mean of the truncated standardized Gaussian
    #     # (First order moment)
    #     mu = 1.0 / sqrt( 2.0 * pi ) * exp( -0.5 * (zu) ** 2.0 )
    #     ml = 1.0 / sqrt( 2.0 * pi ) * exp( -0.5 * (zl) ** 2.0 )
    #     m = (mu - ml) / n
    #
    #     # Moment matched covariance of the truncated standardized Gaussian
    #     # (Second order moment)
    #     if mu == 0:
    #         zumu = 0.0
    #     else:
    #         zumu = zu * mu
    #
    #     if ml == 0:
    #         zlml = 0.0
    #     else:
    #         zlml = zl * ml
    #
    #     l = 1.0 - (zumu - zlml) / (n) - ((mu - ml) / (n)) ** 2.0
    #
    # # Auxiliary vector with moment matched mean in first entry
    # bdash = zeros( shape( mean ) )
    # bdash[0] = m
    #
    # # Auxiliary matrix with moment matched variance in (1,1) entry
    # Bdash = ones( shape( mean ) )
    # Bdash[0] = l
    # Bdash = diag( Bdash )
    #
    # # Build matrix R rotating e onto v: v = Re
    # # Now solved via svd, can be optimized
    # e = eye( 1, shape( mean )[0] )
    # v_norm = atleast_2d( v ) / norm( v )
    # R = concatenate( (v_norm.T, nullspace( v )), axis=1 )

    # mode_matched_mean = (M.T @ R @ bdash) + mean
    # mode_matched_cov = M.T @ R @ Bdash @ R.T @ M

    # ToDo: Program mode matching such that it makes sense! Or does it?
    #       Apparently reducing the covariance means hard clipping?
    mode_matched_mean = mean
    mode_matched_cov = cov
    if mean > hyperplane_upper_bound:
        mode_matched_mean = hyperplane_upper_bound
        mode_matched_cov = cov * 1e-8
    elif mean < hyperplane_lower_bound:
        mode_matched_mean = hyperplane_lower_bound
        mode_matched_cov = cov * 1e-8

    return mode_matched_mean, mode_matched_cov


def moment_matched_mean_cov_of_doubly_truncated_gaussian(mean, cov, hyperplane_normal, hyperplane_upper_bound,
                                                         hyperplane_lower_bound):
    """
    Function to project a given truncated Gaussian distribution onto a Gaussian by matching moments.

    For x as the independent variable, the hyperplane is defined by
        trunc(x) := [[hyperplane_normal.T @ x - hyperplane_lower_bound >= 0]]
        trunc(x) := [[hyperplane_normal.T @ x - hyperplane_upper_bound <= 0]]
    where [[ . ]] is the indicator function.
    The (oneside) truncated Gaussian is therefore given by
        p_T(x) = trunc(x) * p(x)

    The algorithm follows the technical note by Marc Toussaint:
    Marc Toussaint (October 16, 2009): Technical Note:
    Computing moments of a truncated Gaussian for EP in high-dimensions.
    Machine Learning & Robotics group, TU Berlin

    It proceeds by rotating the coordinate basis, such that the constraint is aligned with the
    first axis. It hence first standardizes the Gaussian, then rotates it for alignment with the
    first axis. The truncated moments are then computed based on formulae for the 1D case.
    Finally the truncated mean and covariance are transformed to revert the rotation.

    :param mean: Mean vector of the Gaussian distribution to be truncated.
    :param cov: Covariance matrix of the Gaussian distribution to be truncated.
    :param hyperplane_upper_bound: Offset describing the upper bound of the truncation.
    :param hyperplane_lower_bound: Offset describing the lower bound of the truncation.
    :return: Mean and covariance of a Gaussian distribution q with minimal Kullback-Leibler divergence w.r.t. the
    truncated Gaussian.
    """

    # Define hyperplane
    # hyperplane_normal = np.array([1, 0])

    M = cholesky(atleast_2d(cov + eye(shape(cov)[0]) * 1e-12))
    hyperplane_normal = atleast_2d(hyperplane_normal)
    mean = atleast_2d(mean)

    # If the mean of the Gaussian to be truncated is too far off, numerical issues will occur due to the normalization
    # This is fixed here by adding a small epsilon, which will act as a limit on the "certainty" of clipping.
    # Consequently the resulting variance of the truncated Gaussian is bounded from below by a small quantity
    zu_unnorm = (hyperplane_normal.T @ (mean) - hyperplane_upper_bound)
    # if zu_unnorm >= 1e1:
    #     zu_unnorm = 1e1

    zl_unnorm = (hyperplane_normal.T @ (mean) - hyperplane_lower_bound)
    # if zl_unnorm <= -1e1:
    #     zl_unnorm = -1e1

    # Normalized hyperplane offset
    zl = zl_unnorm / norm(M @ hyperplane_normal)
    zu = zu_unnorm / norm(M @ hyperplane_normal)

    # Normalized and rotated hyperplane normal
    v = (M @ hyperplane_normal) / norm(M @ hyperplane_normal)

    # Normalization constant
    n1 = 1.0 / 2.0 * (1 + spsp.erf(zu / sqrt(2.0)))
    n2 = 1.0 / 2.0 * (1 + spsp.erf(zl / sqrt(2.0)))

    n = n1 - n2
    if n == 0.0:
        # Violations of upper bound
        zu_viol = sign(zu_unnorm) > 0.0
        # Violations of lower bound
        zl_viol = sign(zl_unnorm) < 0.0

        m = zeros(shape(mean))
        m[zu_viol] = -zu[zu_viol]  # atleast_2d(hyperplane_upper_bound)[zu_viol]
        m[zl_viol] = -zl[zl_viol]  # atleast_2d(hyperplane_lower_bound)[zu_viol]

        l = 1e-8
    else:

        # Moment matched mean of the truncated standardized Gaussian
        # (First order moment)
        mu = 1.0 / sqrt(2.0 * pi) * exp(-0.5 * (zu) ** 2.0)
        ml = 1.0 / sqrt(2.0 * pi) * exp(-0.5 * (zl) ** 2.0)
        m = (mu - ml) / n

        # Moment matched covariance of the truncated standardized Gaussian
        # (Second order moment)
        if mu == 0:
            zumu = 0.0
        else:
            zumu = zu * mu

        if ml == 0:
            zlml = 0.0
        else:
            zlml = zl * ml

        l = 1.0 - (zumu - zlml) / (n) - ((mu - ml) / (n)) ** 2.0

    # Auxiliary vector with moment matched mean in first entry
    bdash = zeros(shape(mean))
    bdash[0] = m

    # Auxiliary matrix with moment matched variance in (1,1) entry
    Bdash = ones(shape(mean))
    Bdash[0] = l
    Bdash = diag(Bdash)

    # Build matrix R rotating e onto v: v = Re
    # Now solved via svd, can be optimized
    e = eye(1, shape(mean)[0])
    v_norm = atleast_2d(v) / norm(v)
    R = concatenate((v_norm.T, nullspace(v)), axis=1)

    moment_matched_mean = (M.T @ R @ bdash) + mean
    moment_matched_cov = M.T @ R @ Bdash @ R.T @ M
    moment_matched_cov = atleast_2d(moment_matched_cov)

    return moment_matched_mean, moment_matched_cov


def moment_matched_weighted_mean_info_of_doubly_truncated_gaussian(weighted_mean, info, hyperplane_normal,
                                                                   hyperplane_upper_bound, hyperplane_lower_bound):
    """
    Function to project a given truncated Gaussian distribution onto a Gaussian by matching moments.

    Refer to function moment_matched_mean_cov_of_doubly_truncated_gaussian for details.
    Right now, this version is simply implemented by first converting mean and covariance into weighted mean and
    information matrix.

    todo: Improve the algorithm to avoid inversion.


    :param weighted_mean: Weighted mean vector of the Gaussian distribution to be truncated.
    :param cov: Covariance matrix of the Gaussian distribution to be truncated.
    :param hyperplane_normal: Normal vector of the hyperplane describing the truncation.
    :param hyperplane_upper_bound: Offset describing the upper bound of the truncation.
    :param hyperplane_lower_bound: Offset describing the lower bound of the truncation.
    :return: Weighted mean and information matrix of a Gaussian distribution q with minimal Kullback-Leibler divergence
    w.r.t. the truncated Gaussian.
    """

    cov = inv(atleast_2d(info))
    mean = cov @ atleast_2d(weighted_mean)

    moment_matched_mean, moment_matched_cov = \
        moment_matched_mean_cov_of_doubly_truncated_gaussian(mean,
                                                             cov,
                                                             hyperplane_normal,
                                                             hyperplane_upper_bound,
                                                             hyperplane_lower_bound)

    moment_matched_info = inv(atleast_2d(moment_matched_cov))
    moment_matched_weighted_mean = moment_matched_info @ moment_matched_mean

    return moment_matched_weighted_mean, moment_matched_info
