# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
import numpy as np
import scipy.stats as spst
import scipy.special as spsp
import matplotlib.pyplot as plt
from ime_fgs.utils import kronecker_delta, nullspace
from ime_fgs.divergence_measures import moment_matched_mean_cov_of_doubly_truncated_gaussian
from ime_fgs.divergence_measures import KL_projection_gauss

x = np.linspace(-10, 10, 1000)
pos = x

# Define hyperplane
c = np.array([1])
upper_bound = np.array([1])
lower_bound = np.array([-1])

trunc1 = (1 + np.sign(x - upper_bound))
trunc2 = (-1 + np.sign(x - lower_bound))
trunc = kronecker_delta(trunc1 + trunc2)

# if the mean of the Gaussian to be truncated is too far off, numerical issues will occur due to the normalization
mu1 = np.array([2])
cov1 = np.array([3])

# incoming backward messega
mu2 = np.array([4])
cov2 = np.array([5])

dist1 = spst.multivariate_normal(mean=mu1, cov=cov1, allow_singular=False)
dist2 = spst.multivariate_normal(mean=mu2, cov=cov2, allow_singular=False)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

normalize = np.trapz(trunc * dist1.pdf(pos) * dist2.pdf(pos), x)
ax1.plot(x, 1 / normalize * trunc * dist1.pdf(pos) * dist2.pdf(pos), 'b-', label='truncated prior x incoming msg')

# Initialize approximate forward message
moment_matched_mean, moment_matched_cov = moment_matched_mean_cov_of_doubly_truncated_gaussian(
    mu1, cov1, c, upper_bound, lower_bound)
covhat = 1e8  # moment_matched_cov
muhat = 0  # moment_matched_mean

for ii in range(10):

    # Compute marginal
    covmarg = 1 / (1 / covhat + 1 / cov2)
    mumarg = covmarg * (muhat / covhat + mu2 / cov2)

    disthat = spst.multivariate_normal(mean=muhat, cov=covhat, allow_singular=False)
    dist2 = spst.multivariate_normal(mean=mu2, cov=cov2, allow_singular=False)
    distmarg = spst.multivariate_normal(mean=mumarg, cov=covmarg, allow_singular=False)

    # cavity should be equal to dist2!
    covcavity = 1 / (1 / distmarg.cov - 1 / covhat)
    mucavity = covcavity * (distmarg.mean / distmarg.cov - muhat / covhat)
    distcavity = spst.multivariate_normal(mean=mucavity, cov=covcavity, allow_singular=False)

    # First compute marginal w.r.t. untruncated
    covnonapprox = 1 / (1 / covcavity + 1 / cov1)
    munonapprox = covnonapprox * (mucavity / covcavity + mu1 / cov1)

    # Moment matching the marginal
    moment_matched_mean, moment_matched_cov = moment_matched_mean_cov_of_doubly_truncated_gaussian(
        munonapprox, covnonapprox, c, upper_bound, lower_bound)
    distmarg = spst.multivariate_normal(mean=moment_matched_mean, cov=moment_matched_cov, allow_singular=False)

    # Moment matching the true marginal
    p_pdf = trunc * (np.asarray(distcavity.pdf(pos)) * np.asarray(dist1.pdf(pos)))
    normalize = np.trapz(p_pdf, x)
    p_pdf = 1 / normalize * p_pdf
    q = KL_projection_gauss(p_pdf, x)

    ax1.plot(x, q.pdf(pos), 'b-', label='mm(real prior x incoming msg)')
    ax1.plot(x, distmarg.pdf(pos), 'm-', label='mm(real prior) x incoming msg)')

    print()
    # New disthat
    covhat = 1 / (1 / distmarg.cov - 1 / covcavity)
    muhat = covhat * (distmarg.mean / distmarg.cov - mucavity / covcavity)
    # covhat = 1 / (1 / q.cov - 1 / covcavity)
    # muhat = covhat * (q.mean / q.cov - mucavity / covcavity)

# plt.legend( )
plt.show()


# # Variable shift approach
# v = x + gain*(trunc1 + trunc2)
#
# fig1 = plt.figure()
# ax = fig1.add_subplot(111)
# # ax.plot(x,  v)
# ax.plot(x,  dist1.pdf(x))
# ax.plot(x,  dist1.pdf(v))
# plt.show()
