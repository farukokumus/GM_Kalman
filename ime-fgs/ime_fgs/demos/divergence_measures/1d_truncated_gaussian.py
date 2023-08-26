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
from ime_fgs.divergence_measures import moment_matched_mean_cov_of_doubly_truncated_gaussian, compute_moments

x = np.linspace(-100, 100, 1000000)
pos = x

# if the mean of the Gaussian to be truncated is too far off, numerical issues will occur due to the normalization
mu1 = np.array([10])
cov1 = np.array([8])
mu2 = np.array([-1])
cov2 = np.array([4])
mu1 = np.array([0.0])
cov1 = np.array([0.25])
mu2 = mu1
cov2 = cov1

covmarg = 1 / (1 / cov1 + 1 / cov2)
mumarg = covmarg * (mu1 / cov1 + mu2 / cov2)

dist1 = spst.multivariate_normal(mean=mu1, cov=cov1, allow_singular=False)
dist2 = spst.multivariate_normal(mean=mu2, cov=cov2, allow_singular=False)
distmarg = spst.multivariate_normal(mean=mumarg, cov=covmarg, allow_singular=False)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
# ax1.contour(x, y, dist1.pdf(pos))
ax1.plot(x, dist1.pdf(pos))
ax1.plot(x, dist2.pdf(pos))
ax1.plot(x, distmarg.pdf(pos))
# plt.show()

# Define hyperplane
c = np.array([1])
upper_bound = np.array([0.75])
lower_bound = np.array([-0.75])

moment_matched_mean, moment_matched_cov = moment_matched_mean_cov_of_doubly_truncated_gaussian(
    mu1, cov1, c, upper_bound, lower_bound)
dist4 = spst.multivariate_normal(mean=moment_matched_mean, cov=moment_matched_cov, allow_singular=False)
print(moment_matched_mean)
print(moment_matched_cov**0.5)


moment_matched_mean, moment_matched_cov = moment_matched_mean_cov_of_doubly_truncated_gaussian(
    mu2, cov2, c, upper_bound, lower_bound)
dist5 = spst.multivariate_normal(mean=moment_matched_mean, cov=moment_matched_cov, allow_singular=False)

moment_matched_mean, moment_matched_cov = moment_matched_mean_cov_of_doubly_truncated_gaussian(
    mumarg, covmarg, c, upper_bound, lower_bound)
dist6 = spst.multivariate_normal(mean=moment_matched_mean, cov=moment_matched_cov, allow_singular=False)

ax1.plot(x, 1 / np.trapz(dist4.pdf(pos) * dist2.pdf(pos), pos) * dist4.pdf(pos)
         * dist2.pdf(pos), 'g-')  # Moment matched dist1 times dist2 to form marginal
ax1.plot(x, dist6.pdf(pos), 'k-')  # Moment matched marginal


# ax1.plot(x, dist5.pdf(pos)*dist1.pdf(pos))

trunc1 = (1 + np.sign(x - upper_bound))
trunc2 = (-1 + np.sign(x - lower_bound))
# trunc1 = ( 1 + np.tanh(1000*(x - upper_bound)))
# trunc2 = (-1 + np.tanh(1000*(x - lower_bound)))
# trunc1 = ( 1 + (x - a)/np.sqrt((x - upper_bound)**2 + 1e-3))
# trunc2 = (-1 + (x - b)/np.sqrt((x - lower_bound)**2 + 1e-3))
trunc = kronecker_delta(trunc1) * kronecker_delta(trunc2)
trunc = kronecker_delta(trunc1 + trunc2)
ax1.plot(x, trunc)
ax1.plot(x, trunc * dist1.pdf(pos))
plt.show()

# Real moments of truncated marginal: dist1 with dist2
marg_mean_cov = compute_moments(1 / np.trapz(trunc * dist1.pdf(pos) * dist2.pdf(pos), pos)
                                * trunc * dist1.pdf(pos) * dist2.pdf(pos), pos)
# print(marg_mean_cov[1])
# print(marg_mean_cov[2])

# Moments of truncated dist1
marg_mean_cov = compute_moments(1 / np.trapz(trunc * dist1.pdf(pos), pos) * trunc * dist1.pdf(pos), pos)
# print(marg_mean_cov[1])
# print(marg_mean_cov[2])
dist7 = spst.multivariate_normal(mean=marg_mean_cov[1], cov=marg_mean_cov[2], allow_singular=False)

# Moments of moment matched truncated dist1 marginalized with dist 2
marg_mean_cov = compute_moments(1 / np.trapz(dist7.pdf(pos) * dist2.pdf(pos), pos)
                                * dist7.pdf(pos) * dist2.pdf(pos), pos)
# print(marg_mean_cov[1])
# print(marg_mean_cov[2])
# Shows that there is a different in projecting the marginal or projecting factors of the marginal first

# # Variable shift approach
# v = x + gain*(trunc1 + trunc2)
#
# fig1 = plt.figure()
# ax = fig1.add_subplot(111)
# # ax.plot(x,  v)
# ax.plot(x,  dist1.pdf(x))
# ax.plot(x,  dist1.pdf(v))
# plt.show()
