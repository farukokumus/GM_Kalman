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

x = np.linspace(-10, 10, 1000000)
pos = x

# if the mean of the Gaussian to be truncated is too far off, numerical issues will occur due to the normalization
mu1 = np.array([4])
cov1 = np.array([3])
mu2 = np.array([-1])
cov2 = np.array([4])

covmarg = 1 / (1 / cov1 + 1 / cov2)
mumarg = covmarg * (mu1 / cov1 + mu2 / cov2)

dist1 = spst.multivariate_normal(mean=mu1, cov=cov1, allow_singular=False)
dist2 = spst.multivariate_normal(mean=mu2, cov=cov2, allow_singular=False)
distmarg = spst.multivariate_normal(mean=mumarg, cov=covmarg, allow_singular=False)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
# plt.show()

# Define hyperplane
c = np.array([1])
upper_bound = np.array([1])
lower_bound = np.array([-1])

# trunc1 = ( 1 + np.sign(x - upper_bound))
# trunc2 = (-1 + np.sign(x - lower_bound))
# trunc1 = ( 1 + np.tanh(1000*(x - upper_bound)))
# trunc2 = (-1 + np.tanh(1000*(x - lower_bound)))
trunc1 = ((x - upper_bound) / np.sqrt((x - upper_bound)**2 + 1e-8))
trunc2 = ((x - lower_bound) / np.sqrt((x - lower_bound)**2 + 1e-8))
trunc = kronecker_delta(trunc1) * kronecker_delta(trunc2)
trunc = kronecker_delta(trunc1 + trunc2)
truncapprox = 1 / 4 * (1 - trunc1) * (trunc2 + 1)
ax1.plot(x, truncapprox)


ax1.plot(x, np.log(truncapprox))
ax1.plot(x, np.log(dist1.pdf(x)))
print(np.trapz(np.log(truncapprox), x))
plt.show()
