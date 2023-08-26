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
import matplotlib.pyplot as plt
from ime_fgs.divergence_measures import KL_projection_gauss, mode_matching_gauss
from ime_fgs.utils import kronecker_delta, nullspace

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
N = 100000
pos = np.linspace(-10, 10, N)

mu1 = np.array([3])
cov1 = np.array([5])

mu2 = np.array([-6])
cov2 = np.array([2])

# Define hyperplane
c = np.array([1])
upper_bound = np.array([1])
lower_bound = np.array([-1])

trunc1 = (1 + np.sign(pos - upper_bound))
trunc2 = (-1 + np.sign(pos - lower_bound))
trunc = kronecker_delta(trunc1 + trunc2)
ax1.plot(pos, trunc)

dist1 = spst.multivariate_normal(mean=mu1, cov=cov1, allow_singular=False)
dist2 = spst.multivariate_normal(mean=mu2, cov=cov2, allow_singular=False)

dist1_pdf = np.asarray(dist1.pdf(pos))
dist2_pdf = np.asarray(dist2.pdf(pos))

ax1.plot(pos, trunc * dist1.pdf(pos))

q_moment_matching = KL_projection_gauss(trunc * dist1.pdf(pos), pos)
q_mode_matching = mode_matching_gauss(trunc * dist1.pdf(pos), pos)

ax1.plot(pos, q_moment_matching.pdf(pos))
ax1.plot(pos, q_mode_matching.pdf(pos))
plt.show()

# spst.describe(dist1_pdf)
