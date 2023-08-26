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
from ime_fgs.divergence_measures import KL_projection_gauss

mu1 = np.array([2])
cov1 = np.array([5])

mu2 = np.array([-6])
cov2 = np.array([2])

dist1 = spst.multivariate_normal(mean=mu1, cov=cov1, allow_singular=False)
dist2 = spst.multivariate_normal(mean=mu2, cov=cov2, allow_singular=False)

N = 100000
x1 = np.linspace(-100, 100, N)
dx = np.tile(np.mean(np.ediff1d(x1)), N)

pos = x1
dist1_pdf = np.asarray(dist1.pdf(pos))
dist2_pdf = np.asarray(dist2.pdf(pos))

p_pdf = (0.5 * (dist1_pdf + dist2_pdf))
# p_pdf = dist1_pdf

fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.plot(x1, p_pdf)

q = KL_projection_gauss(p_pdf, x1)

ax.plot(x1, q.pdf(x1))
plt.show()

# spst.describe(dist1_pdf)
