from ime_fgs import messages
import numpy as np
from matplotlib import pyplot as plt
from ime_fgs.messages import GaussianMixtureMeanCovMessage, GaussianMixtureWeightedMeanInfoMessage
from scipy.stats import multivariate_normal

def calc_prob_dens_1d(msg: GaussianMixtureMeanCovMessage, bounds: tuple=(-15, 15)):
    x = np.linspace(*bounds, num=300)
    p = np.zeros(x.shape)
    n = msg.weights.shape[0]
    d = msg.mean.shape[1]

    for k in range(n):
        gaussian = multivariate_normal.pdf(x, msg.mean[k, :, :],msg.cov[k, :, :])
        p[:] += msg.weights[k] * gaussian[0,:]
    return x, p

def calc_prob_dens_2d(msg: GaussianMixtureMeanCovMessage, bounds: tuple=(-15, 15)):

    x, y = np.mgrid[bounds[0]:bounds[1]:.1, bounds[0]:bounds[1]:.1]
    pos = np.dstack((x, y))


    p = np.zeros(pos.shape[:2])

    for k in range(msg.weights.shape[0]):
        print(msg.mean[k, :, :])
        print(msg.cov[k, :, :])
        gaussian = multivariate_normal(msg.mean[k, :, :].reshape(2),msg.cov[k, :, :])
        print(gaussian.pdf(pos).shape)
        p += msg.weights[k] * gaussian.pdf(pos)
    return x, p


wa = [[0.4], [0.6]]
ma = [[[-5],[2]], [[1],[0]]]
va = [[1.2,0],[0,2]], [[1,1],[0,0.2]]

wb = [[0.1], [0.9]]
mb = [[[-2],[1]], [[5],[0]]]
vb = [[0.2,0],[0,3]], [[2,1],[1,1]]

a = GaussianMixtureMeanCovMessage(wa, ma, va)
b = GaussianMixtureMeanCovMessage(wb, mb, vb)
c = a + b

d = a.convert(GaussianMixtureWeightedMeanInfoMessage).combine(b.convert(GaussianMixtureWeightedMeanInfoMessage))
d = d.convert(GaussianMixtureMeanCovMessage)

xa, pa = calc_prob_dens_2d(a)
xb, pb = calc_prob_dens_2d(b)
xc, pc = calc_prob_dens_2d(c)
xd, pd = calc_prob_dens_2d(d)

inta = ((xa[1] - xa[0])**2 * pa).sum()
intb = ((xb[1] - xb[0])**2 * pb).sum()
intc = ((xc[1] - xc[0])**2 * pc).sum()
intd = ((xd[1] - xd[0])**2 * pd).sum()

print(inta, intb, intc,intd)

vmax = np.max([pa.max(),pb.max(),pc.max(), pd.max()])
plt.subplot(2,2,1)
plt.title("A message")
plt.imshow(pa, vmax=vmax)
plt.subplot(2,2,2)
plt.title("B message")
plt.imshow(pb, vmax=vmax)
plt.subplot(2,2,3)
plt.title("A+B message")
plt.imshow(pc, vmax=vmax)
plt.subplot(2,2,4)
plt.title("A=B message")
plt.imshow(pd, vmax=vmax)
plt.show()

plt.subplot(2,2,1)
plt.title("A message")
plt.plot(xa, pa)
plt.ylim(0,0.25)
plt.subplot(2,2,2)
plt.title("B message")
plt.plot(xb, pb)
plt.ylim(0,0.25)
plt.subplot(2,2,3)
plt.title("A+B message")
plt.plot(xc, pc)
plt.ylim(0,0.25)
plt.subplot(2,2,4)
plt.title("A=B message")
plt.plot(xd, pd)
plt.ylim(0,0.25)
plt.show()