import numpy as np
from matplotlib import pyplot as plt

def calc_prob_dens(w: np.ndarray, m: np.ndarray, v: np.ndarray, bounds: tuple=(-15, 15)):
    x = np.linspace(*bounds, num=300)
    p = np.zeros(x.shape)
    n = w.shape[0]
    d = m.shape[1]

    for i, x_i in enumerate(x):
        for k in range(n):
            sqrt_k = np.sqrt((2*np.pi)**d * np.linalg.det(v[k, :, :]))
            exp_k = np.exp(-1/2* (x_i - m[k, :]).T * np.linalg.inv(v[k, :, :]) *(x_i - m[k, :]))
            p[i] += w[k] / sqrt_k  * exp_k

    return x, p

def equal_messages(a, b):
    wa, ma, va = a
    wb, mb, vb = b

    na = wa.shape[0]
    da = ma.shape[1]
    nb = wb.shape[0]
    db = ma.shape[1]

    nc = na * nb

    wc = np.kron(wa, wb)

    mc = np.zeros((nc, da))
    vc = np.zeros((nc, da, da))
    for i in range(wa.shape[0]):
        for k in range(wb.shape[0]):
            vc[i * na + k, :, :] = va[i, :, :] + vb[k, :, :]
            mc[i * na + k, :] = np.linalg.inv(vc[i * na + k, :, :]) @ (va[i, :, :] @ ma[i, :] + vb[k, :, :] @ mb[k, :])

    return wc, mc, vc

wa = np.array([0.4, 0.6])
ma = np.array([[-5], [1]])
va = np.array([[[1.2]], [[1]]])

a = (wa, ma, va)

wb = np.array([0.1, 0.9])
mb = np.array([[-2], [5]])
vb = np.array([[[0.2]], [[4]]])

b = (wb, mb, vb)

c = equal_messages(a, b)

xa, pa = calc_prob_dens(*a)
xb, pb = calc_prob_dens(*b)


xc, pc = calc_prob_dens(*c)

inta = ((xa[1] - xa[0]) * pa).sum()
intb = ((xb[1] - xb[0]) * pb).sum()
intc = ((xc[1] - xc[0]) * pc).sum()

print(inta, intb, intc)
plt.subplot(1,3,1)
plt.title("A message")
plt.plot(xa, pa)
plt.subplot(1,3,2)
plt.title("B message")
plt.plot(xb, pb)
plt.subplot(1,3,3)
plt.title("A=B message")
plt.plot(xc, pc)
plt.show()