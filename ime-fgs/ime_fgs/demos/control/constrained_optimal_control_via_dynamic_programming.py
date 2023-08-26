# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
'''
Institute for Electrical Engineering in Medicine
University of Lübeck

2017/11/06 Christian Hoffmann
'''

import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import linspace, squeeze, concatenate
from numpy.linalg import inv
from numpy.random import normal, seed
from scipy import optimize

'''
Compute the constrained optimal control solution by dynamic programming
1. recursion from future
2. forward solution and constraint check
3. repeat part of backward recursion until constraint check occurred
4. goto 2 until no additional constraint check is positive
'''


if __name__ == '__main__':
    #
    # Example due to Welsh, Bishop (2006): An Introduction to the Kalman Filter
    # http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
    #
    # Corresponds to estimation of a constant given noisy measurements of that constant.
    #
    # Code in parts due to http://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html
    #

    # Define system model
    A = np.atleast_2d(1.0)
    C = np.atleast_2d(1.0)
    B = np.atleast_2d(3.0)
    tn = 1.0
    ta = -0.5
    tb = 0.5

    nx = 1
    nu = 1

    # Number of time instants
    num_meas = 20

    seed(1)
    z_real = 0 * concatenate([linspace(0, 0, 1 * num_meas // 5),
                              linspace(10, 10, 3 * num_meas // 5), linspace(0, 0, 1 * num_meas // 5)])
    z = z_real + 0 * normal(0, 1, num_meas)
    u = linspace(0, 0, num_meas)

    R = np.atleast_2d(1e1)
    Q = np.atleast_2d(1e0)

    # Dynamic programming
    Qk = [np.atleast_2d(Q)] * (num_meas)
    Rk = [np.atleast_2d(R)] * num_meas
    Wx = [np.atleast_2d(0)] * (num_meas + 1)
    Wx[num_meas] = Q
    uk = [np.atleast_2d(0)] * num_meas
    u0 = [np.atleast_2d(0)] * num_meas
    xk = [np.atleast_2d(0)] * (num_meas + 1)
    y_dynprog = [np.atleast_2d(0)] * (num_meas + 1)
    u_dynprog = [np.atleast_2d(0)] * num_meas
    xk[0] = np.atleast_2d(10.)

    def bwdRecursion(startTime, endTime, Wx, Qk, Rk):
        for kk in range(startTime - 1, endTime - 1, -1):
            Wx[kk] = Qk[kk] + A.T * inv(inv(Wx[kk + 1]) + B * inv(Rk[kk]) * B.T) * A
            # Wx[kk] = Qk[kk] + A.T * (Wx[kk + 1] - Wx[kk + 1] * B * inv( Rk[kk] + B.T * Wx[kk + 1] * B ) *
            #                          B.T * Wx[kk + 1]) * A
        return Wx, Qk, Rk

    Wx, Qk, Rk = bwdRecursion(startTime=num_meas, endTime=0, Wx=Wx, Qk=Qk, Rk=Rk)

    kk = 0
    y_dynprog[kk] = xk[kk]
    while kk < num_meas - 1:
        # for kk in range(0, num_meas):
        uk[kk] = - (inv(Rk[kk] + B.T * Wx[kk + 1] * B) * B.T * Wx[kk + 1] * A) * xk[kk]
        xk[kk + 1] = A * xk[kk] + B * (uk[kk] + u0[kk])
        y_dynprog[kk + 1] = xk[kk + 1]
        u_dynprog[kk] = uk[kk] + u0[kk]

        if uk[kk] > tb:
            uk[kk] = np.atleast_2d(0.)
            u0[kk] = tb
            Rk[kk] = np.atleast_2d(1e8)
            Wx, Qk, Rk = bwdRecursion(startTime=kk, endTime=0, Wx=Wx, Qk=Qk, Rk=Rk)
            kk = 0
        elif uk[kk] < ta:
            uk[kk] = np.atleast_2d(0.)
            u0[kk] = ta
            Rk[kk] = np.atleast_2d(1e8)
            Wx, Qk, Rk = bwdRecursion(startTime=kk, endTime=0, Wx=Wx, Qk=Qk, Rk=Rk)
            kk = 0
        else:
            kk = kk + 1

    # Now compare with quadratic programming
    # minimize
    #     J = u'Hu + 2x0'F'u + x0'Gx0

    # subject to:
    #      Au <= b

    calC_lr = B
    calC = B
    calM_lr = A
    calM = A
    x0 = np.atleast_2d(10.)

    for ii in range(num_meas - 1):
        calC_lr = np.concatenate((A @ calC_lr, B), axis=1)
        calC = np.concatenate((calC, np.zeros([nx * (ii + 1), nu])), axis=1)
        calC = np.concatenate((calC, calC_lr), axis=0)
        calM_lr = A @ calM_lr
        calM = np.concatenate((calM, calM_lr), axis=0)

    Qtilde = np.kron(np.eye(num_meas), Q)
    Rtilde = np.kron(np.eye(num_meas), R)

    H = calC.T @ Qtilde @ calC + Rtilde
    G = calM.T @ Qtilde @ calM + Q
    F = calC.T @ Qtilde @ calM
    c = x0.T @ F.T
    c0 = x0.T @ G @ x0

    A_ineq = np.concatenate((np.sign(tb) * np.eye(num_meas), np.sign(ta) * np.eye(num_meas)), axis=0)

    b_ineq = np.concatenate((tb * np.ones([num_meas, 1]), abs(ta) * np.ones([num_meas, 1])), axis=0)

    u0 = 0 * np.random.randn(num_meas)

    def loss(x, sign=1.):
        return sign * (0.5 * x.T @ H @ x + c @ x + c0)

    def jac(x, sign=1.):
        return sign * (x.T @ H + c)

    cons = {'type': 'ineq',
            'fun': lambda x: b_ineq.squeeze() - A_ineq @ x,
            'jac': lambda x: -A_ineq}

    opt = {'disp': True}

    start = time.perf_counter()

    res_cons = optimize.minimize(loss, u0, jac=jac, constraints=cons,
                                 method='SLSQP', options=opt)

    end = time.perf_counter()

    print('Time elapsed to compute optimal control input via quadratic programming: ' + str(end - start))

    res_uncons = optimize.minimize(loss, u0, jac=jac, method='SLSQP',
                                   options=opt)

    print('\nConstrained:')
    print(res_cons)

    print('\nUnconstrained:')
    print(res_uncons)

    u_cons = res_cons['x']
    f = res_cons['fun']

    u_unc = res_uncons['x']
    f_unc = res_uncons['fun']

    # G = ctrl.ss(A, B, C, 0, 1)
    # T = np.linspace(1, num_meas, num_meas)
    # T_cons, y_cons, x_cons = ctrl.forced_response(G, T, u_cons, x0)
    y_cons = np.atleast_2d(x0)
    for ii in range(num_meas):
        y_cons = np.concatenate((y_cons, A * y_cons[ii] + B * u_cons[ii]), axis=0)

    y_unc = np.atleast_2d(x0)
    for ii in range(num_meas):
        y_unc = np.concatenate((y_unc, A * y_unc[ii] + B * u_unc[ii]), axis=0)

    # plotting
    plt.figure()
    plt.plot(z, 'k-', label='reference')
    # plt.plot(squeeze([fwd_estimate.mean for fwd_estimate in lqg.get_state_out_msgs()]),
    #          'b-', label='forward estimate')
    # plt.plot(squeeze([smoothed_estimate.mean for smoothed_estimate in lqg.get_state_in_marginal_msgs()]),
    #          'r-', label='a posteriori estimate')
    plt.plot(squeeze([y_cons]), 'k-', label='quadratic programming solution')
    plt.plot(squeeze([y_unc]), 'b-', label='unconstrained quadratic programming solution')
    plt.plot(squeeze([y_dynprog]), 'g-', label='dynamic programming solution')

    plt.plot(z_real, color='g', label='truth value')
    plt.legend()
    plt.title('Estimate vs. iteration step', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('')

    plt.figure()
    plt.plot(squeeze(u_cons), 'k-', label='quadratic programming solution')
    plt.plot(squeeze(u_unc), 'b-', label='unconstrained quadratic programming solution')
    plt.plot(squeeze(u_dynprog), 'r-', label='dynamic programming solution')
    plt.legend()
    plt.title('Estimate vs. iteration step', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('')

    plt.show()
