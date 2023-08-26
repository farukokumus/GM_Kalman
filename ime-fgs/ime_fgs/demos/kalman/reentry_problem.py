# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
# Vehicle reentry example from
# S. J. Julier and J. K. Uhlmann, "Unscented filtering and nonlinear estimation,"
# in Proceedings of the IEEE, vol. 92, no. 3, pp. 401-422, Mar 2004.
#
# S. J. Julier and J. K. Uhlmann, "Corrections to “Unscented Filtering and Nonlinear Estimation”,"
# in Proceedings of the IEEE, vol. 92, no. 12, pp. 1958-1958, Dec. 2004.
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np


def reentry_state_transform(y, t, v=None):
    """
    Implementation of ODE of vehicle in reentry problem
    :param y: state, y[0],y[1] = position of the body
                     y[3], y[4] = velocity
                     y[5] = aerodynamic properties
    :param t: time (unused)
    :param v: v(k) = process noise vector (variance of y[0], y[1], y[2])
    :return: dy/dt
    """
    if v is None:
        v = [0, 0, 0]

    # typical environmental and vehicle characteristics
    Gm0 = 3.9860e5
    b0 = -0.59783
    H0 = 13.406
    R0 = 6374

    # V(k) = speed
    V = np.sqrt(y[2] ** 2 + y[3] ** 2)
    b = b0 * np.exp(y[4])
    # R(k) = distance from the center of the earth
    R = np.sqrt(y[0] * y[0] + y[1] * y[1])

    # D(k) and G(k): force Terms
    D = b * np.exp((R0 - R) / H0) * V
    G = -Gm0 / R**3

    dy1dt = y[2]
    dy2dt = y[3]
    dy3dt = D * y[2] + G * y[0] + np.random.normal(0, v[0])
    dy4dt = D * y[3] + G * y[1] + np.random.normal(0, v[1])
    dy5dt = np.random.normal(0, v[2])

    dydt = [dy1dt, dy2dt, dy3dt, dy4dt, dy5dt]
    return dydt


def radar_measurement(y, position=(6375, 0)):
    """
    Radar measurement of the vehicle (range and bearing)
    :param y: vehicle state
    :param position: position of the radare station
    :return: tuple of range and bearing
    """
    xr = position[0]
    yr = position[1]

    # range
    range = np.sqrt((y[0] - xr) ** 2 + (y[1] - yr) ** 2)
    # bearing
    bearing = np.arctan((y[1] - yr) / (y[0] - xr))
    radar = (range, bearing)
    return radar


def solve_reentry_problem(y0, t, process_noise_cov_vec=None, measure_noise_cov=None):
    """
    Simulate the reentry problem given a start state y and a time vector t. Calculate states and measurement with
    applied gaussian noise
    :param y0: start state
    :param t: time vector t
    :param process_noise_cov_vec: vector of the variance for the first three states
    :param measure_noise_cov: covariance matrix of the measurement noise
    :return: tuple of measurements and states
    """
    if process_noise_cov_vec is None:
        process_noise_cov_vec = [0, 0, 0]
    if measure_noise_cov is None:
        measure_noise_cov = [[0, 0], [0, 0]]
    # solve ode
    states = odeint(reentry_state_transform, y0, t, args=(process_noise_cov_vec,))
    # calculate measurement from radar station
    measurement = np.array([radar_measurement(r) for r in states])
    # add measurement noise
    measurement += np.random.multivariate_normal([0, 0], measure_noise_cov, np.size(measurement, 0))

    return measurement, states


if __name__ == '__main__':
    # Solve the system
    y0 = [6500.4, 349.14, -1.8093, -6.7967, 0.6932]
    t = np.linspace(0, 200, 100)
    t0 = 0

    _, y = solve_reentry_problem(y0, t)

    # plot position
    plt.figure(1)
    plt.plot(y[:, 0], y[:, 1], 'b.-')
    plt.grid(True)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.ylim((-200, 500))
    plt.xlim((6350, 6510))

    # plot velocity
    plt.figure(2)
    plt.subplot(211)
    plt.plot(t, y[:, 2], 'r')
    plt.grid(True)
    plt.xlabel('t')
    plt.ylabel('x3')
    plt.subplot(212)
    plt.plot(t, y[:, 3], 'g')
    plt.grid(True)
    plt.xlabel('t')
    plt.ylabel('x4')

    plt.show()
