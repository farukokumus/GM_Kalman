# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
import numpy as np
import matplotlib.pyplot as plt

from ime_fgs.basic_nodes import PriorNode
from ime_fgs.advanced_nodes import UnscentedNode, StatisticalLinearizationNode
from ime_fgs.messages import GaussianMeanCovMessage
from ime_fgs.unscented_utils import SigmaPointScheme

# Insert manual computation
# Rigorously compare with EP (first compute the correct forward message, then marginalize, then project)

if __name__ == '__main__':
    def f(x):
        return x * x * x

    def finv(x):
        return x ** (1. / 3.)

    x0 = 10.0
    xTrue = 2.0

    prior_in = PriorNode(GaussianMeanCovMessage([[x0]], [[1]]))
    prior_out = PriorNode(GaussianMeanCovMessage([[f(xTrue)]], [[0.01]]))
    unscented_node = StatisticalLinearizationNode(f, linearization_about_marginal=True, expectation_propagation=True,
                                                  sigma_point_scheme=SigmaPointScheme.GaussHermite)
    prior_in.port_a.connect(unscented_node.port_a)
    unscented_node.port_b.connect(prior_out.port_a)

    num_iter = 30
    prior_est = np.zeros((num_iter + 1, 1))
    prior_goal = xTrue * np.ones((num_iter + 1, 1))
    result_est = np.zeros((num_iter + 1, 1))
    result_goal = f(xTrue) * np.ones((num_iter + 1, 1))
    prior_est[0] = x0
    result_est[0] = f(x0)

    # Linearization parameters
    M_lin = np.zeros((num_iter + 1, 1))
    n_lin = np.zeros((num_iter + 1, 1))
    V_E = np.zeros((num_iter + 1, 1))

    for ii in range(0, num_iter):
        unscented_node.port_b.update()
        unscented_node.port_a.update()
        prior_est[ii + 1] = unscented_node.port_a.marginal(target_type=GaussianMeanCovMessage).mean
        result_est[ii + 1] = unscented_node.port_b.marginal(target_type=GaussianMeanCovMessage).mean
        M_lin[ii + 1] = unscented_node.M_matrix
        n_lin[ii + 1] = unscented_node.n_offset
        V_E[ii + 1] = unscented_node.cov_err

    plt.figure()
    plt.title('Estimation Error')
    plt.subplot(211)
    plt.plot(prior_est, label='estimation')
    plt.plot(prior_goal, label='goal')
    plt.xlabel('Iteration')
    plt.ylabel('Input Estimation Error')
    plt.grid()
    plt.subplot(212)
    plt.plot(result_est, label='estimation')
    plt.plot(result_goal, label='goal')
    plt.xlabel('Iteration')
    plt.ylabel('Output Estimation Error')
    plt.grid()

    plt.figure()
    plt.title('Linearization Parameters')
    plt.subplot(221)
    plt.plot(M_lin, label='M')
    plt.ylabel('M')
    plt.xlabel('Iteration')
    plt.grid()
    plt.subplot(222)
    plt.plot(n_lin, label='n')
    plt.ylabel('n')
    plt.xlabel('Iteration')
    plt.grid()
    plt.subplot(223)
    plt.plot(V_E, label='V_E')
    plt.ylabel('V_E')
    plt.xlabel('Iteration')
    plt.grid()
    plt.show()
