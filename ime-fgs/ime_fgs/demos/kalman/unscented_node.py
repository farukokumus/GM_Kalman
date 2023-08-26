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
from ime_fgs.advanced_nodes import UnscentedNode
from ime_fgs.messages import GaussianMeanCovMessage

if __name__ == '__main__':
    def f(x):
        return x * x * x

    prior_in = PriorNode(GaussianMeanCovMessage([[5]], [[100]]))
    prior_out = PriorNode(GaussianMeanCovMessage([[f(0.5)]], [[0.5]]))
    unscented_node = UnscentedNode(f)
    prior_in.port_a.connect(unscented_node.port_a)
    unscented_node.port_b.connect(prior_out.port_a)

    num_iter = 1000
    prior_est = np.zeros((num_iter + 1, 1))
    prior_est[0] = 0

    for ii in range(0, num_iter):
        unscented_node.port_b.update()
        unscented_node.port_a.update()
        prior_est[ii + 1] = unscented_node.port_a.marginal(target_type=GaussianMeanCovMessage).mean

    plt.figure()
    plt.plot(prior_est - 0.5, label='estimation error')
    plt.title('Input estimation error')
    plt.xlabel('Iteration')
    plt.ylabel('Estimation Error')
    plt.grid()
    plt.show()
