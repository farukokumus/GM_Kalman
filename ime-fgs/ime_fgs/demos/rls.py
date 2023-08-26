# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
from numpy import linspace, sin, squeeze, asarray, atleast_2d
from numpy.random import normal, seed

import matplotlib.pyplot as plt

from ime_fgs.advanced_nodes import ForgettingFactorNode
from ime_fgs.basic_nodes import PriorNode, AdditionNode
from ime_fgs.compound_nodes import CompoundEqualityMatrixNode
from ime_fgs.messages import GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage

if __name__ == '__main__':
    # Simple recursive least squares (RLS) example

    num_meas = 50

    t = linspace(1, 50, num_meas)
    input_a = sin(t)
    input_b = ((t - 15) / 5)**2

    k1 = 3
    k2 = 2

    seed(1)
    meas_noise_cov = [[5]]
    y_true = k1 * input_a + k2 * input_b
    y_meas = y_true + normal(0, meas_noise_cov, num_meas)

    initial_guess = GaussianMeanCovMessage([[0], [0]], [[5, 0], [0, 5]])

    # Forward RLS
    slices = [PriorNode(initial_guess)]

    for kk in range(0, num_meas):
        new_node = CompoundEqualityMatrixNode([input_a[kk], input_b[kk]])
        if len(slices) == 1:
            slices[0].port_a.connect(new_node.port_a)
        else:
            slices[-1].port_c.connect(new_node.port_a)
        new_node.port_b.connect(PriorNode(GaussianMeanCovMessage(atleast_2d(y_meas[kk]), meas_noise_cov)).port_a)
        new_node.port_c.update()
        slices.append(new_node)

    final_estimate = slices[-1].port_c.out_msg.mean
    y_est = final_estimate[0] * input_a + final_estimate[1] * input_b

    print('True constants: ' + str([k1, k2]))
    print('Estimated constants: ' + str(final_estimate))

    plt.figure()
    plt.plot(y_meas, 'k+', label='noisy measurements')
    plt.plot(y_est, 'b-', label='a posteriori estimate')
    plt.plot(y_true, color='g', label='truth value')
    plt.legend()
    plt.title('Estimate vs. iteration step (forward RLS)', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Time [samples]')
    plt.show()

    # Adaptive Forward/Backward RLS with process noise

    num_meas = 500

    t = linspace(1, 50, num_meas)
    input_a = sin(t)
    input_b = ((t - 15) / 5)**2

    # Note time-changing params!
    k1 = linspace(0, 4, num_meas)
    k2 = linspace(3, 0, num_meas)

    seed(1)
    meas_noise_cov = [[5]]
    y_true = k1 * input_a + k2 * input_b
    y_meas = y_true + normal(0, meas_noise_cov, num_meas)

    initial_guess = GaussianMeanCovMessage([[0], [0]], [[5, 0], [0, 5]])
    process_noise_cov = [[1e-3]]
    initial_node = PriorNode(initial_guess)
    eq_mat_nodes = []
    add_nodes = []
    meas_nodes = []

    for kk in range(0, num_meas):
        add_noise_node = AdditionNode()
        if len(eq_mat_nodes) == 0:
            initial_node.port_a.connect(add_noise_node.port_a)
        else:
            eq_mat_nodes[-1].port_c.connect(add_noise_node.port_a)
        add_noise_node.port_b.connect(PriorNode(GaussianMeanCovMessage([[0]], process_noise_cov)).port_a)
        add_nodes.append(add_noise_node)
        new_node = CompoundEqualityMatrixNode([input_a[kk], input_b[kk]])
        add_noise_node.port_c.connect(new_node.port_a)
        add_noise_node.port_c.update()
        meas_node = PriorNode(GaussianMeanCovMessage(atleast_2d(y_meas[kk]), meas_noise_cov))
        new_node.port_b.connect(meas_node.port_a)
        meas_nodes.append(meas_node)
        new_node.port_c.update()
        eq_mat_nodes.append(new_node)

    k_est = [node.port_c.out_msg for node in eq_mat_nodes]
    k1_est = squeeze(asarray([est.mean[0] for est in k_est]))
    k2_est = squeeze(asarray([est.mean[1] for est in k_est]))
    y_est = k1_est * input_a + k2_est * input_b

    plt.figure()
    plt.plot(y_meas, 'k+', label='noisy measurements')
    plt.plot(y_est, 'b-', label='a posteriori estimate')
    plt.plot(y_true, color='g', label='truth value')
    plt.legend()
    plt.title('Measurement estimate vs. iteration step (adaptive forward only /w noise)', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Measurement')

    plt.figure()
    plt.plot(k1_est, 'b-', label='A posteriori estimate of k1')
    plt.plot(k2_est, 'k-', label='A posteriori estimate of k2')
    plt.plot(k1, color='g', label='True k1')
    plt.plot(k2, color='r', label='True k2')
    plt.legend()
    plt.title('Parameter estimates vs. iteration step (adaptive forward only /w noise)', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.show()

    final_node = PriorNode(eq_mat_nodes[-1].port_c.out_msg, target_type=GaussianWeightedMeanInfoMessage)
    eq_mat_nodes[-1].port_c.connect(final_node.port_a)

    for kk in range(num_meas - 1, -1, -1):
        meas_nodes[kk].update_prior(meas_nodes[kk].port_a.out_msg, GaussianWeightedMeanInfoMessage)
        eq_mat_nodes[kk].port_a.update(GaussianMeanCovMessage)
        if kk > 0:
            add_nodes[kk].port_a.update(GaussianWeightedMeanInfoMessage)

    k_est = [node.port_a.out_msg for node in eq_mat_nodes]
    k1_est = squeeze(asarray([est.mean[0] for est in k_est]))
    k2_est = squeeze(asarray([est.mean[1] for est in k_est]))
    y_est = k1_est * input_a + k2_est * input_b

    plt.figure()
    plt.plot(y_meas, 'k+', label='noisy measurements')
    plt.plot(y_est, 'b-', label='a posteriori estimate')
    plt.plot(y_true, color='g', label='truth value')
    plt.legend()
    plt.title('Measurement estimate vs. iteration step (adaptive forward/backward /w noise)', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Measurement')

    plt.figure()
    plt.plot(k1_est, 'b-', label='A posteriori estimate of k1')
    plt.plot(k2_est, 'k-', label='A posteriori estimate of k2')
    plt.plot(k1, color='g', label='True k1')
    plt.plot(k2, color='r', label='True k2')
    plt.legend()
    plt.title('Parameter estimates vs. iteration step (adaptive forward/backward /w noise)', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.show()

    # Adaptive Forward/Backward RLS with forgetting factor

    forgetting_factor = 0.95

    initial_node = PriorNode(initial_guess)
    eq_mat_nodes = []
    meas_nodes = []
    forgetting_nodes = []

    for kk in range(0, num_meas):
        forgetting_node = ForgettingFactorNode(forgetting_factor)
        if len(eq_mat_nodes) == 0:
            initial_node.port_a.connect(forgetting_node.port_a)
        else:
            eq_mat_nodes[-1].port_c.connect(forgetting_node.port_a)
        forgetting_nodes.append(forgetting_node)
        new_node = CompoundEqualityMatrixNode([input_a[kk], input_b[kk]])
        forgetting_node.port_b.connect(new_node.port_a)
        forgetting_node.port_b.update()
        meas_node = PriorNode(GaussianMeanCovMessage(atleast_2d(y_meas[kk]), meas_noise_cov))
        new_node.port_b.connect(meas_node.port_a)
        meas_nodes.append(meas_node)
        new_node.port_c.update()
        eq_mat_nodes.append(new_node)

    k_est = [node.port_c.out_msg for node in eq_mat_nodes]
    k1_est = squeeze(asarray([est.mean[0] for est in k_est]))
    k2_est = squeeze(asarray([est.mean[1] for est in k_est]))
    y_est = k1_est * input_a + k2_est * input_b

    plt.figure()
    plt.plot(y_meas, 'k+', label='noisy measurements')
    plt.plot(y_est, 'b-', label='a posteriori estimate')
    plt.plot(y_true, color='g', label='truth value')
    plt.legend()
    plt.title('Measurement estimate vs. iteration step (adaptive forward only /w forgetting))', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Measurement')

    plt.figure()
    plt.plot(k1_est, 'b-', label='A posteriori estimate of k1')
    plt.plot(k2_est, 'k-', label='A posteriori estimate of k2')
    plt.plot(k1, color='g', label='True k1')
    plt.plot(k2, color='r', label='True k2')
    plt.legend()
    plt.title('Parameter estimates vs. iteration step (adaptive forward only /w forgetting)', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.show()

    final_node = PriorNode(eq_mat_nodes[-1].port_c.out_msg, target_type=GaussianWeightedMeanInfoMessage)
    eq_mat_nodes[-1].port_c.connect(final_node.port_a)

    for kk in range(num_meas - 1, -1, -1):
        meas_nodes[kk].update_prior(meas_nodes[kk].port_a.out_msg, GaussianWeightedMeanInfoMessage)
        eq_mat_nodes[kk].port_a.update(GaussianMeanCovMessage)
        if kk > 0:
            forgetting_nodes[kk].port_a.update(GaussianWeightedMeanInfoMessage)

    k_est = [node.port_a.out_msg for node in eq_mat_nodes]
    k1_est = squeeze(asarray([est.mean[0] for est in k_est]))
    k2_est = squeeze(asarray([est.mean[1] for est in k_est]))
    y_est = k1_est * input_a + k2_est * input_b

    plt.figure()
    plt.plot(y_meas, 'k+', label='noisy measurements')
    plt.plot(y_est, 'b-', label='a posteriori estimate')
    plt.plot(y_true, color='g', label='truth value')
    plt.legend()
    plt.title('Measurement estimate vs. iteration step (adaptive forward/backward /w forgetting)', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Measurement')

    plt.figure()
    plt.plot(k1_est, 'b-', label='A posteriori estimate of k1')
    plt.plot(k2_est, 'k-', label='A posteriori estimate of k2')
    plt.plot(k1, color='g', label='True k1')
    plt.plot(k2, color='r', label='True k2')
    plt.legend()
    plt.title('Parameter estimates vs. iteration step (adaptive forward/backward /w forgetting)', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.show()
