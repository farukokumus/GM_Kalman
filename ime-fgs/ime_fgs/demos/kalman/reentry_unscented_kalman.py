# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
# from numpy import linspace, inf, squeeze, log, sin, cos, ones, zeros, identity
import numpy as np
import matplotlib.pyplot as plt
from ime_fgs.demos.kalman.reentry_problem import reentry_state_transform, radar_measurement, solve_reentry_problem

from ime_fgs.base import NodePort, Node
from ime_fgs.basic_nodes import AdditionNode, PriorNode, EqualityNode
from ime_fgs.advanced_nodes import UnscentedNode
from ime_fgs.messages import GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage
from ime_fgs.utils import col_vec


class KalmanSliceDoubleFrequency(Node):
    def __init__(self, A, C, sigma_point_scheme=None, alpha=None, name=None):
        super().__init__(name)
        self.name = name

        # Initialize ports of the macro (slice) node
        self.port_state_in = NodePort(self, self.calc_msg_state_in, name="port_state_in")
        self.port_state_out = NodePort(self, self.calc_msg_state_out, name="port_state_out")
        self.port_meas = NodePort(self, self.calc_msg_meas, name="port_meas")
        self.port_process_noise1 = NodePort(self, self.calc_msg_process_noise, name="port_process_noise1")
        self.port_process_noise2 = NodePort(self, self.calc_msg_process_noise, name="port_process_noise2")

        # Initialize all relevant nodes
        self.A1_node = UnscentedNode(A, sigma_point_scheme=sigma_point_scheme, alpha=alpha)
        self.A2_node = UnscentedNode(A, sigma_point_scheme=sigma_point_scheme, alpha=alpha)
        self.add_process_noise_node1 = AdditionNode()
        self.add_process_noise_node2 = AdditionNode()
        self.process_noise_in_node1 = PriorNode(GaussianMeanCovMessage([[0]], [[np.inf]]))
        self.process_noise_in_node2 = PriorNode(GaussianMeanCovMessage([[0]], [[np.inf]]))

        self.equality_node = EqualityNode(allow_unconnected_ports=True)
        self.C_node = UnscentedNode(C, sigma_point_scheme=sigma_point_scheme, alpha=alpha)  # radar Position
        self.state_in_node = PriorNode(GaussianMeanCovMessage([[0]], [[np.inf]]))  # in MeanCovMessage
        self.state_out_node = PriorNode(GaussianMeanCovMessage([[0]], [[np.inf]]))  # out Message
        self.meas_in_node = PriorNode(GaussianMeanCovMessage([[0]], [[np.inf]]))

        # Connect the nodes
        self.state_in_node.port_a.connect(self.A1_node.port_a)
        self.A1_node.port_b.connect(self.add_process_noise_node1.port_a)
        self.process_noise_in_node1.port_a.connect(self.add_process_noise_node1.port_b)

        self.add_process_noise_node1.port_c.connect(self.A2_node.port_a)
        self.A2_node.port_b.connect(self.add_process_noise_node2.port_a)
        self.add_process_noise_node2.port_c.connect(self.equality_node.ports[0])
        self.process_noise_in_node2.port_a.connect(self.add_process_noise_node2.port_b)

        # Measurement_noise branch
        self.equality_node.ports[1].connect(self.C_node.port_a)
        self.C_node.port_b.connect(self.meas_in_node.port_a)

    def calc_msg_state_out(self):
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.A1_node.port_b.update(GaussianMeanCovMessage)
        self.process_noise_in_node1.update_prior(self.port_process_noise1.in_msg)
        self.add_process_noise_node1.port_c.update(GaussianMeanCovMessage)
        self.A2_node.port_b.update(GaussianMeanCovMessage)
        self.process_noise_in_node2.update_prior(self.port_process_noise2.in_msg)
        self.add_process_noise_node2.port_c.update(GaussianWeightedMeanInfoMessage)

        self.equality_node.ports[1].update(GaussianMeanCovMessage)
        self.equality_node.ports[2].connect(self.state_out_node.port_a)
        self.C_node.port_b.update(GaussianMeanCovMessage)
        self.meas_in_node.update_prior(self.port_meas.in_msg, GaussianMeanCovMessage)
        self.C_node.port_a.update(GaussianWeightedMeanInfoMessage)

        return self.equality_node.ports[2].update(GaussianMeanCovMessage)

# No changes have been made yet
# ------------------------------------------------------------------------------------------------------------------
    def calc_msg_state_in(self):
        # Caution: Untested
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.input_in_node.update_prior(self.port_input.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg, target_type=GaussianWeightedMeanInfoMessage)
        self.meas_in_node.update_prior(self.port_meas.in_msg, target_type=GaussianWeightedMeanInfoMessage)

        self.compound_eq_mat_node.port_a.update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_a.update(GaussianMeanCovMessage)
        self.add_process_noise_node.port_a.update(GaussianMeanCovMessage)

        return self.A_node.port_a.update()

    def calc_msg_meas(self):
        # Caution: Untested
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.meas_in_node.update_prior(self.port_meas.in_msg)
        self.input_in_node.update_prior(self.port_input.in_msg)
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)

        self.A_node.port_b.update()
        self.add_process_noise_node.port_c.update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_c.update(GaussianWeightedMeanInfoMessage)

        return self.compound_eq_mat_node.port_b.update()

    def calc_msg_input(self):
        # Caution: Untested
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        self.meas_in_node.update_prior(self.port_meas.in_msg)
        self.input_in_node.update_prior(self.port_input.in_msg)

        self.A_node.port_b.update()
        self.add_process_noise_node.port_c.update(GaussianMeanCovMessage)
        self.compound_eq_mat_node.port_a.update(GaussianMeanCovMessage)
        self.add_input_node.port_b.update(GaussianMeanCovMessage)

        return self.B_node.port_a.update(GaussianMeanCovMessage)

    def calc_msg_process_noise(self):
        # Caution: Untested
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        self.meas_in_node.update_prior(self.port_meas.in_msg)
        self.input_in_node.update_prior(self.port_input.in_msg)

        self.A_node.port_b.update()

        self.compound_eq_mat_node.port_a.update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_a.update(GaussianMeanCovMessage)

        return self.add_process_noise_node.port_b.update(GaussianMeanCovMessage)
# -----------------------------------------------------------------------------------------------------------

    def get_ports(self):
        return [self.port_state_in, self.port_state_out, self.port_meas, self.port_process_noise1,
                self.port_process_noise2]


class UnscentedKalmanFilter(object):
    def __init__(self, A_fun, C_fun, initial_state_msg, process_noise_cov=None, meas_noise_cov=None,
                 slice_type=KalmanSliceDoubleFrequency, sigma_point_scheme=None, alpha=None):
        """
        Initialize a Kalman filter object given an initial state.

        :param A_fun: System function to be used as the default in all time slices.
        :param C_fun: Output function to be used as the default in all time slices.
        :param initial_state_msg: Message specifying information about the initial system state.
        :param input_noise_cov: Input noise covariance to be used as the default in all time slices. If not provided,
          input noise messages must be provided in each time slice.
        :param process_noise_cov: Process noise covariance to be used as the default in all time slices. If not
          provided, process noise messages must be provided in each time slice.
        :param meas_noise_cov: Measurement noise covariance to be used as the default in all time slices. If not
          provided, measurement noise messages must be provided in each time slice.
        :param slice_type: The Kalman filter time slice model to be used.
        """
        # Store default parameter values
        self.A_fun = A_fun
        self.C = C_fun
        self.slice_type = slice_type
        self.process_noise_cov = process_noise_cov
        self.meas_noise_cov = meas_noise_cov
        self.alpha = alpha
        self.sigma_point_scheme = sigma_point_scheme

        # Initialize factor graph
        self.slices = []
        self.init_state_node = PriorNode(initial_state_msg, name="init_state_node")
        n = len(initial_state_msg.mean)
        final_wm = np.zeros((n, 1))
        final_info = np.zeros((n, n))
        self.final_state_node = PriorNode(GaussianWeightedMeanInfoMessage(final_wm, final_info),
                                          name="final_state_node")

    def add_slice(self, meas_val, A=None, C=None, process_noise_msg=None,
                  meas_noise_msg=None):
        """
        Add a new time slice to the Kalman filter, connect it to the previous slices and update the state estimate.

        :param input_val: Input values in each time slice. Note that input noise is already taken into account in the
          system model. Time-varying input noise covariances can be passed separately.
        :param meas_val: Measurement values in this time slice. Note that measurement noise is already taken into
          account in the system model. Time-varying measurement noise covariances can be passed separately.
        :param A: System matrix for this time slice. If not provided, the default one passed to the Kalman filter
          constructor is used.
        :param B: Input matrix for this time slice. If not provided, the default one passed to the Kalman filter
          constructor is used.
        :param C: Output matrix for this time slice. If not provided, the default one passed to the Kalman filter
          constructor is used.
        :param input_noise_msg: Message specifying knowledge about the input noise signal in this time slice. If not
          provided, the default input noise covariance passed to the Kalman filter constructor is used.
        :param process_noise_msg: Message specifying knowledge about the process noise signal in this time slice. If not
          provided, the default process noise covariance passed to the Kalman filter constructor is used.
        :param meas_noise_msg: Message specifying knowledge about the measurement noise signal in this time slice. If
          not provided, the default measurement noise covariance passed to the Kalman filter constructor is used.
        :return: The updated state estimate obtained by connecting the new factor nodes to the previous ones and running
          new_slice.port_state_out.update().
        """
        if A is None:
            A = self.A_fun
        if C is None:
            C = self.C
        if process_noise_msg is None:
            process_noise_msg = GaussianMeanCovMessage([[0], [0], [0], [0], [0]], self.process_noise_cov)
        if meas_noise_msg is None:
            meas_noise_msg = GaussianMeanCovMessage([[0], [0]], self.meas_noise_cov)

        new_slice = self.slice_type(A, C, sigma_point_scheme=self.sigma_point_scheme, alpha=self.alpha)
        # new_slice.meas_noise_in_node._prior = meas_noise_msg

        # np.identity --> nicht sinnvoll
        new_slice.port_meas.connect(PriorNode(GaussianMeanCovMessage(meas_val, meas_noise_msg.cov)).port_a)
        new_slice.port_process_noise1.connect(PriorNode(process_noise_msg).port_a)
        new_slice.port_process_noise2.connect(PriorNode(process_noise_msg).port_a)

        if len(self.slices) == 0:
            self.init_state_node.port_a.connect(new_slice.port_state_in)
        else:
            self.slices[-1].port_state_out.disconnect()
            self.slices[-1].port_state_out.connect(new_slice.port_state_in)

        # connect the 'terminator' node to this slice since in this case it is the last one
        new_slice.port_state_out.connect(self.final_state_node.port_a)
        self.final_state_node.port_a.update()

        self.slices.append(new_slice)

    def do_forward(self):
        # function we need in main
        for s in self.slices:
            s.port_state_out.update()

    def do_backward(self):
        for s in reversed(self.slices):
            s.port_state_in.update()

    def get_forward_state_msgs(self):
        return [s.port_state_out.out_msg for s in self.slices]

    def get_state_marginals(self):
        return [s.port_state_out.marginal(target_type=GaussianMeanCovMessage) for s in self.slices]


if __name__ == '__main__':
    # make program deterministic
    np.random.seed(0)

    # time step in s
    ts = 0.1

    # process noise vector v
    process_noise_cov_vec = [0, 0, 2.4064e-5, 2.4064e-5, 1e-6]
    # measurement noise vector w
    measurement_noise_cov = np.diag([1, 0.017])

    # Simulate system forward
    t = np.arange(0, 200, ts)
    num_slices = np.size(t) - 1
    y0 = [6500.4, 349.14, -1.8093, -6.7967, 0.6932]

    [z, reference_value] = solve_reentry_problem(y0, t, process_noise_cov_vec[2:5], measurement_noise_cov)

    # Kalman filter
    # discretize non linear state transform function (sampled at twice the frequency)
    def A(x):
        return x + ts / 2 * np.transpose(reentry_state_transform(x, 0))

    # discretize non linear measurement function
    def C(x):
        return radar_measurement(x)

    # discretize process noise (sampled at twice the frequency)
    process_noise_cov_discrete = np.diag(process_noise_cov_vec) * ts / 2

    # initial state
    initial_state_kalman = GaussianMeanCovMessage([[6500.4], [349.14], [-1.8093], [-6.7967], [0]],
                                                  np.diag([10e-6, 10e-6, 10e-6, 10e-6, 1]))

    kf_reentry = UnscentedKalmanFilter(A, C, initial_state_kalman,
                                       process_noise_cov=process_noise_cov_discrete,
                                       meas_noise_cov=measurement_noise_cov,
                                       slice_type=KalmanSliceDoubleFrequency,
                                       sigma_point_scheme=None,
                                       alpha=None)

    for ii in range(num_slices):
        kf_reentry.add_slice(meas_val=col_vec(z[ii]))

    kf_reentry.do_forward()

    # Plots results

    # Create vectors with Covariances
    x5_cov = np.squeeze([estimate.cov[4][4] for estimate in kf_reentry.get_forward_state_msgs()])
    x3_cov = np.squeeze([estimate.cov[2][2] for estimate in kf_reentry.get_forward_state_msgs()])
    x1_cov = np.squeeze([estimate.cov[0][0] for estimate in kf_reentry.get_forward_state_msgs()])

    x5_mean2error = np.zeros(num_slices)
    x3_mean2error = np.zeros(num_slices)
    x1_mean2error = np.zeros(num_slices)
    x1 = np.zeros(num_slices)
    x2 = np.zeros(num_slices)
    x1_real = np.zeros(num_slices)
    x2_real = np.zeros(num_slices)
    # Create vectors with Mean-squared-error
    for i in range(num_slices):
        x1[i] = kf_reentry.get_forward_state_msgs()[i].mean[0]
        x2[i] = kf_reentry.get_forward_state_msgs()[i].mean[1]
        x1_real[i] = reference_value[i + 1][0]
        x2_real[i] = reference_value[i + 1][1]
        x5_mean2error[i] = (kf_reentry.get_forward_state_msgs()[i].mean[4] - reference_value[i + 1][4]) ** 2
        x3_mean2error[i] = (kf_reentry.get_forward_state_msgs()[i].mean[2] - reference_value[i + 1][2]) ** 2
        x1_mean2error[i] = (kf_reentry.get_forward_state_msgs()[i].mean[0] - reference_value[i + 1][0]) ** 2

    # Plot: mean-squared estimation error (diagonal elements of covariance) and its estimated covariance
    # x1
    plt.figure(1)
    plt.title("Mean squared error and covariance of x1")
    plt.semilogy(t[:-1], x1_cov, label="covariance")
    plt.semilogy(t[:-1], x1_mean2error, label="mean squared error")
    plt.legend()
    plt.xlabel("Time s")
    plt.ylabel("Position variance km^2")
    plt.grid()

    # x3
    plt.figure(2)
    plt.title("Mean squared error and covariance of x3")
    plt.semilogy(t[:-1], x3_cov, label="covariance")
    plt.semilogy(t[:-1], x3_mean2error, label="mean squared error")
    plt.legend()
    plt.xlabel("Time s")
    plt.ylabel("Velocity variance (km/s)^2")
    plt.grid()

    # x5
    plt.figure(3)
    plt.title("Mean squared error and covariance of x5")
    plt.semilogy(t[:-1], x5_cov, label="covariance")
    plt.semilogy(t[:-1], x5_mean2error, label="mean squared error")
    plt.legend()
    plt.xlabel("Time s")
    plt.ylabel("Coefficient variance")
    plt.grid()

    plt.figure(4)
    plt.plot(x1, x2, label="estimation")
    plt.plot(x1_real, x2_real, label="true value")
    plt.legend()

    plt.figure(5)
    plt.plot(t[:-1], x1, label="x1")
    plt.plot(t[:-1], x1_real, label="true x1")
    plt.grid()
    plt.legend()

    plt.figure(6)
    plt.plot(t[:-1], x2, label="x2")
    plt.plot(t[:-1], x2_real, label="true x2")
    plt.legend()
    plt.grid()

    plt.figure(7)
    plt.plot(t, z[:, 0])

    plt.figure(8)
    plt.plot(t, z[:, 1])

    plt.show()
