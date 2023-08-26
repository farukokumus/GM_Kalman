# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
from numpy import linspace, inf, squeeze, concatenate
from numpy.random import normal, seed
import numpy as np
import time
import matplotlib.pyplot as plt

from ime_fgs.base import NodePort, Node
from ime_fgs.basic_nodes import MatrixNode, AdditionNode, PriorNode, EqualityNode
from ime_fgs.compound_nodes import CompoundEqualityMatrixNode
from ime_fgs.messages import GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage
from ime_fgs.divergence_measures import moment_matched_mean_cov_of_doubly_truncated_gaussian,\
    mode_matched_mean_cov_of_doubly_truncated_gaussian
from ime_fgs.ep_node import EPNode

import csv


class OptimalConstrainedControlSliceEP(Node):

    def __init__(self, A, B, C, tn, ta, tb):

        # Initialize functions for EP nodes
        super().__init__()

        def truncation_node_function(msg=None, hyperplane_normal=tn, upper_bounds=tb, lower_bounds=ta):

            # moment_matched_mean, moment_matched_cov = \
            #     moment_matched_mean_cov_of_doubly_truncated_gaussian( msg.mean, msg.cov, hyperplane_normal,
            #                                                           upper_bounds, lower_bounds )
            moment_matched_mean, moment_matched_cov = \
                mode_matched_mean_cov_of_doubly_truncated_gaussian(msg.mean, msg.cov, hyperplane_normal,
                                                                   upper_bounds, lower_bounds)

            return GaussianMeanCovMessage(moment_matched_mean, moment_matched_cov)

        # Initialize ports of the macro (slice) node
        self.port_state_in = NodePort(self, self.calc_msg_state_in)
        self.port_state_out = NodePort(self, self.calc_msg_state_out)
        self.port_meas = NodePort(self, self.calc_msg_meas)
        self.port_input = NodePort(self, self.calc_msg_input)
        self.port_process_noise = NodePort(self, self.calc_msg_process_noise)

        # Initialize all relevant nodes
        self.A_node = MatrixNode(A)
        self.add_process_noise_node = AdditionNode()
        self.process_noise_in_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))
        self.truncate_input_node = EPNode(init_port_a_out_msg=GaussianMeanCovMessage([[0]], [[1e-1]]),
                                          init_port_b_out_msg=GaussianMeanCovMessage([[0]], [[1e-1]]),
                                          node_function_a=truncation_node_function,
                                          node_function_b=truncation_node_function, name='u_trunc')
        self.add_input_node = AdditionNode()
        self.input_in_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))
        self.B_node = MatrixNode(B)
        self.equality_node = EqualityNode()
        self.C_node = MatrixNode(C)
        self.state_in_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))
        self.state_out_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))
        self.meas_in_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))

        # Connect the nodes
        self.state_in_node.port_a.connect(self.A_node.port_a)
        self.A_node.port_b.connect(self.add_input_node.port_a)
        self.process_noise_in_node.port_a.connect(self.add_process_noise_node.port_b)
        self.add_input_node.port_c.connect(self.add_process_noise_node.port_a)
        self.input_in_node.port_a.connect(self.truncate_input_node.port_a)
        self.truncate_input_node.port_b.connect(self.B_node.port_a)
        self.B_node.port_b.connect(self.add_input_node.port_b)
        self.add_process_noise_node.port_c.connect(self.equality_node.ports[0])
        self.equality_node.ports[1].connect(self.C_node.port_a)
        self.C_node.port_b.connect(self.meas_in_node.port_a)
        self.equality_node.ports[2].connect(self.state_out_node.port_a)

    def calc_msg_state_out(self):
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.A_node.port_b.update(GaussianMeanCovMessage)
        self.input_in_node.update_prior(self.port_input.in_msg)
        self.truncate_input_node.port_b.update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_c.update(GaussianMeanCovMessage)
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.add_process_noise_node.port_c.update(GaussianWeightedMeanInfoMessage)

        self.meas_in_node.update_prior(self.port_meas.in_msg, GaussianWeightedMeanInfoMessage)
        self.C_node.port_a.update(GaussianWeightedMeanInfoMessage)

        return self.equality_node.ports[2].update(GaussianMeanCovMessage)

    def calc_msg_state_in(self):
        # Caution: Untested
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.input_in_node.update_prior(self.port_input.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        self.meas_in_node.update_prior(self.port_meas.in_msg, GaussianWeightedMeanInfoMessage)

        self.C_node.port_a.update(GaussianWeightedMeanInfoMessage)
        self.equality_node.ports[0].update(GaussianMeanCovMessage)
        self.add_process_noise_node.port_a.update(GaussianMeanCovMessage)
        self.truncate_input_node.port_b.update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_a.update(GaussianWeightedMeanInfoMessage)

        return self.A_node.port_a.update(GaussianWeightedMeanInfoMessage)

    def calc_msg_output(self):
        """
        Update schedule to calculate the output from input and current and future state priors.
        :return: Forward message of the output (y)
        """
        # Caution: Untested
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.meas_in_node.update_prior(self.port_meas.in_msg)
        self.input_in_node.update_prior(self.port_input.in_msg)
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)

        self.A_node.port_b.update()
        self.truncate_input_node.port_b.update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_c.update(GaussianMeanCovMessage)
        self.add_process_noise_node.port_c.update(GaussianWeightedMeanInfoMessage)
        self.equality_node.ports[1].update(GaussianMeanCovMessage)

        return self.C_node.port_b.update()

    def calc_msg_meas(self):
        # Caution: Untested
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.meas_in_node.update_prior(self.port_meas.in_msg)
        self.input_in_node.update_prior(self.port_input.in_msg)
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)

        self.A_node.port_b.update()
        self.truncate_input_node.port_b.update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_c.update(GaussianMeanCovMessage)
        self.add_process_noise_node.port_c.update(GaussianWeightedMeanInfoMessage)
        self.equality_node.ports[1].update(GaussianMeanCovMessage)

        return self.C_node.port_b.update()

    def calc_msg_input(self):
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        self.meas_in_node.update_prior(self.port_meas.in_msg, GaussianWeightedMeanInfoMessage)
        self.input_in_node.update_prior(self.port_input.in_msg)

        self.A_node.port_b.update()
        self.C_node.port_a.update(GaussianWeightedMeanInfoMessage)
        self.equality_node.ports[0].update(GaussianMeanCovMessage)
        self.add_process_noise_node.port_a.update(GaussianMeanCovMessage)
        self.add_input_node.port_b.update(GaussianWeightedMeanInfoMessage)
        self.B_node.port_a.update(GaussianWeightedMeanInfoMessage)

        return self.truncate_input_node.port_a.update(GaussianMeanCovMessage)

    def calc_msg_process_noise(self):
        # Caution: Untested
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        self.meas_in_node.update_prior(self.port_meas.in_msg, GaussianWeightedMeanInfoMessage)
        self.input_in_node.update_prior(self.port_input.in_msg)

        self.A_node.port_b.update()

        self.C_node.port_a.update(GaussianWeightedMeanInfoMessage)
        self.equality_node.ports[0].update(GaussianMeanCovMessage)
        self.truncate_input_node.port_b.update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_b.update(GaussianMeanCovMessage)

        return self.add_process_noise_node.port_b.update(GaussianMeanCovMessage)

    def get_ports(self):
        return [self.port_state_in, self.port_state_out, self.port_meas, self.port_input, self.port_process_noise]


class PredictiveStochasticController(object):

    def __init__(self, A, B, C, tn, ta, tb, initial_state_msg, input_noise_cov=None, process_noise_cov=None,
                 meas_noise_cov=None, slice_type=None):
        """
        Initialize a Kalman filter object given an initial state.

        :param A: System matrix to be used as the default in all time slices.
        :param B: Input matrix to be used as the default in all time slices.
        :param C: Output matrix to be used as the default in all time slices.
        :param tn: Normal vector w.r.t. the hyperplane corresponding to the truncation of the input signal.
        (to be used as the default in all time slices.)
        :param ta: Lower limit vector w.r.t. the hyperplane corresponding to the truncation of the input signal.
        (to be used as the default in all time slices.)
        :param tb: Lower limit vector w.r.t. the hyperplane corresponding to the truncation of the input signal.
        (to be used as the default in all time slices.)
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
        self.A = A
        self.B = B
        self.C = C
        self.tn = tn
        self.ta = ta
        self.tb = tb
        self.slice_type = slice_type
        self.input_noise_cov = input_noise_cov
        self.process_noise_cov = process_noise_cov
        self.meas_noise_cov = meas_noise_cov

        # Initialize factor graph
        self.slices = []
        self.initial_state = PriorNode(initial_state_msg)

    def add_slice(self, input_val, meas_val, A=None, B=None, C=None, tn=None, ta=None, tb=None, input_noise_msg=None,
                  process_noise_msg=None, meas_noise_msg=None):
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
        :param tn: Normal vector w.r.t. the hyperplane corresponding to the truncation of the input signal for this time
          slice. If not provided, the default one passed to the Kalman filter
        :param ta: Lower limit vector w.r.t. the hyperplane corresponding to the truncation of the input signal for this
          time slice. If not provided, the default one passed to the Kalman filter
        :param tb: Lower limit vector w.r.t. the hyperplane corresponding to the truncation of the input signal for this
          time slice. If not provided, the default one passed to the Kalman filter
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
            A = self.A
        if B is None:
            B = self.B
        if C is None:
            C = self.C
        if tn is None:
            tn = self.tn
        if ta is None:
            ta = self.ta
        if tb is None:
            tb = self.tb
        if input_noise_msg is None:
            input_noise_msg = GaussianMeanCovMessage([[0]], self.input_noise_cov)
        if process_noise_msg is None:
            process_noise_msg = GaussianMeanCovMessage([[0]], self.process_noise_cov)
        if meas_noise_msg is None:
            meas_noise_msg = GaussianMeanCovMessage([[0]], self.meas_noise_cov)

        new_slice = self.slice_type(A, B, C, tn, ta, tb)

        new_slice.port_meas.connect(PriorNode(GaussianMeanCovMessage(np.atleast_2d(meas_val),
                                                                     [[0]]) + meas_noise_msg).port_a)
        new_slice.port_input.connect(PriorNode(GaussianMeanCovMessage(np.atleast_2d(input_val),
                                                                      [[0]]) + input_noise_msg).port_a)
        new_slice.port_process_noise.connect(PriorNode(process_noise_msg).port_a)

        if len(self.slices) == 0:
            self.initial_state.port_a.connect(new_slice.port_state_in)
        else:
            self.slices[-1].port_state_out.connect(new_slice.port_state_in)

        new_slice.port_state_out.update()
        self.slices.append(new_slice)

        return new_slice.port_state_out.out_msg

    def add_slices(self, input_vals, meas_vals, As=None, Bs=None, Cs=None, input_noise_msgs=None,
                   process_noise_msgs=None, meas_noise_msgs=None):
        """
        Adds a sequence of new input and output measurements to the Kalman filter factor graph and updates the current
        state estimate, incorporating the new measurements.

        :param input_vals: List of input values in each time slice. Note that input noise is already taken into account
          in the system model. Time-varying input noise covariances can be passed separately.
        :param meas_vals: List of measured values in each time slice. Note that measurement noise is already taken into
          account in the system model. Time-varying measurement noise covariances can be passed separately.
        :param As: List of system matrices to employ in each time slice. If not provided, the default one passed to the
          Kalman filter constructor is used.
        :param Bs: List of input matrices to employ in each time slice. If not provided, the default one passed to the
          Kalman filter constructor is used.
        :param Cs: List of output matrices to employ in each time slice. If not provided, the default one passed to the
          Kalman filter constructor is used.
        :param input_noise_msgs: List of messages specifying knowledge about the input noise signal in each time
          slice. If not provided, the default input noise covariance passed to the Kalman filter constructor is used.
        :param process_noise_msgs: List of messages specifying knowledge about the process noise signal in each time
          slice. If not provided, the default process noise covariance passed to the Kalman filter constructor is used.
        :param meas_noise_msgs: List of messages specifying knowledge about the measurement noise signal in each time
          slice. If not provided, the default measurement noise covariance passed to the Kalman filter constructor is
          used.
        :return: The update Kalman filter object, including the new slices.
        """
        # Input handling
        num_slices = len(input_vals)
        if As is None:
            As = [self.A for _ in range(num_slices)]
        if Bs is None:
            Bs = [self.B for _ in range(num_slices)]
        if Cs is None:
            Cs = [self.C for _ in range(num_slices)]
        if input_noise_msgs is None:
            input_noise_msgs = [GaussianMeanCovMessage(0, self.input_noise_cov) for _ in range(num_slices)]
        if process_noise_msgs is None:
            process_noise_msgs = [GaussianMeanCovMessage(0, self.process_noise_cov) for _ in range(num_slices)]
        if meas_noise_msgs is None:
            meas_noise_msgs = [GaussianMeanCovMessage(0, self.meas_noise_cov) for _ in range(num_slices)]
        # Assert the same length of the input signals
        assert(len(input_vals) == len(meas_vals) == len(As) == len(Bs) == len(Cs) == len(input_noise_msgs)
               == len(process_noise_msgs) == len(meas_noise_msgs))

        for kk in range(num_slices):
            self.add_slice(input_vals[kk], meas_vals[kk], As[kk], Bs[kk], Cs[kk], input_noise_msgs[kk],
                           process_noise_msgs[kk], meas_noise_msgs[kk])

        return self

    def forward_message_passing(self):
        """
        Compute forward messages of the optimal control factor graph.

        :return: Nothing.
        """
        for slice in self.slices:
            slice.calc_msg_state_out()
            slice.port_state_out.update(GaussianMeanCovMessage)
        return

    def backward_message_passing(self):
        """
        Compute backward messages of the optimal control factor graph.

        :return: Nothing.
        """
        for slice in self.slices[::-1]:
            slice.calc_msg_state_in()
            slice.port_state_in.update(GaussianWeightedMeanInfoMessage)

            slice.calc_msg_input()
            slice.port_input.update(GaussianWeightedMeanInfoMessage)

            slice.calc_msg_process_noise()
            slice.port_process_noise.update(GaussianWeightedMeanInfoMessage)

            slice.calc_msg_meas()
            slice.port_meas.update(GaussianMeanCovMessage)

        return

    def get_state_out_msgs(self):
        return [slice.port_state_out.out_msg for slice in self.slices]

    def get_state_in_msgs(self):
        return [slice.port_state_in.in_msg for slice in self.slices]

    def get_state_out_marginal_msgs(self):
        # todo: This appears clumsy and should be invoked by the marginal() function of the port
        state_out_marginals = [slice.port_state_out.in_msg.combine(
            slice.port_state_out.out_msg.convert(GaussianWeightedMeanInfoMessage)) for slice in self.slices]
        return [marginal.convert(GaussianMeanCovMessage) for marginal in state_out_marginals]

    def get_state_in_marginal_msgs(self):
        # todo: This appears clumsy and should be invoked by the marginal() function of the port
        state_in_marginals = [slice.A_node.port_a.out_msg.combine(
            slice.A_node.port_a.in_msg.convert(GaussianWeightedMeanInfoMessage)) for slice in self.slices]
        return [marginal.convert(GaussianMeanCovMessage) for marginal in state_in_marginals]

    def get_output_marginal_msgs(self):
        # todo: This appears clumsy and should be invoked by the marginal() function of the port
        output_marginals = [slice.port_meas.in_msg.convert(GaussianWeightedMeanInfoMessage).combine(
            slice.port_meas.out_msg.convert(GaussianWeightedMeanInfoMessage)) for slice in self.slices]
        return [marginal.convert(GaussianMeanCovMessage) for marginal in output_marginals]

    def get_input_marginal_msgs(self):
        # todo: This appears clumsy and should be invoked by the marginal() function of the port
        # Amounts to returning the marginal between prior and truncation node!
        input_marginals = [slice.port_input.out_msg.combine(
            slice.port_input.in_msg.convert(GaussianWeightedMeanInfoMessage)) for slice in self.slices]

        # Amounts to returning the marginal after both prior and truncation node!
        # input_marginals = [slice.truncate_input_node.port_b.in_msg.combine(
        # slice.truncate_input_node.port_b.out_msg.convert(GaussianWeightedMeanInfoMessage)) for slice in self.slices]
        #
        # This amounts to performing the projection on the marginal
        # input_marginals = [marginal.convert( GaussianMeanCovMessage ) for marginal in input_marginals]
        # moment_matched_mean_cov = [moment_matched_mean_cov_of_doubly_truncated_gaussian( m.mean, m.cov, 1, 0.5, -0.5 )
        #                            for m in input_marginals ]
        # marginal = []
        # for ii in range(50):
        #     marginal.append(GaussianMeanCovMessage(moment_matched_mean_cov[ii][0], moment_matched_mean_cov[ii][1]))

        # return marginal
        return [marginal.convert(GaussianMeanCovMessage) for marginal in input_marginals]

    def get_process_disturbance_marginal_msgs(self):
        # todo: This appears clumsy and should be invoked by the marginal() function of the port
        # Amounts to returning the marginal between prior and truncation node!
        process_disturbance_marginals = [slice.port_process_noise.out_msg.combine(
            slice.port_process_noise.in_msg.convert(GaussianWeightedMeanInfoMessage)) for slice in self.slices]

        # return marginal
        return [marginal.convert(GaussianMeanCovMessage) for marginal in process_disturbance_marginals]


if __name__ == '__main__':
    #
    # Example due Christian Hoffmann, University of Lübeck

    # Define system model
    A = np.atleast_2d(1.0)
    C = np.atleast_2d(1.0)
    B = np.atleast_2d(0.5)
    tn = 10.0
    ta = -10
    tb = 10

    nx = 1
    nu = 1

    # Number of time instants
    num_meas = 400

    # Initialize inference
    i_KF = linspace(0, 0, num_meas)
    V_i_KF = linspace(1e-8, 1e-8, num_meas)

    for jj in range(2):

        seed(1)
        z_real = 1 * concatenate([linspace(0, 0, 1 * num_meas // 5), linspace(2, 2, 3 * num_meas // 5),
                                  linspace(0, 0, 1 * num_meas // 5)])
        z = z_real + 0 * normal(0, 1, num_meas)
        u = linspace(0, 0, num_meas)

        R = np.atleast_2d(1e-1)
        Q = np.atleast_2d(1e0)

        x0 = np.atleast_2d(0.)
        initial_state = GaussianMeanCovMessage(mean=x0, cov=[[1e-12]])
        lqg = PredictiveStochasticController(A, B, C, tn, ta, tb, initial_state,
                                             input_noise_cov=np.linalg.inv(R),
                                             process_noise_cov=V_i_KF[0],
                                             meas_noise_cov=np.linalg.inv(Q),
                                             slice_type=OptimalConstrainedControlSliceEP)

        for ii in range(num_meas):
            i_KF_msg = GaussianMeanCovMessage(np.atleast_2d(i_KF[ii]), np.atleast_2d(V_i_KF[ii]))
            lqg.add_slice(u[ii], z[ii], process_noise_msg=i_KF_msg)

        # Undefined terminal state
        lqg.slices[-1].port_state_out.connect(PriorNode(GaussianWeightedMeanInfoMessage([[0]], [[0]])).port_a)

        start = time.perf_counter()

        for ii in range(1):
            lqg.backward_message_passing()
            lqg.forward_message_passing()
            u_LQG = [input.mean for input in lqg.get_input_marginal_msgs()]
            x_LQG = [state.mean for state in lqg.get_state_out_marginal_msgs()]
            y_LQG = [output.mean for output in lqg.get_output_marginal_msgs()]
            # i_fg = [process_disturbance.mean for process_disturbance in lqg.get_process_disturbance_marginal_msgs( )]

        end = time.perf_counter()

        print('Time elapsed to compute optimal control input via factor graphs and EP: ' + str(end - start))

        # y_LQG = np.atleast_2d( C @ x0 )
        # for ii in range(num_meas):
        #     # x_LQG = np.concatenate( (x_LQG, np.atleast_2d(A @ x_LQG[ii] + B @ u_LQG[ii])), axis=0)
        #     y_LQG = np.concatenate( (y_LQG, np.atleast_2d( C @ x_LQG[ii] )), axis=0 )

        # plotting
        plt.figure()
        # plt.plot( z, 'k-', label='Noisy Reference' )
        plt.plot(squeeze([y_LQG]), 'm-', label='Controlled Output')
        # plt.plot( squeeze( [i_fg] ), 'k-', label='Process Disturbance' )
        plt.plot(squeeze([u_LQG]), 'r-', label='Marginalized Inputs')

        plt.plot(z_real, color='g', label='Reference')
        plt.legend()
        plt.title('Estimate vs. Time Instant', fontweight='bold')
        plt.xlabel('Time Instant')
        plt.ylabel('')
        plt.show()

        # with open( 'TikZ-IterativeLearningControlEP.dat', 'w', newline='' ) as csvfile:
        #     fieldnames = ['t', 'y', 'u', 'r']
        #     writer = csv.DictWriter( csvfile, fieldnames=fieldnames, delimiter='\t' )
        #     writer.writeheader( )
        #     for ii in range(len(z))[0::4]:
        #         writer.writerow( {fieldnames[0]: str(ii), fieldnames[1]: str( squeeze( y_LQG[ii] ) ),
        #                           fieldnames[2]: str( squeeze( u_LQG[ii] ) ), fieldnames[3]: str( squeeze( z[ii] ) )})

        # Kalman Filtering to infer process disturbance
        z_real = y_LQG
        process_disturbance = 1 * concatenate([linspace(0, 0, 2 * num_meas // 5),
                                               linspace(-0.1, -0.1, 1 * num_meas // 5),
                                               linspace(0, 0, 2 * num_meas // 5)])

        x_actual = np.atleast_2d(x_LQG[0])
        y_actual = np.atleast_2d(C @ x_actual[0])
        for ii in range(num_meas - 1):
            x_actual = np.concatenate((x_actual, np.atleast_2d(A @ x_actual[ii] + B @ u_LQG[ii + 1]
                                                               + process_disturbance[ii])), axis=0)
            y_actual = np.concatenate((y_actual, np.atleast_2d(C @ x_actual[ii + 1])), axis=0)

        # plotting
        plt.figure()
        plt.plot(squeeze([y_LQG]), 'm-', label='Ideal Output')
        plt.plot(squeeze([y_actual]), 'b-', label='Actual Output')
        plt.plot(squeeze([process_disturbance]), 'k-', label='Process Disturbance')
        plt.plot(squeeze([u_LQG]), 'r-', label='Marginalized Inputs')
        plt.legend()
        plt.title('Estimate vs. Time Instant', fontweight='bold')
        plt.xlabel('Time Instant')
        plt.ylabel('')
        plt.show()

        i_KF = linspace(0, 0, num_meas)
        V_i_KF = linspace(1e-8, 1e-8, num_meas)

        z = y_actual
        u = u_LQG

        R = np.atleast_2d(1e12)
        Q = np.atleast_2d(1e12)

        x0 = np.atleast_2d(x_LQG[0])
        initial_state = GaussianMeanCovMessage(mean=x0, cov=[[1e-12]])
        KF = PredictiveStochasticController(A, B, C, tn, ta, tb, initial_state,
                                            input_noise_cov=np.linalg.inv(R),
                                            process_noise_cov=V_i_KF[0],
                                            meas_noise_cov=np.linalg.inv(Q),
                                            slice_type=OptimalConstrainedControlSliceEP)

        for ii in range(num_meas):
            i_KF_msg = GaussianMeanCovMessage(np.atleast_2d(i_KF[ii]), np.atleast_2d(V_i_KF[ii]))
            KF.add_slice(u[ii], z[ii], process_noise_msg=i_KF_msg)

        # Undefined terminal state
        KF.slices[-1].port_state_out.connect(PriorNode(GaussianWeightedMeanInfoMessage([[0]], [[0]])).port_a)

        start = time.perf_counter()

        for ii in range(1):
            KF.forward_message_passing()
            KF.backward_message_passing()
            u_KF = [input.mean for input in KF.get_input_marginal_msgs()]
            x_KF = [state.mean for state in KF.get_state_out_marginal_msgs()]
            i_KF = [process_disturbance.mean for process_disturbance in KF.get_process_disturbance_marginal_msgs()]
            V_i_KF = [process_disturbance.cov for process_disturbance in KF.get_process_disturbance_marginal_msgs()]

        end = time.perf_counter()

        # plotting
        plt.figure()
        # plt.plot( squeeze( [y_KF] ), 'm-', label='Inferred Output' )
        plt.plot(squeeze([i_KF]), 'k-', label='Inferred Process Disturbance')
        # plt.plot( squeeze( [u_KF] ), 'r-', label='Inferred Inputs' )

        # plt.plot( z_real, color='g', label='Reference' )
        plt.legend()
        plt.title('Estimate vs. Time Instant', fontweight='bold')
        plt.xlabel('Time Instant')
        plt.ylabel('')
        plt.show()
