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
import scipy as sp
from scipy import optimize
import time
import matplotlib.pyplot as plt
# import control as ctrl

from ime_fgs.base import NodePort, Node
from ime_fgs.basic_nodes import MatrixNode, AdditionNode, PriorNode, EqualityNode
from ime_fgs.compound_nodes import CompoundEqualityMatrixNode
from ime_fgs.messages import GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage
from ime_fgs.advanced_nodes import TruncationNode
from ime_fgs.divergence_measures import moment_matched_mean_cov_of_doubly_truncated_gaussian


class OptimalControlSliceNaive(Node):

    def __init__(self, A, B, C):
        # Initialize ports of the macro (slice) node
        super().__init__()
        self.port_state_in = NodePort(self, self.calc_msg_state_in)
        self.port_state_out = NodePort(self, self.calc_msg_state_out)
        self.port_meas = NodePort(self, self.calc_msg_meas)
        self.port_input = NodePort(self, self.calc_msg_input)
        self.port_process_noise = NodePort(self, self.calc_msg_process_noise)

        # Initialize all relevant nodes
        self.A_node = MatrixNode(A)
        self.add_process_noise_node = AdditionNode()
        self.process_noise_in_node = PriorNode(GaussianMeanCovMessage(0, inf))
        self.add_input_node = AdditionNode()
        self.input_in_node = PriorNode(GaussianMeanCovMessage(0, inf))
        self.B_node = MatrixNode(B)
        self.equality_node = EqualityNode()
        self.C_node = MatrixNode(C)
        self.state_in_node = PriorNode(GaussianMeanCovMessage(0, inf))
        self.state_out_node = PriorNode(GaussianMeanCovMessage(0, inf))
        self.meas_in_node = PriorNode(GaussianMeanCovMessage(0, inf))

        # Connect the nodes
        self.state_in_node.port_a.connect(self.A_node.port_a)
        self.A_node.port_b.connect(self.add_process_noise_node.port_a)
        self.process_noise_in_node.port_a.connect(self.add_process_noise_node.port_b)
        self.add_process_noise_node.port_c.connect(self.add_input_node.port_a)
        self.input_in_node.port_a.connect(self.B_node.port_a)
        self.B_node.port_b.connect(self.add_input_node.port_b)
        self.add_input_node.port_c.connect(self.equality_node.ports[0])
        self.equality_node.ports[1].connect(self.C_node.port_a)
        self.C_node.port_b.connect(self.meas_in_node.port_a)
        self.equality_node.ports[2].connect(self.state_out_node.port_a)

    def calc_msg_state_out(self):
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.A_node.port_b.update(GaussianMeanCovMessage)
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.add_process_noise_node.port_c.update(GaussianMeanCovMessage)
        self.input_in_node.update_prior(self.port_input.in_msg)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_c.update(GaussianWeightedMeanInfoMessage)

        self.meas_in_node.update_prior(self.port_meas.in_msg, GaussianWeightedMeanInfoMessage)
        self.C_node.port_a.update(GaussianWeightedMeanInfoMessage)

        return self.equality_node.ports[2].update(GaussianMeanCovMessage)

    def calc_msg_state_in(self):
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.input_in_node.update_prior(self.port_input.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        self.meas_in_node.update_prior(self.port_meas.in_msg, GaussianWeightedMeanInfoMessage)

        self.C_node.port_a.update(GaussianWeightedMeanInfoMessage)
        self.equality_node.ports[0].update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_a.update(GaussianMeanCovMessage)
        self.add_process_noise_node.port_a.update(GaussianWeightedMeanInfoMessage)

        return self.A_node.port_a.update(GaussianWeightedMeanInfoMessage)

    def calc_msg_output(self):
        """
        Update schedule to calculate the output from input and current and future state priors.
        :return: Forward message of the output (y)
        """
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.meas_in_node.update_prior(self.port_meas.in_msg)
        self.input_in_node.update_prior(self.port_input.in_msg)
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)

        self.A_node.port_b.update()
        self.add_process_noise_node.port_c.update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_c.update(GaussianWeightedMeanInfoMessage)
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
        self.add_process_noise_node.port_c.update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_c.update(GaussianWeightedMeanInfoMessage)
        self.equality_node.ports[1].update(GaussianMeanCovMessage)

        return self.C_node.port_b.update()

    def calc_msg_input(self):
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        self.meas_in_node.update_prior(self.port_meas.in_msg, GaussianWeightedMeanInfoMessage)
        self.input_in_node.update_prior(self.port_input.in_msg)

        self.A_node.port_b.update()
        self.add_process_noise_node.port_c.update(GaussianMeanCovMessage)
        self.C_node.port_a.update(GaussianWeightedMeanInfoMessage)
        self.equality_node.ports[0].update(GaussianMeanCovMessage)
        self.add_input_node.port_b.update(GaussianWeightedMeanInfoMessage)

        return self.B_node.port_a.update(GaussianMeanCovMessage)

    def calc_msg_process_noise(self):
        # Caution: Untested
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        self.meas_in_node.update_prior(self.port_meas.in_msg)
        self.input_in_node.update_prior(self.port_input.in_msg)

        self.A_node.port_b.update()

        self.C_node.port_a.update(GaussianWeightedMeanInfoMessage)
        self.equality_node.ports[0].update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_a.update(GaussianMeanCovMessage)

        return self.add_process_noise_node.port_b.update(GaussianMeanCovMessage)

    def get_ports(self):
        return [self.port_state_in, self.port_state_out, self.port_meas, self.port_input, self.port_process_noise]


class OptimalConstrainedControlSliceNaive(Node):

    def __init__(self, A, B, C, tn, ta, tb):
        # Initialize ports of the macro (slice) node
        super().__init__()
        self.port_state_in = NodePort(self, self.calc_msg_state_in)
        self.port_state_out = NodePort(self, self.calc_msg_state_out)
        self.port_meas = NodePort(self, self.calc_msg_meas)
        self.port_input = NodePort(self, self.calc_msg_input)
        self.port_process_noise = NodePort(self, self.calc_msg_process_noise)

        # Initialize all relevant nodes
        self.A_node = MatrixNode(A)
        self.add_process_noise_node = AdditionNode()
        self.process_noise_in_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))
        self.truncate_input_node = TruncationNode(hyperplane_normal=tn, upper_bounds=tb, lower_bounds=ta)
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
        self.A_node.port_b.connect(self.add_process_noise_node.port_a)
        self.process_noise_in_node.port_a.connect(self.add_process_noise_node.port_b)
        self.add_process_noise_node.port_c.connect(self.add_input_node.port_a)
        self.input_in_node.port_a.connect(self.truncate_input_node.port_a)
        self.truncate_input_node.port_b.connect(self.B_node.port_a)
        self.B_node.port_b.connect(self.add_input_node.port_b)
        self.add_input_node.port_c.connect(self.equality_node.ports[0])
        self.equality_node.ports[1].connect(self.C_node.port_a)
        self.C_node.port_b.connect(self.meas_in_node.port_a)
        self.equality_node.ports[2].connect(self.state_out_node.port_a)

    def calc_msg_state_out(self):
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.A_node.port_b.update(GaussianMeanCovMessage)
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.add_process_noise_node.port_c.update(GaussianMeanCovMessage)
        self.input_in_node.update_prior(self.port_input.in_msg)
        self.truncate_input_node.port_b.update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_c.update(GaussianWeightedMeanInfoMessage)

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
        self.truncate_input_node.port_b.update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_a.update(GaussianMeanCovMessage)
        self.add_process_noise_node.port_a.update(GaussianWeightedMeanInfoMessage)

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
        self.add_process_noise_node.port_c.update(GaussianMeanCovMessage)
        self.truncate_input_node.port_b.update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_c.update(GaussianWeightedMeanInfoMessage)
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
        self.add_process_noise_node.port_c.update(GaussianMeanCovMessage)
        self.truncate_input_node.port_b.update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_c.update(GaussianWeightedMeanInfoMessage)
        self.equality_node.ports[1].update(GaussianMeanCovMessage)

        return self.C_node.port_b.update()

    def calc_msg_input(self):
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        self.meas_in_node.update_prior(self.port_meas.in_msg, GaussianWeightedMeanInfoMessage)
        self.input_in_node.update_prior(self.port_input.in_msg)

        self.A_node.port_b.update()
        self.add_process_noise_node.port_c.update(GaussianMeanCovMessage)
        self.C_node.port_a.update(GaussianWeightedMeanInfoMessage)
        self.equality_node.ports[0].update(GaussianMeanCovMessage)
        self.add_input_node.port_b.update(GaussianWeightedMeanInfoMessage)
        self.B_node.port_a.update(GaussianWeightedMeanInfoMessage)

        return self.truncate_input_node.port_a.update(GaussianMeanCovMessage)

    def calc_msg_process_noise(self):
        # Caution: Untested
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        self.meas_in_node.update_prior(self.port_meas.in_msg)
        self.input_in_node.update_prior(self.port_input.in_msg)

        self.A_node.port_b.update()

        self.C_node.port_a.update(GaussianWeightedMeanInfoMessage)
        self.equality_node.ports[0].update(GaussianMeanCovMessage)
        self.truncate_input_node.port_b.update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_a.update(GaussianMeanCovMessage)

        return self.add_process_noise_node.port_b.update(GaussianMeanCovMessage)

    def get_ports(self):
        return [self.port_state_in, self.port_state_out, self.port_meas, self.port_input, self.port_process_noise]


class PredictiveStochasticController(object):

    def __init__(self, A, B, C, tn, ta, tb, initial_state_msg, input_noise_cov=None, process_noise_cov=None,
                 meas_noise_cov=None, slice_type=OptimalControlSliceNaive):
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

        new_slice.port_meas.connect(PriorNode(GaussianMeanCovMessage(meas_val, [[0]]) + meas_noise_msg).port_a)
        new_slice.port_input.connect(PriorNode(GaussianMeanCovMessage(input_val, [[0]]) + input_noise_msg).port_a)
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
        assert (len(input_vals) == len(meas_vals) == len(As) == len(Bs) == len(Cs) == len(input_noise_msgs)
                == len(process_noise_msgs) == len(meas_noise_msgs))

        for kk in range(num_slices):
            self.add_slice(input_vals[kk], meas_vals[kk], As[kk], Bs[kk], Cs[kk], input_noise_msgs[kk],
                           process_noise_msgs[kk], meas_noise_msgs[kk])

        return self

    def backward_message_passing(self):
        """
        Compute backward messages of the Kalman filter.

        :return: Nothing.
        """
        for slice in self.slices[::-1]:
            slice.calc_msg_state_in()
            slice.port_state_in.update(GaussianWeightedMeanInfoMessage)

            slice.calc_msg_input()
            slice.port_input.update(GaussianWeightedMeanInfoMessage)

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

    def get_input_marginal_msgs(self):
        # todo: This appears clumsy and should be invoked by the marginal() function of the port
        # Amounts to returning the marginal between prior and truncation node!
        input_marginals = [slice.port_input.out_msg.combine(
            slice.port_input.in_msg.convert(GaussianWeightedMeanInfoMessage)) for slice in self.slices]

        # Amounts to returning the marginal after both prior and truncation node!
        # input_marginals = [slice.truncate_input_node.port_b.in_msg.combine(
        #     slice.truncate_input_node.port_b.out_msg.convert(GaussianWeightedMeanInfoMessage))
        #     for slice in self.slices]
        #
        # This amounts to performing the projection on the marginal
        # input_marginals = [marginal.convert( GaussianMeanCovMessage ) for marginal in input_marginals]
        # moment_matched_mean_cov = [moment_matched_mean_cov_of_doubly_truncated_gaussian( m.mean, m.cov, 1, 0.5, -0.5 )
        #                            for m in input_marginals ]
        # marginal = []
        # for ii in range(50):
        #     marginal.append(GaussianMeanCovMessage(moment_matched_mean_cov[ii][0], moment_matched_mean_cov[ii][1]))

        # return input_marginals
        return [marginal.convert(GaussianMeanCovMessage) for marginal in input_marginals]


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
    B = np.atleast_2d(1.0)
    tn = 1.0
    ta = -0.5
    tb = 0.5

    nx = 1
    nu = 1

    # Number of time instants
    num_meas = 50

    seed(1)
    z_real = 0 * concatenate([linspace(0, 0, 1 * num_meas // 5), linspace(10, 10, 3 * num_meas // 5),
                              linspace(0, 0, 1 * num_meas // 5)])
    z = z_real + 0 * normal(0, 1, num_meas)
    u = linspace(0, 0, num_meas)

    start = time.perf_counter()

    R = np.atleast_2d(1e1)
    Q = np.atleast_2d(1e0)

    initial_state = GaussianMeanCovMessage(mean=[[10]], cov=[[1e-8]])
    lqg = PredictiveStochasticController(A, B, C, tn, ta, tb, initial_state,
                                         input_noise_cov=np.linalg.inv(R),
                                         process_noise_cov=[[1e-12]],
                                         meas_noise_cov=np.linalg.inv(Q),
                                         slice_type=OptimalConstrainedControlSliceNaive)

    for ii in range(num_meas):
        lqg.add_slice(np.atleast_2d(u[ii]), np.atleast_2d(z[ii]))

    # Undefined terminal state
    lqg.slices[-1].port_state_out.connect(PriorNode(GaussianWeightedMeanInfoMessage([[0]], [[0]])).port_a)

    lqg.backward_message_passing()
    u_fg = [input.mean for input in lqg.get_input_marginal_msgs()]

    end = time.perf_counter()

    print('Time elapsed to compute optimal control input: ' + str(end - start))

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

    A_ineq = np.concatenate((-np.eye(num_meas), np.eye(num_meas)), axis=0)

    b_ineq = 0.5 * np.concatenate((np.ones([num_meas, 1]), np.ones([num_meas, 1])), axis=0)

    u0 = 0 * np.random.randn(num_meas)

    def loss(x, sign=1.):
        return sign * (0.5 * x.T @ H @ x + c @ x + c0)

    def jac(x, sign=1.):
        return sign * (x.T @ H + c)

    cons = {'type': 'ineq',
            'fun': lambda x: b_ineq.squeeze() - A_ineq @ x,
            'jac': lambda x: -A_ineq}

    opt = {'disp': True}

    res_cons = optimize.minimize(loss, u0, jac=jac, constraints=cons,
                                 method='SLSQP', options=opt)

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
    y_fg = np.atleast_2d(x0)
    for ii in range(num_meas):
        y_fg = np.concatenate((y_fg, A * y_fg[ii] + B * u_fg[ii]), axis=0)

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
    plt.plot(squeeze([y_fg]), 'm-', label='factor graph solution')

    plt.plot(z_real, color='g', label='truth value')
    plt.legend()
    plt.title('Estimate vs. iteration step', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('')

    plt.figure()
    plt.plot(squeeze(u_fg), 'r-', label='marginalized inputs')
    plt.plot(squeeze(u_cons), 'k-', label='quadratic programming solution')
    plt.plot(squeeze(u_unc), 'b-', label='unconstrained quadratic programming solution')
    plt.legend()
    plt.title('Estimate vs. iteration step', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('')

    plt.figure()
    plt.plot(squeeze([fwd_estimate.cov for fwd_estimate in lqg.get_state_out_msgs()]), label='error estimate')
    plt.plot(squeeze([smoothed_estimate.cov for smoothed_estimate in lqg.get_state_in_marginal_msgs()]),
             'r-', label='a posteriori error estimate')
    plt.title('Estimated error covariance vs. iteration step', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('')
    plt.show()
