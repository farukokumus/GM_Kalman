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

2017/09/26 Eike Petersen and Felix Vollmer
2017/11/06 Christian Hoffmann
'''

import numpy as np
import numpy.random as rnd
import time
import matplotlib.pyplot as plt

from ime_fgs.advanced_nodes import UnscentedNode
from ime_fgs.base import NodePort, Node, NodePortType
from ime_fgs.basic_nodes import MatrixNode, AdditionNode, PriorNode, EqualityNode
from ime_fgs.compound_nodes import CompoundEqualityMatrixNode
from ime_fgs.messages import GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage


class UnscentedKalmanSliceNaive(Node):
    def __init__(self, func, B, C, method=1, alpha=0.9, inv_func=None):
        super().__init__('UnscentedKalmanSliceNaive')

        # Initialize ports of the macro (slice) node
        self.port_state_in = NodePort(self, self.calc_msg_state_in, NodePortType.InPort, name="port_state_in")
        self.port_state_out = NodePort(self, self.calc_msg_state_out, NodePortType.OutPort, name="port_state_out")
        self.port_meas = NodePort(self, self.calc_msg_meas, NodePortType.OutPort, name="port_meas")
        self.port_input = NodePort(self, self.calc_msg_input, NodePortType.InPort, name="port_input")
        self.port_process_noise = NodePort(self, self.calc_msg_process_noise, NodePortType.InPort,
                                           name="port_process_noise")

        # Initialize all relevant nodes
        # TODO Initialize using correct dimensions of variables!
        self.A_node = UnscentedNode(func)
        self.add_process_noise_node = AdditionNode()
        self.process_noise_in_node = PriorNode(GaussianMeanCovMessage(0, np.inf))
        self.add_input_node = AdditionNode()
        self.input_in_node = PriorNode(GaussianMeanCovMessage(0, np.inf))
        self.B_node = MatrixNode(B)
        self.equality_node = EqualityNode()
        self.C_node = MatrixNode(C)
        self.state_in_node = PriorNode(GaussianMeanCovMessage(0, np.inf))
        self.state_out_node = PriorNode(GaussianMeanCovMessage(0, np.inf))
        self.meas_in_node = PriorNode(GaussianMeanCovMessage(0, np.inf))

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

        return self.A_node.port_a.update()

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
        # Caution: Untested
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        self.meas_in_node.update_prior(self.port_meas.in_msg)
        self.input_in_node.update_prior(self.port_input.in_msg)

        self.A_node.port_b.update()
        self.add_process_noise_node.port_c.update(GaussianMeanCovMessage)
        self.C_node.port_a.update(GaussianWeightedMeanInfoMessage)
        self.equality_node.ports[0].update(GaussianMeanCovMessage)
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

        self.C_node.port_a.update(GaussianWeightedMeanInfoMessage)
        self.equality_node.ports[0].update(GaussianMeanCovMessage)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_a.update(GaussianMeanCovMessage)

        return self.add_process_noise_node.port_b.update(GaussianMeanCovMessage)

    def get_ports(self):
        return [self.port_state_in, self.port_state_out, self.port_meas, self.port_input, self.port_process_noise]


class UnscentedKalmanSlice(Node):
    def __init__(self, func, B, C, method=1, alpha=0.9, name='UnscentedKalmanSlice'):
        super().__init__(name=name)

        # Initialize ports of the macro (slice) node
        self.port_state_in = NodePort(self, self.calc_msg_state_in, NodePortType.InPort, name="port_state_in")
        self.port_state_out = NodePort(self, self.calc_msg_state_out, NodePortType.OutPort, name="port_state_out")
        self.port_meas = NodePort(self, self.calc_msg_meas, NodePortType.OutPort, name="port_meas")
        self.port_input = NodePort(self, self.calc_msg_input, NodePortType.InPort, name="port_input")
        self.port_process_noise = NodePort(self, self.calc_msg_process_noise, NodePortType.InPort,
                                           name="port_process_noise")

        # Initialize all relevant nodes
        self.A_node = UnscentedNode(func)
        self.add_process_noise_node = AdditionNode()
        self.process_noise_in_node = PriorNode(GaussianMeanCovMessage([[0]], [[np.inf]]))
        self.add_input_node = AdditionNode()
        self.input_in_node = PriorNode(GaussianMeanCovMessage([[0]], [[np.inf]]))
        self.B_node = MatrixNode(B)
        self.compound_eq_mat_node = CompoundEqualityMatrixNode(C)
        self.meas_in_node = PriorNode(GaussianMeanCovMessage([[0]], [[np.inf]]))
        self.state_in_node = PriorNode(GaussianMeanCovMessage([[0]], [[np.inf]]))
        self.state_out_node = PriorNode(GaussianMeanCovMessage([[0]], [[np.inf]]))

        # Connect the nodes
        self.state_in_node.port_a.connect(self.A_node.port_a)
        self.A_node.port_b.connect(self.add_process_noise_node.port_a)
        self.process_noise_in_node.port_a.connect(self.add_process_noise_node.port_b)
        self.add_process_noise_node.port_c.connect(self.add_input_node.port_a)
        self.input_in_node.port_a.connect(self.B_node.port_a)
        self.B_node.port_b.connect(self.add_input_node.port_b)
        self.add_input_node.port_c.connect(self.compound_eq_mat_node.port_a)
        self.compound_eq_mat_node.port_b.connect(self.meas_in_node.port_a)
        self.compound_eq_mat_node.port_c.connect(self.state_out_node.port_a)

    def calc_msg_state_out(self):
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.A_node.port_b.update(GaussianMeanCovMessage)
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.add_process_noise_node.port_c.update(GaussianMeanCovMessage)
        self.input_in_node.update_prior(self.port_input.in_msg)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.add_input_node.port_c.update(GaussianMeanCovMessage)

        self.meas_in_node.update_prior(self.port_meas.in_msg, GaussianMeanCovMessage)

        return self.compound_eq_mat_node.port_c.update(GaussianMeanCovMessage)

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

    def get_ports(self):
        return [self.port_state_in, self.port_state_out, self.port_meas, self.port_input, self.port_process_noise]


class UnscentedKalmanFilter(object):
    def __init__(self, A_fun, B, C, initial_state_msg, input_noise_cov=None, process_noise_cov=None,
                 meas_noise_cov=None, slice_type=UnscentedKalmanSliceNaive, method=1, alpha=0.9):
        """
        Initialize a Kalman filter object given an initial state.

        :param A_fun: System matrix to be used as the default in all time slices.
        :param B: Input matrix to be used as the default in all time slices.
        :param C: Output matrix to be used as the default in all time slices.
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
        self.B = B
        self.C = C
        self.slice_type = slice_type
        self.input_noise_cov = input_noise_cov
        self.process_noise_cov = process_noise_cov
        self.meas_noise_cov = meas_noise_cov
        self.alpha = alpha
        self.method = method

        # Initialize factor graph
        self.slices = []
        self.init_state_node = PriorNode(initial_state_msg, name="init_state_node")
        n = len(initial_state_msg.mean)
        final_wm = np.zeros((n, 1))
        final_info = np.zeros((n, n))
        self.final_state_node = PriorNode(GaussianWeightedMeanInfoMessage(final_wm, final_info),
                                          name="final_state_node")

    def add_slice(self, input_val, meas_val, A=None, B=None, C=None, input_noise_msg=None, process_noise_msg=None,
                  meas_noise_msg=None):
        """
        Add a new time slice to the Kalman filter and connect it to the previous slices. Does *not* perform any
        filtering or smoothing.

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
        :return: Nothing.
        """
        if A is None:
            A = self.A_fun
        if B is None:
            B = self.B
        if C is None:
            C = self.C
        if input_noise_msg is None:
            input_noise_msg = GaussianMeanCovMessage([[0]], self.input_noise_cov)
        if process_noise_msg is None:
            process_noise_msg = GaussianMeanCovMessage([[0]], self.process_noise_cov)
        if meas_noise_msg is None:
            meas_noise_msg = GaussianMeanCovMessage([[0]], self.meas_noise_cov)

        new_slice = self.slice_type(A, B, C, method=self.method, alpha=self.alpha)

        new_slice.port_meas.connect(PriorNode(GaussianMeanCovMessage(meas_val, [[0]]) + meas_noise_msg).port_a)
        new_slice.port_input.connect(PriorNode(GaussianMeanCovMessage(input_val, [[0]]) + input_noise_msg).port_a)
        new_slice.port_process_noise.connect(PriorNode(process_noise_msg).port_a)

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
        for s in self.slices:
            s.port_state_out.update()

    def do_backward(self):
        for s in reversed(self.slices):
            s.port_state_in.update()

    def get_forward_state_msgs(self):
        return [s.port_state_out.out_msg for s in self.slices]

    def get_state_marginals(self):
        return [s.port_state_out.marginal(target_type=GaussianMeanCovMessage) for s in self.slices]

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
            As = [self.A_fun for _ in range(num_slices)]
        if Bs is None:
            Bs = [self.B for _ in range(num_slices)]
        if Cs is None:
            Cs = [self.C for _ in range(num_slices)]
        if input_noise_msgs is None:
            N_input_noise = self.input_noise_cov.shape[0]
            input_noise_msgs = [GaussianMeanCovMessage(
                np.zeros((N_input_noise, 1)), self.input_noise_cov) for _ in range(num_slices)]
        if process_noise_msgs is None:
            N_process_noise = self.process_noise_cov.shape[0]
            process_noise_msgs = [GaussianMeanCovMessage(
                np.zeros((N_process_noise, 1)), self.process_noise_cov) for _ in range(num_slices)]
        if meas_noise_msgs is None:
            N_meas_noise = self.meas_noise_cov.shape[0]
            meas_noise_msgs = [GaussianMeanCovMessage(
                np.zeros((N_meas_noise, 1)), self.meas_noise_cov) for _ in range(num_slices)]
        # Assert the same length of the input signals
        assert(len(input_vals) == len(meas_vals) == len(As) == len(Bs) == len(Cs) == len(input_noise_msgs)
               == len(process_noise_msgs) == len(meas_noise_msgs))

        for kk in range(num_slices):
            self.add_slice(input_vals[kk], meas_vals[kk], As[kk], Bs[kk], Cs[kk], input_noise_msgs[kk],
                           process_noise_msgs[kk], meas_noise_msgs[kk])

        return self

    def add_final_node(self, rts_start_msg):
        self.slices[-1].port_state_out.connect(PriorNode(rts_start_msg).port_a)

    def get_state_filtered_msgs(self):
        return [slice.port_state_out.out_msg for slice in self.slices]

    def get_state_smoothed_msgs(self):
        [slice.port_state_in.update() for slice in reversed(self.slices)]
        return [slice.port_state_out.marginal() for slice in self.slices]


def example_eike():
    # Define system model
    def A(x):
        return x + 1 / (1 + x * x) + 5 * np.cos(x)

    C = 1
    B = 1

    process_noise = [[0.5]]
    measurement_noise = [[10]]
    input_noise = [[1e-9]]

    # Number of time instants
    num_meas = 200

    np.random.seed(2)

    z_real = np.ones(num_meas)
    for i in range(1, num_meas):
        z_real[i] = A(z_real[i - 1]) + np.random.normal(0, process_noise)

    z = z_real + np.random.normal(0, measurement_noise, num_meas)
    u = np.zeros(num_meas)

    start = time.perf_counter()

    initial_state = GaussianMeanCovMessage(mean=[[10]], cov=[[1]])
    kf_slow = UnscentedKalmanFilter(A, B, C, initial_state,
                                    input_noise_cov=input_noise,
                                    process_noise_cov=process_noise,
                                    meas_noise_cov=measurement_noise,
                                    slice_type=UnscentedKalmanSlice,
                                    method=1,
                                    alpha=0.9)

    for ii in range(num_meas):
        kf_slow.add_slice(np.atleast_2d(u[ii]), np.atleast_2d(z[ii]))

    kf_slow.do_forward()
    kf_slow.do_backward()

    end = time.perf_counter()

    print('Time elapsed with 2*n+1 sigma points slice: ' + str(end - start))

    start = time.perf_counter()

    initial_state = GaussianMeanCovMessage(mean=[[10]], cov=[[1]])
    kf_fast = UnscentedKalmanFilter(A, B, C, initial_state,
                                    input_noise_cov=input_noise,
                                    process_noise_cov=process_noise,
                                    meas_noise_cov=measurement_noise,
                                    slice_type=UnscentedKalmanSlice,
                                    method=2,
                                    alpha=0.9)

    for ii in range(num_meas):
        kf_fast.add_slice(np.atleast_2d(u[ii]), np.atleast_2d(z[ii]))

    kf_fast.do_forward()
    kf_fast.do_backward()

    end = time.perf_counter()

    print('Time elapsed with n+1 sigma points slice: ' + str(end - start))

    z_est_slow_filt = np.squeeze([estimate.mean for estimate in kf_slow.get_forward_state_msgs()])
    z_est_slow_smoothed = np.squeeze([estimate.mean for estimate in kf_slow.get_state_marginals()])
    z_est_fast_filt = np.squeeze([estimate.mean for estimate in kf_fast.get_forward_state_msgs()])
    z_est_fast_smoothed = np.squeeze([estimate.mean for estimate in kf_fast.get_state_marginals()])

    # num_iterations = 50
    # mean_abs_err = np.zeros((2 + 2 * num_iterations, 1))
    # mean_abs_err[0] = abs(z_est_slow_filt - z_real).mean()
    # mean_abs_err[1] = abs(z_est_slow_smoothed - z_real).mean()
    # for ii in range(0, num_iterations):
    #     kf_slow.do_forward()
    #     states = np.squeeze([estimate.mean for estimate in kf_slow.get_state_marginals()])
    #     mean_abs_err[2*ii+2] = abs(states - z_real).mean()
    #     kf_slow.do_backward()
    #     states = np.squeeze([estimate.mean for estimate in kf_slow.get_state_marginals()])
    #     mean_abs_err[2*ii+3] = abs(states - z_real).mean()

    print('Mean abs error of fast filter')
    print(abs(z_est_fast_filt - z_real).mean())
    print('Mean abs error of fast smoother')
    print(abs(z_est_fast_smoothed - z_real).mean())
    print('Mean abs error of slow filter')
    print(abs(z_est_slow_filt - z_real).mean())
    print('Mean abs error of slow smoother')
    print(abs(z_est_slow_smoothed - z_real).mean())
#    print('Mean abs error of iterated slow smoother, final iteration')
#    print(mean_abs_err[-1])

    # 2n+1 sigma points
    plt.figure()
    plt.plot(z, 'k+', label='noisy measurements')
    plt.plot(z_est_slow_filt, 'b-',
             label='a posteriori estimate')
    plt.plot(z_est_slow_smoothed, 'r-',
             label='a posteriori estimate (s)')
#    plt.plot(states, 'k-',
#             label='a posteriori estimate, final iteration')
    plt.plot(z_real, color='g', label='truth value')
    plt.legend()
    plt.title('Estimates, 2*n+1 sigma points', fontweight='bold')
    plt.xlabel('Time step')
    plt.ylabel('')

#    plt.figure()
#    plt.plot(mean_abs_err)
#    plt.title('Mean absolute error vs. iteration step, 2*n+1 SPs')
#    plt.xlabel('Iteration')
#    plt.ylabel('Mean abs estimation error')
#    plt.grid()

    plt.figure()
    plt.plot(np.squeeze([estimate.cov for estimate in kf_slow.get_forward_state_msgs()]), label='error estimate')
    plt.plot(np.squeeze([estimate.cov for estimate in kf_slow.get_state_marginals()]), label='error estimate (s)')
    plt.legend()
    plt.title('Estimated error covariance, 2*n+1 sigma points', fontweight='bold')
    plt.xlabel('Time step')
    plt.ylabel('')

    # n+1 sigma points
    plt.figure()
    plt.plot(z, 'k+', label='noisy measurements')
    plt.plot(z_est_fast_filt, 'b-',
             label='a posteriori estimate')
    plt.plot(z_est_fast_smoothed, 'r-',
             label='a posteriori estimate (s)')
    plt.plot(z_real, color='g', label='truth value')
    plt.legend()
    plt.title('Estimates, n+1 sigma points', fontweight='bold')
    plt.xlabel('Time step')
    plt.ylabel('')

    plt.figure()
    plt.plot(np.squeeze([estimate.cov for estimate in kf_fast.get_forward_state_msgs()]), label='error estimate')
    plt.plot(np.squeeze([estimate.cov for estimate in kf_fast.get_state_marginals()]), label='error estimate (s)')
    plt.title('Estimated error covariance, n+1 sigma points', fontweight='bold')
    plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('')

    plt.show()


def example_maximilian():
    method_id = 1  # 0 is reduced, 1 is classic, 2 is quasi_random_riemann_sums, 3 is Gauss Hermite
    name = 'filtered'

    # Define system model
    start_value = -0.1
    process_error_var = 1e-1
    meas_error_var = 1e-1

    def A(x):
        return np.array([x[1] * x[0], x[1]])
        # return array([exp(x[1]) * x[0], x[1]])

    C = np.identity(2)
    B = np.identity(2)

    # Number of time instants
    num_meas = 100

    np.random.seed(1)

    z_real_mean = np.array([np.linspace(start_value, start_value, num_meas),
                            np.linspace(start_value, start_value, num_meas)])

    for i in range(0, num_meas):
        if i == 0:
            z_real_mean[..., i] = z_real_mean[..., i]
        else:
            z_real_mean[..., i] = A(z_real_mean[..., i - 1]) + \
                np.array([rnd.normal(0, process_error_var), rnd.normal(0, process_error_var)])

    z = z_real_mean + np.array([rnd.normal(0, meas_error_var, num_meas), rnd.normal(0, meas_error_var, num_meas)])

    u = np.array([np.linspace(0, 0, num_meas), np.linspace(0, 0, num_meas)])

    # create Kalman Filter factorgraph
    initial_state = GaussianMeanCovMessage(mean=np.array([[start_value], [start_value]]), cov=np.identity(2))
    kf = UnscentedKalmanFilter(A, B, C, initial_state,
                               input_noise_cov=np.zeros((2, 2)),
                               process_noise_cov=np.array([[process_error_var, 0], [0, process_error_var]]),
                               meas_noise_cov=np.array([[meas_error_var, 0], [0, meas_error_var]]),
                               slice_type=UnscentedKalmanSlice,
                               method=method_id,
                               alpha=0.1)

    for i in range(num_meas):
        kf.add_slice(u[..., i], z[..., i])

    kf.add_final_node(GaussianWeightedMeanInfoMessage(np.zeros((2, 1)), np.zeros((2, 2))))

    def get_msgs():
        if name == 'smoothed':
            return kf.get_state_smoothed_msgs()
        elif name == 'filtered':
            return kf.get_state_filtered_msgs()
        else:
            raise TypeError('unknown display case')

    if method_id == 1:
        method_name = 'classic'
    elif method_id == 0:
        method_name = 'reduced'
    elif method_id == 2:
        method_name = 'quasi_random_riemann_sums'
    elif method_id == 3:
        method_name = 'gauss_hermite'
    else:
        method_name = 'unknown_method'

    estimates = np.squeeze([estimate.convert(GaussianMeanCovMessage).mean for estimate in get_msgs()])
    variances = np.squeeze([estimate.convert(GaussianMeanCovMessage).cov for estimate in get_msgs()])

    # print sum of errors
    sum_of_errors = sum(np.sqrt(sum(np.square(estimates.T - z_real_mean), axis=0)))

    time_axis = np.linspace(1, num_meas, num_meas)

    # plot x_1
    fig = plt.figure()
    plt.plot(z[0, ...], 'k+', label='Messungen $Y_t$')
    plt.plot(estimates[..., 0], 'b',
             label='geschätzter Erwartungswert')
    plt.plot(z_real_mean[0, ...], 'g', label='tatsächlicher Zustand')
    plt.legend()
    plt.xlabel('Zeitpunkt t')
    plt.ylabel('Wert von $x_{%d}$' % int(0 + 1))
    # plt.figtext(0.4,0.15,'Summe der Fehlerquadrate: ' + str(sq_error))
    plt.savefig(method_name + name + '_mean%d.pdf' % 1, format='pdf')

    # plot x_2
    fig = plt.figure()
    plt.plot(z[1, ...], 'k+', label='Messungen $Y_t$')
    plt.plot(estimates[..., 1], 'b',
             label='geschätzter Erwartungswert')
    plt.plot(z_real_mean[1, ...], 'g', label='tatsächlicher Zustand')
    plt.legend()
    plt.xlabel('Zeitpunkt t')
    plt.ylabel('Wert von $x_{%d}$' % int(1 + 1))
    # plt.figtext(0.4,0.15,'Summe der Fehlerquadrate: ' + str(sq_error))
    plt.savefig(method_name + name + '_mean%d.pdf' % 2, format='pdf')

    # plot errors
    fig = plt.figure()
    plt.plot(np.sqrt(sum(np.square(estimates.T - z_real_mean), axis=0)), 'k+',
             label='euklidische Norm der Differenz von \n geschätztem Erwartungswert und tatsächlichem Zustand')
    plt.legend()
    plt.xlabel('Zeitpunkt t')
    plt.ylabel('Wert des Abstands')
    plt.figtext(0.4, 0.7, 'Summe der Abstände: ' + str(sum_of_errors))
    plt.savefig(method_name + name + '_distance.pdf', format='pdf')

    # plot variances
    fig = plt.figure()
    plt.plot(variances[2::][..., 0][..., 0], 'b',
             label='Varianz von $x_1$')
    plt.plot(variances[2::][..., 1][..., 1], 'g',
             label='Varianz von $x_2$')
    plt.plot(variances[1::][..., 0][..., 1], 'r',
             label='Kovarianz von $x_1$ und $x_2$ ')
    plt.legend()
    plt.xlabel('Zeitpunkt t')
    plt.ylabel('Kovarianz')
    plt.savefig(method_name + name + '_cov.pdf', format='pdf')

    plt.show()


if __name__ == '__main__':
    example_eike()
    # example_maximilian()
