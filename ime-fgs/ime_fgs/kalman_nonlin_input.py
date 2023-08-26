# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#

from numpy import inf, atleast_2d, ndarray, zeros, concatenate, identity
from numpy.linalg import matrix_power, matrix_rank
from enum import Enum
from inspect import signature
import time

from ime_fgs.base import NodePort, Node, NodePortType
from ime_fgs.basic_nodes import MatrixNode, AdditionNode, PriorNode, EqualityNode
from ime_fgs.advanced_nodes import Jacobian_or_Statistical_LinearizationNode
from ime_fgs.em_node import NUVPrior
from ime_fgs.compound_nodes import CompoundEqualityMatrixNode
from ime_fgs.messages import GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage, GaussianTildeMessage, \
    GaussianNonInformativeMessage, PortMessageDirection
from ime_fgs.unscented_utils import SigmaPointScheme


class KalmanSlice(Node):
    class BackwardPassMode(Enum):
        MeanCov = 1
        Tilde = 2

    def __init__(self, A, B, C, backward_pass_mode=None, unscented_backward_pass_mode=None, sigma_point_scheme=None,
                 alpha=None, linearization_about_marginal=False, expectation_propagation=False, dAdx=None,
                 jacobian_linearization=False):
        super().__init__('KalmanSlice')
        assert A is not None
        assert B is not None
        assert C is not None

        if backward_pass_mode is None:
            backward_pass_mode = KalmanSlice.BackwardPassMode.Tilde
        self.backward_pass_mode = backward_pass_mode

        # NonlinearNode_type = StatisticalLinearizationNode
        NonlinearNode_type = Jacobian_or_Statistical_LinearizationNode
        # NonlinearNode_type = UnscentedNode

        if unscented_backward_pass_mode is None:
            if self.backward_pass_mode == KalmanSlice.BackwardPassMode.MeanCov:
                unscented_backward_pass_mode = NonlinearNode_type.BackwardPassMode.MeanCov
            elif self.backward_pass_mode == KalmanSlice.BackwardPassMode.Tilde:
                unscented_backward_pass_mode = NonlinearNode_type.BackwardPassMode.Tilde
            else:
                raise NotImplementedError

        # Initialize ports of the macro (slice) node
        self.port_state_in = NodePort(self, self.calc_msg_state_in, NodePortType.InPort)
        self.port_state_out = NodePort(self, self.calc_msg_state_out, NodePortType.OutPort)
        self.port_meas = NodePort(self, self.calc_msg_meas, NodePortType.OutPort)
        self.port_input = NodePort(self, self.calc_msg_input, NodePortType.InPort)
        self.port_process_noise = NodePort(self, self.calc_msg_process_noise, NodePortType.InPort)

        # Initialize all relevant nodes
        if callable(A):
            Asig = signature(A)
            A_number_of_in_ports = len(Asig.parameters)
            self.A_node = NonlinearNode_type(function=A, deriv_func=dAdx, number_of_in_ports=A_number_of_in_ports,
                                             backward_pass_mode=unscented_backward_pass_mode,
                                             sigma_point_scheme=sigma_point_scheme, alpha=alpha,
                                             linearization_about_marginal=linearization_about_marginal,
                                             expectation_propagation=expectation_propagation,
                                             jacobian_linearization=jacobian_linearization)
        else:
            self.A_node = MatrixNode(A)

        # There should be at least two input ports
        assert (self.A_node.number_of_in_ports == 2)

        self.add_process_noise_node = AdditionNode()
        self.process_noise_in_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))

        self.input_in_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))
        self.B_node = MatrixNode(B)
        self.compound_eq_mat_node = CompoundEqualityMatrixNode(C)
        self.meas_in_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))
        self.state_in_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))
        self.state_out_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))

        # Connect the nodes
        self.state_in_node.port_a.connect(self.A_node.ports_in[0])
        self.A_node.ports_out[0].connect(self.add_process_noise_node.port_a)
        self.process_noise_in_node.port_a.connect(self.add_process_noise_node.port_b)
        self.add_process_noise_node.port_c.connect(self.compound_eq_mat_node.port_a)
        self.input_in_node.port_a.connect(self.B_node.port_a)
        self.B_node.port_b.connect(self.A_node.ports_in[1])
        self.compound_eq_mat_node.port_b.connect(self.meas_in_node.port_a)
        self.compound_eq_mat_node.port_c.connect(self.state_out_node.port_a)

    def calc_msg_state_out(self):
        start = time.perf_counter()

        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.input_in_node.update_prior(self.port_input.in_msg)
        self.B_node.port_b.update(GaussianMeanCovMessage)
        self.A_node.ports_out[0].update(GaussianMeanCovMessage)
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.add_process_noise_node.port_c.update(GaussianMeanCovMessage)
        self.meas_in_node.update_prior(self.port_meas.in_msg, GaussianMeanCovMessage)

        # end = time.perf_counter()
        # total_time = end - start
        #
        # print('Time for forward step', str(total_time))
        return self.compound_eq_mat_node.port_c.update(GaussianMeanCovMessage)

    def calc_msg_state_in(self):
        self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
        self.input_in_node.update_prior(self.port_input.in_msg)

        if self.backward_pass_mode == KalmanSlice.BackwardPassMode.MeanCov:
            self.state_out_node.update_prior(self.port_state_out.in_msg.convert(GaussianWeightedMeanInfoMessage))
            self.meas_in_node.update_prior(self.port_meas.in_msg.convert(GaussianWeightedMeanInfoMessage))
            self.compound_eq_mat_node.port_a.update(GaussianMeanCovMessage)
            self.B_node.port_b.update()
            self.add_process_noise_node.port_a.update(GaussianWeightedMeanInfoMessage)
            return self.A_node.ports_in[0].update()
        elif self.backward_pass_mode == KalmanSlice.BackwardPassMode.Tilde:
            self.state_out_node.update_prior(self.port_state_out.in_msg)
            self.meas_in_node.update_prior(self.port_meas.in_msg)
            # If the KS has run in MeanCov before, the target_type of this node is currently set to GaussianMeanCovMsg
            # Simply setting by .update(target_type=None) doesn't work, since None is the default param that's
            # interpreted as 'do not change the current settings'.
            self.compound_eq_mat_node.port_a.target_type = None
            self.compound_eq_mat_node.port_a.update()
            self.B_node.port_b.update()
            self.add_process_noise_node.port_a.target_type = None  # see above
            self.add_process_noise_node.port_a.update()
            self.A_node.ports_in[1].update()
            return self.A_node.ports_in[0].update()
        else:
            raise NotImplementedError('Unknown backward pass type provided.')

    def calc_msg_meas(self):
        # TBD
        raise NotImplementedError

    def calc_msg_input(self):
        if self.backward_pass_mode == KalmanSlice.BackwardPassMode.Tilde:
            # self.process_noise_in_node.update_prior(self.port_process_noise.in_msg)
            # self.state_in_node.update_prior(self.port_state_in.in_msg)
            # self.state_out_node.update_prior(self.port_state_out.in_msg)
            # self.meas_in_node.update_prior(self.port_meas.in_msg)
            # self.input_in_node.update_prior(self.port_input.in_msg)
            #
            # self.add_process_noise_node.port_a.update()
            # self.compound_eq_mat_node.port_a.update()
            # self.A_node.ports_in[1].update()

            return self.B_node.port_a.update()
        else:
            # Could probably also be implemented in MeanCov mode but wouldn't work for unobservable systems due to
            # required inversions of singular matrices.
            raise NotImplementedError

    def calc_msg_process_noise(self):
        # TBD
        raise NotImplementedError

    def get_ports(self):
        return [self.port_state_in, self.port_state_out, self.port_meas, self.port_input, self.port_process_noise]


class KalmanFilter(object):

    def __init__(self, A=None, B=None, C=None, initial_state_msg=None, input_noise_cov=None, process_noise_cov=None,
                 meas_noise_cov=None, slice_type=KalmanSlice,
                 backward_pass_mode=KalmanSlice.BackwardPassMode.Tilde,
                 sigma_point_scheme=None,
                 regularized_input_estimation=False,
                 linearization_about_marginal=False,
                 expectation_propagation=False,
                 jacobian_linearization=False,
                 dAdx=None,
                 **kwargs):
        """
        Initialize a Kalman filter object given an initial state.

        :param A: System matrix to be used as the default in all time slices.
        :param B: Input matrix to be used as the default in all time slices.
        :param C: Output matrix to be used as the default in all time slices.
        :param initial_state_msg: Message specifying information about the initial system state.
        :param input_noise_cov: Input noise covariance to be used as the default in all time slices. If not provided,
          input noise messages must be provided in each time slice.
        :param process_noise_cov: Process noise covariance to be used as the default in all time slices. If not
          provided, process noise messages must be provided in each time slice.
        :param meas_noise_cov: Measurement noise covariance to be used as the default in all time slices. If not
          provided, measurement noise messages must be provided in each time slice.
        :param backward_pass_mode: Message passing schedule to be used for the backward (smoothing) pass. For available
          options, see KalmanSlice.BackwardPassMode.
        :param slice_type: The Kalman filter time slice model to be used.
        :param **kwargs: Passed on to the slice_type constructor.
        """
        # Store default parameter values
        if A is not None and not callable(A):
            self.A = atleast_2d(A)
        else:
            self.A = A
        if dAdx is not None and not callable(dAdx):
            self.dAdx = atleast_2d(dAdx)
        else:
            self.dAdx = dAdx
        if B is not None:
            self.B = atleast_2d(B)
        if C is not None:
            self.C = atleast_2d(C)
        self._initial_state_set = False
        if initial_state_msg is not None:
            self.initial_state = PriorNode(initial_state_msg, name="initial_state_node")
            self._initial_state_set = True
        if input_noise_cov is not None:
            self.input_noise_cov = atleast_2d(input_noise_cov)
        if process_noise_cov is not None:
            self.process_noise_cov = atleast_2d(process_noise_cov)
        if meas_noise_cov is not None:
            self.meas_noise_cov = atleast_2d(meas_noise_cov)

        self.slice_type = slice_type
        self.backward_pass_mode = backward_pass_mode
        self.slice_type_kwargs = kwargs

        # Initialize factor graph
        self.slices = []
        self.final_state_node = None

        self.regularized_input_estimation = regularized_input_estimation
        self.linearization_about_marginal = linearization_about_marginal
        self.sigma_point_scheme = sigma_point_scheme
        self.expectation_propagation = expectation_propagation
        self.jacobian_linearization = jacobian_linearization

        # ToDo: Eliminate nasty hard coded stuff
        if self.regularized_input_estimation is True:
            self.NUVPriors = []

    def set_model(self, A, B, C, initial_state_msg, input_noise_cov, process_noise_cov, meas_noise_cov, dAdx=None):
        if callable(A):
            self.A = A
        else:
            self.A = atleast_2d(A)
        if callable(dAdx):
            self.dAdx = dAdx
        else:
            self.dAdx = atleast_2d(dAdx)
        self.B = atleast_2d(B)
        self.C = atleast_2d(C)
        self.initial_state = PriorNode(initial_state_msg, name="initial_state_node")
        self._initial_state_set = True
        self.input_noise_cov = atleast_2d(input_noise_cov)
        self.meas_noise_cov = atleast_2d(meas_noise_cov)
        self.process_noise_cov = atleast_2d(process_noise_cov)

    def observability_matrix(self):
        if callable(self.A):
            raise AttributeError('Cannot compute observability matrix since state transition is a function')
        n_states = self.A.shape[0]
        observability_matrix_rows = [self.C]
        for i in range(1, n_states):
            observability_matrix_rows.append(self.C @ matrix_power(self.A, i))
        obs_matrix = concatenate(observability_matrix_rows, axis=0)
        assert obs_matrix.shape == (self.C.shape[0] * n_states, n_states)
        return obs_matrix

    def is_observable(self):
        if callable(self.A):
            return None
        else:
            return matrix_rank(self.observability_matrix()) == self.A.shape[0]

    def add_slice(self, meas_vals, input_vals_or_port=None, A=None, B=None, C=None, input_noise_msg=None,
                  process_noise_msg=None,
                  meas_noise_msg=None, dAdx=None):
        """
        Add a new time slice to the Kalman filter and connect it to the previous slices. Does *not* perform any
        filtering or smoothing.

        :param meas_vals: Measurement values in this time slice. Note that measurement noise is already taken into
          account in the system model. Time-varying measurement noise covariances can be passed separately.
        :param input_vals_or_port: Scalar of array representing nput values in this time slice. Note that input noise is
          already taken into account in the system model. Time-varying input noise covariances can be passed separately.
          Alternatively, a NodePort object can also be passed, to which external logic can be connected.
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
        :return: the updated KF object..
        """
        assert self._initial_state_set

        if A is None:
            A = self.A
        # if dAdx is None:
        self.dAdx = dAdx
        if B is None:
            B = self.B
        if C is None:
            C = self.C
        if input_noise_msg is None:
            input_noise_msg = GaussianMeanCovMessage(0, self.input_noise_cov)
        if process_noise_msg is None:
            process_noise_msg = GaussianMeanCovMessage(0, self.process_noise_cov)
        if meas_noise_msg is None:
            meas_noise_msg = GaussianMeanCovMessage(0, self.meas_noise_cov)

        if self.slice_type == KalmanSlice:
            new_slice = self.slice_type(A, B, C, backward_pass_mode=self.backward_pass_mode,
                                        linearization_about_marginal=self.linearization_about_marginal,
                                        sigma_point_scheme=self.sigma_point_scheme,
                                        expectation_propagation=self.expectation_propagation,
                                        jacobian_linearization=self.jacobian_linearization,
                                        dAdx=self.dAdx,
                                        **self.slice_type_kwargs)
        else:
            new_slice = self.slice_type(A, B, C, **self.slice_type_kwargs)

        if isinstance(meas_vals, ndarray):
            dim_meas = len(meas_vals)
        else:
            dim_meas = 1
        meas_cov = zeros((dim_meas, dim_meas))
        new_slice.port_meas.connect(PriorNode(GaussianMeanCovMessage(meas_vals, meas_cov) + meas_noise_msg).port_a)

        if input_vals_or_port is None:
            if self.regularized_input_estimation is True:
                self.NUVPriors.append(NUVPrior(name="node_NUV", A_0=identity(self.B.shape[1]),
                                               mean_offset=zeros([self.B.shape[1], 1]),
                                               non_informative_prior=True))
                input_vals_or_port = self.NUVPriors[-0].port_b
            else:
                # Usual (unregularized) input estimation
                N_inputs = B.shape[1]
                input_vals_or_port = zeros((N_inputs, 1))

        if isinstance(input_vals_or_port, ndarray):
            dim_inputs = len(input_vals_or_port)
            input_cov = zeros((dim_inputs, dim_inputs))
            new_slice.port_input.connect(PriorNode(
                GaussianMeanCovMessage(input_vals_or_port, input_cov) + input_noise_msg).port_a)
        elif isinstance(input_vals_or_port, NodePort):
            new_slice.port_input.connect(input_vals_or_port)
        else:
            input_cov = 0
            new_slice.port_input.connect(PriorNode(
                GaussianMeanCovMessage(input_vals_or_port, input_cov) + input_noise_msg).port_a)

        new_slice.port_process_noise.connect(PriorNode(process_noise_msg).port_a)

        if len(self.slices) == 0:
            self.initial_state.port_a.connect(new_slice.port_state_in)
        else:
            if self.slices[-1].port_state_out.connected:
                self.slices[-1].port_state_out.disconnect()
            self.slices[-1].port_state_out.connect(new_slice.port_state_in)

        if self.final_state_node is not None:
            # connect the 'terminator' node to this slice since it is the last one
            new_slice.port_state_out.connect(self.final_state_node.port_a)
            self.final_state_node.port_a.update()

        self.slices.append(new_slice)
        return self

    def add_slices(self, meas_vals, input_vals_or_ports=None, As=None, Bs=None, Cs=None, input_noise_msgs=None,
                   process_noise_msgs=None, meas_noise_msgs=None, dAdxs=None):
        """
        Adds a sequence of new input and output measurements to the Kalman filter factor graph and updates the current
        state estimate, incorporating the new measurements.

        :param meas_vals: List or numpy array of measured values in each time slice. Note that measurement noise is
          already taken into account in the system model. Time-varying measurement noise covariances can be passed
          separately. If a numpy array, the array is expected to be of shape MxN, where M is the number of measurement
          signals and N the number of slices.
        :param input_vals_or_ports: List or numpy array of input values in each time slice, or list of NodePort objects
          to which each slice's input will be connected. Note that input noise is already taken into account in the
          system model. Time-varying input noise covariances can be passed separately. If a numpy array, the array is
          expected to be of shape MxN, where M is the number of input signals and N the number of slices.
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
        input_ports_passed = False

        assert meas_vals.shape[0] == self.C.shape[0]
        num_slices = meas_vals.shape[1]

        if input_vals_or_ports is None:

            if self.regularized_input_estimation is True:
                self.NUVPriors = []
                for ii in range(0, num_slices):
                    self.NUVPriors.append(NUVPrior(name="node_NUV", A_0=identity(self.B.shape[1]),
                                                   mean_offset=zeros([self.B.shape[1], 1]), non_informative_prior=True))

                input_vals_or_ports = [current_NUVPrior.port_b for current_NUVPrior in self.NUVPriors]
            else:
                # Usual (unregularized) input estimation
                N_inputs = self.B.shape[1]
                input_vals_or_ports = zeros((N_inputs, num_slices))

        # Input handling: if a list, transform to an array
        if isinstance(input_vals_or_ports, list):
            if isinstance(input_vals_or_ports[0], NodePort):
                input_ports_passed = True
            else:
                input_vals_or_ports = concatenate(input_vals_or_ports, axis=1)
        if not input_ports_passed:
            assert isinstance(input_vals_or_ports, ndarray)
            assert input_vals_or_ports.shape[0] == self.B.shape[1]
        if isinstance(meas_vals, list):
            meas_vals = concatenate(meas_vals, axis=1)

        if As is None:
            # TODO: Implement this in a more memory efficient way using a generator function?
            As = [self.A for _ in range(num_slices)]
        if dAdxs is None:
            # TODO: Implement this in a more memory efficient way using a generator function?
            dAdxs = [self.dAdx for _ in range(num_slices)]
        if Bs is None:
            Bs = [self.B for _ in range(num_slices)]
        if Cs is None:
            Cs = [self.C for _ in range(num_slices)]
        if input_noise_msgs is None:
            N_input_noise = self.input_noise_cov.shape[0]
            input_noise_msgs = [GaussianMeanCovMessage(zeros((N_input_noise, 1)), self.input_noise_cov) for _ in
                                range(num_slices)]
        if process_noise_msgs is None:
            N_process_noise = self.process_noise_cov.shape[0]
            process_noise_msgs = [GaussianMeanCovMessage(zeros((N_process_noise, 1)), self.process_noise_cov) for _
                                  in range(num_slices)]
        if meas_noise_msgs is None:
            N_meas_noise = self.meas_noise_cov.shape[0]
            meas_noise_msgs = [GaussianMeanCovMessage(zeros((N_meas_noise, 1)), self.meas_noise_cov) for _ in
                               range(num_slices)]
        # Assert the same length of the input signals
        assert (num_slices == meas_vals.shape[1] == len(As) == len(Bs) == len(Cs) == len(input_noise_msgs)
                == len(process_noise_msgs) == len(meas_noise_msgs))

        for kk in range(num_slices):
            self.add_slice(meas_vals[:, [kk]],
                           input_vals_or_ports[kk] if input_ports_passed else input_vals_or_ports[:, [kk]],
                           As[kk], Bs[kk], Cs[kk], input_noise_msgs[kk],
                           process_noise_msgs[kk], meas_noise_msgs[kk], dAdx=dAdxs[kk])

        self._add_final_node()
        return self

    def do_forward(self):
        for s in self.slices:
            s.port_state_out.update()
        return self

    def do_backward(self):
        if self.is_observable() is not None and not self.is_observable() \
                and self.backward_pass_mode == KalmanSlice.BackwardPassMode.MeanCov:
            # Requires inversion of singular matrix
            raise RuntimeError('Cannot perform backward MeanCov smoothing for unobservable systems.')
        for s in reversed(self.slices):
            s.port_state_in.update()
        return self

    def do_input_estimation(self, num_iter=20):
        if self.regularized_input_estimation:
            for ii in range(num_iter):
                self.do_forward()
                self.do_backward()
                for s in self.slices:
                    s.port_input.update()
                self.do_update_input_variance_estimates()
        else:
            for s in self.slices:
                s.port_input.update()

    def do_update_input_variance_estimates(self):
        if self.regularized_input_estimation is True:
            for s in self.NUVPriors:
                s.port_theta.update()
                s.port_b.update()
        else:
            raise NotImplementedError('It does not make sense to ask for regularized input estimation if '
                                      'do_NUV_input_estimation_regularization is False!')

    def get_input_marginals(self):
        return [s.port_input.marginal(target_type=GaussianMeanCovMessage) for s in self.slices]

    def get_state_fwd_messages(self):
        return [self.slices[0].port_state_in.in_msg] + \
               [s.port_state_out.out_msg for s in self.slices]

    def get_state_marginals(self):
        return [self.slices[0].port_state_in.marginal(target_type=GaussianMeanCovMessage)] + \
               [s.port_state_out.marginal(target_type=GaussianMeanCovMessage) for s in self.slices]

    def _add_final_node(self):
        n = len(self.initial_state.port_a.out_msg.mean)
        self.final_state_node = PriorNode(GaussianNonInformativeMessage(n, direction=PortMessageDirection.Backward,
                                                                        inf_approx=None))

        if self.slices[-1].port_state_out.connected:
            self.slices[-1].port_state_out.disconnect()
        self.slices[-1].port_state_out.connect(self.final_state_node.port_a)
        return self


if __name__ == '__main__':
    import ime_fgs.demos.kalman.welsh as welsh
    import ime_fgs.demos.kalman.nonlin_1d as nonlin_1d

    welsh.run_example()
    nonlin_1d.run_example()
