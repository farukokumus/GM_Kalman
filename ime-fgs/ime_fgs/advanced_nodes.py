# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
from ime_fgs.base import NodePort
from numpy import atleast_2d, asarray, ndim, matrix, stack, shape, allclose, vstack, exp, sqrt, pi, squeeze
from scipy.linalg import block_diag
from numpy.linalg import inv, det
from enum import Enum

from ime_fgs.utils import col_vec
from ime_fgs.base import Node, NodePortType
from ime_fgs.basic_nodes import *
from ime_fgs.unscented_utils import backwards_unscented, SigmaPointScheme, unscented_transform_gaussian
from ime_fgs.messages import GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage, GaussianTildeMessage, \
    GaussianNonInformativeMessage, PortMessageDirection


class TruncationNode(Node):
    """
      a +--------+ b
        |   __   |
    --->|  |  |  |--->
        | _|  |_ |
        +--------+
    """

    def __init__(self, hyperplane_normal, lower_bounds, upper_bounds, name=None):
        super().__init__(name=name)

        self.port_a = NodePort(self, self._calc_msg_a, NodePortType.InPort)
        self.port_b = NodePort(self, self._calc_msg_b, NodePortType.OutPort)
        self._lower_bounds = atleast_2d(asarray(lower_bounds))
        # assert ndim(self._lower_bound) == 2 and self._lower_bound.shape == (1, 1)
        self._upper_bounds = atleast_2d(asarray(upper_bounds))
        # assert ndim(self._upper_bound) == 2 and self._upper_bound.shape == (1, 1)
        assert ndim(self._lower_bounds) == ndim(self._upper_bounds)
        self._hyperplane_normal = atleast_2d(asarray(hyperplane_normal))
        # assert ndim(self._hyperplane_normal) == 2 and self._hyperplane_normal.shape == (1, 1)

    @property
    def lower_bounds(self):
        return self._lower_bounds

    @property
    def upper_bounds(self):
        return self._upper_bounds

    @property
    def hyperplane_normal(self):
        return self._hyperplane_normal

    @lower_bounds.setter
    def lower_bounds(self, lower_bounds):
        self._lower_bounds = atleast_2d(asarray(lower_bounds))

    @upper_bounds.setter
    def upper_bounds(self, upper_bounds):
        self._upper_bounds = atleast_2d(asarray(upper_bounds))

    @hyperplane_normal.setter
    def hyperplane_normal(self, hyperplane_normal):
        self._hyperplane_normal = atleast_2d(asarray(hyperplane_normal))

    def _calc_msg_a(self):
        return self.port_b.in_msg.approximate_truncation_by_moment_matching(self._hyperplane_normal, self._upper_bounds,
                                                                            self._lower_bounds, inverse=True)

    def _calc_msg_b(self):
        return self.port_a.in_msg.approximate_truncation_by_moment_matching(self._hyperplane_normal, self._upper_bounds,
                                                                            self._lower_bounds)

    def __repr__(self):
        if self.name is None:
            name = str(id(self))
        else:
            name = self.name

        return type(self).__name__ + "(" + name + \
            ", Lower truncation bound:" + repr(self.lower_bounds.tolist()) + \
            ", Upper truncation bound:" + repr(self.upper_bounds.tolist()) + ")"

    def get_ports(self):
        return [self.port_a, self.port_b]


class UnscentedNode(Node):
    """
      a +----------+ b
    --->| function |--->
        +----------+
    """

    class BackwardPassMode(Enum):
        MeanCov = 1  # Naive implementation of Särkkä's unscented RTS paper (2008)
        WeightedMeanInfo = 2  # Implementation of algorithm from Maximilian Pilz' thesis (2017)
        Tilde = 3  # Implementation of algorithm from Freiburg-paper (Petersen, Hoffmann, Rostalski 2018)

    def __init__(self, function, name=None, inverse_func=None, sigma_point_scheme=None,
                 alpha=None, backward_pass_mode=None, forward_marginal_passing=False):
        """

        :param function:
        :param name: Node name
        :param inverse_func: Inverse of nonlinear function to pass through, if available. If available, normal unscented
        pass is performed backwards. Otherweise, approximative RTS pass is perforemd in the bw direction.
        :param sigma_point_scheme: Selects method to choose sigma points and weights for the UT. See UT functions for
        details.
        :param alpha: Parameter for sigma point calculation in the UT (see there for details).
        :param backward_pass_mode: Selects default Rauch-Tung-Striebel (RTS) type backwards pass implementation. See
        UnscentedNode.BackwardsPassMode for options. Can be overriden explicitly when calling self._calc_msg_a_.
        :param forward_marginal_passing: If False (default), UT is performed for directed messages in the fwd direction,
        and those are also returned by port_b.calc_msg(). If True, UT is performed for marginals instead and those are
        also returned *AS A DIRECTED MESSAGE* (yes, this may be unexpected and should be changed in the future).
        """
        super().__init__(name=name)

        self.port_a = NodePort(self, self._calc_msg_a, NodePortType.InPort)
        self.port_b = NodePort(self, self._calc_msg_b, NodePortType.OutPort)

        # Handles for (nonlinear) functions to pass through
        self.function = function
        self.inverse_function = inverse_func

        # Params for unscented transform
        self.alpha = alpha
        self.sigma_point_scheme = sigma_point_scheme

        # Backup of forward unscented pass
        self.cross_covariance = None

        # Backup of previous marginals for iterative smoothing
        self.b_margin_old = None

        # RTS backwards pass implementation to use
        if backward_pass_mode is None:
            backward_pass_mode = UnscentedNode.BackwardPassMode.MeanCov
        self.backward_pass_mode = backward_pass_mode
        self.backward_pass_method = self.get_backward_pass_method(backward_pass_mode)

        # Forward message passing type
        self.forward_marginal_passing = forward_marginal_passing
        if forward_marginal_passing:
            raise DeprecationWarning('The current implementation of forward marginal passing is old and deprecated.')

    def get_backward_pass_method(self, backward_pass_mode):
        assert isinstance(backward_pass_mode, UnscentedNode.BackwardPassMode)
        if backward_pass_mode == UnscentedNode.BackwardPassMode.MeanCov:
            backward_pass_method = self.rts_mean_cov
        elif backward_pass_mode == UnscentedNode.BackwardPassMode.WeightedMeanInfo:
            backward_pass_method = self.rts_weighted_mean_info
        elif backward_pass_mode == UnscentedNode.BackwardPassMode.Tilde:
            backward_pass_method = self.rts_tilde
        else:
            raise NotImplementedError
        return backward_pass_method

    def rts_mean_cov(self):
        # See Simo Särkkä, Unscented Rauch-Tung-Striebel Smoother.
        # Very naive implementation
        # a_margin_old = self.port_a.marginal(target_type=GaussianMeanCovMessage)
        # b_margin_new = self.port_b.marginal(target_type=GaussianMeanCovMessage)
        #
        # D_a = self.cross_covariance @ inv(self.b_margin_old.cov)
        # mean = a_margin_old.mean + D_a @ (b_margin_new.mean - self.b_margin_old.mean)
        # cov = a_margin_old.cov + D_a @ (b_margin_new.cov - self.b_margin_old.cov) @ D_a.T

        a_fwd_old = self.port_a.in_msg.convert(target_type=GaussianMeanCovMessage)
        b_fwd_old = self.port_b.out_msg.convert(target_type=GaussianMeanCovMessage)
        b_margin_new = self.port_b.marginal(target_type=GaussianMeanCovMessage)

        D_a = self.cross_covariance @ inv(b_fwd_old.cov)
        mean = a_fwd_old.mean + D_a @ (b_margin_new.mean - b_fwd_old.mean)
        cov = a_fwd_old.cov + D_a @ (b_margin_new.cov - b_fwd_old.cov) @ D_a.T

        a_margin = GaussianMeanCovMessage(mean, cov)
        a_out_messsage = a_margin / self.port_a.in_msg

        return a_out_messsage

    def rts_weighted_mean_info(self):
        weighted_mean, info = backwards_unscented(self.port_a.in_msg, self.port_b.out_msg, self.port_b.in_msg,
                                                  self.cross_covariance)
        return GaussianWeightedMeanInfoMessage(weighted_mean, info)

    def rts_tilde(self):
        # See Petersen, Hoffmann, Rostalski (2018): ON APPROXIMATE NONLINEAR GAUSSIAN MESSAGE PASSING ON FACTOR GRAPHS
        assert isinstance(self.port_b.in_msg, GaussianTildeMessage)
        info_a_fwd = self.port_a.in_msg.convert(GaussianWeightedMeanInfoMessage).info
        W_tilde_b_bwd = self.port_b.in_msg.W
        xi_tilde_b_bwd = self.port_b.in_msg.xi
        cr = self.cross_covariance
        W_tilde_a_bwd = info_a_fwd @ cr @ W_tilde_b_bwd @ cr.transpose() @ info_a_fwd
        xi_tilde_a_bwd = info_a_fwd @ cr @ xi_tilde_b_bwd
        return GaussianTildeMessage(xi_tilde_a_bwd, W_tilde_a_bwd)

    def _calc_msg_a(self, backward_pass_mode=None):
        if backward_pass_mode is not None:
            backwards_pass_method = self.get_backward_pass_method(backward_pass_mode)
        else:
            backwards_pass_method = self.backward_pass_method
        # Backward direction
        if not (callable(self.inverse_function)):
            if self.cross_covariance is None:
                raise RuntimeError("Can't calculate reverse without previous forward pass or inverse function.")
            return backwards_pass_method()

        else:
            msg, _ = self.port_a.in_msg.unscented_transform(self.inverse_function,
                                                            sigma_point_scheme=self.sigma_point_scheme,
                                                            alpha=self.alpha)
            return msg

    def _calc_msg_b(self):
        # Forward direction
        if self.forward_marginal_passing:
            if self.port_a.out_msg is None:
                a_margin = self.port_a.in_msg.convert(GaussianMeanCovMessage)
            else:
                a_margin = self.port_a.marginal(target_type=GaussianMeanCovMessage)

            self.b_margin_old, self.cross_covariance \
                = a_margin.unscented_transform(self.function, sigma_point_scheme=self.sigma_point_scheme,
                                               alpha=self.alpha)

        else:
            self.b_margin_old, self.cross_covariance = \
                self.port_a.in_msg.unscented_transform(self.function, sigma_point_scheme=self.sigma_point_scheme,
                                                       alpha=self.alpha)

        return self.b_margin_old

    def __repr__(self):
        if self.name is None:
            name = str(id(self))
        else:
            name = self.name

        return type(self).__name__ + "(" + name + ", Function:" + repr(self.function) + ")"

    def get_ports(self):
        return [self.port_a, self.port_b]


class StatisticalLinearizationNode(Node):
    """
          x1 +----------+ y
         --->|          |
         --->| function |--->
         --->|          |
          xm +----------+



                        | n ~ N(n, Vn)
                        V
         x1 +----+    +---+ y
        --->| M1 |--->| + |--->
            +----+    +---+
               :        A
               :        |
         xm +----+      |
        --->| Mm |------+
            +----+
    """

    class BackwardPassMode(Enum):
        MeanCov = 1  # Naive implementation of Särkkä's unscented RTS paper (2008)
        WeightedMeanInfo = 2  # Implementation of algorithm from Maximilian Pilz' thesis (2017)
        Tilde = 3  # Implementation of algorithm from Freiburg-paper (Petersen, Hoffmann, Rostalski 2018)

    def __init__(self, function, number_of_in_ports=1, number_of_out_ports=1,
                 init_in_port_out_msgs=None, init_out_port_out_msgs=None, name=None, inverse_func=None,
                 sigma_point_scheme=None,
                 alpha=None, backward_pass_mode=None, linearization_about_marginal=False, expectation_propagation=False,
                 M_matrices_init=None, n_offset_init=None, cov_err_init=None):
        """

        :param function:
        :param name: Node name
        :param inverse_func: Inverse of nonlinear function to pass through, if available. If available, normal unscented
        pass is performed backwards. Otherweise, approximative RTS pass is perforemd in the bw direction.
        :param sigma_point_scheme: Selects method to choose sigma points and weights for the UT. See UT functions for
        details.
        :param alpha: Parameter for sigma point calculation in the UT (see there for details).
        :param backward_pass_mode: Selects default Rauch-Tung-Striebel (RTS) type backwards pass implementation. See
        UnscentedNode.BackwardsPassMode for options. Can be overriden explicitly when calling self._calc_msg_a_.
        :param linearization_about_marginal: If False (default), UT is performed for directed messages in the fwd
        direction, and those are also returned by port_b.calc_msg(). If True, UT is performed for marginals instead and
        those are also returned *AS A DIRECTED MESSAGE* (yes, this may be unexpected and should be changed in the
        future).
        """
        super().__init__(name=name)

        self.number_of_out_ports = number_of_out_ports
        self.number_of_in_ports = number_of_in_ports
        self.n_in_dim = [1] * number_of_in_ports
        self.n_out_dim = [1] * number_of_out_ports

        # RTS backwards pass implementation to use
        if backward_pass_mode is None:
            backward_pass_mode = StatisticalLinearizationNode.BackwardPassMode.MeanCov
        self.backward_pass_mode = backward_pass_mode
        self.backward_pass_method = self.get_backward_pass_method(lambda i: i, backward_pass_mode)

        self.ports_out = [NodePort(self, lambda i=i: self._calc_msg_out(i), NodePortType.OutPort) for i in
                          range(self.number_of_out_ports)]
        self.ports_in = [NodePort(self, lambda i=i: self._calc_msg_in(i, backward_pass_mode), NodePortType.InPort) for i
                         in range(self.number_of_in_ports)]

        # Make node compatible with two-port node
        if len(self.ports_in) == 1:
            self.port_a = self.ports_in[0]
        if len(self.ports_out) == 1:
            self.port_b = self.ports_out[0]

        # Handles for (nonlinear) functions to pass through
        self.function = function
        self.inverse_function = inverse_func

        # Params for unscented transform
        self.alpha = alpha
        self.sigma_point_scheme = sigma_point_scheme

        for i in range(self.number_of_in_ports):
            # ToDo: Add assertions
            if init_in_port_out_msgs is not None:
                self.ports_in[i].out_msg = init_in_port_out_msgs[i]
            else:
                self.ports_in[i].out_msg = GaussianNonInformativeMessage(1, direction=PortMessageDirection.Backward,
                                                                         inf_approx=None)
        for i in range(self.number_of_out_ports):
            # ToDo: Add assertions
            if init_out_port_out_msgs is not None:
                self.ports_out[i].out_msg = init_out_port_out_msgs[i]
            else:
                self.ports_out[i].out_msg = GaussianNonInformativeMessage(1, direction=PortMessageDirection.Forward,
                                                                          inf_approx=None)

        self.M_matrix = [None] * self.number_of_in_ports
        self.n_offset = None
        self.n_offsets = [None] * self.number_of_in_ports
        self.cov_errs = [None] * self.number_of_in_ports
        self.cov_err = None

        # ToDo: Add assertions
        if (M_matrices_init is not None) & (n_offset_init is not None) & (self.cov_err is not None):
            for i in range(self.number_of_in_ports):
                assert (isinstance(M_matrices_init[i], matrix))
                self.M_matrix[i] = M_matrices_init[i]
            self.n_offset = n_offset_init
            self.cov_err = cov_err_init

        # Initialize helper matrices and vectors from linearization about marginal
        self._M_marg_matrix = [None] * self.number_of_in_ports
        self._n_marg_offsets = [None] * self.number_of_in_ports
        self._n_marg_offset = None
        self._cov_marg_errs = [None] * self.number_of_in_ports
        self._cov_marg_err = None
        self._xy_ccovs = [None] * self.number_of_in_ports

        # Forward message passing type
        self.linearization_about_marginal = linearization_about_marginal
        self.expectation_propagation = expectation_propagation
        self.y_marginal = None
        self.x_marginal = None
        # if marginal_passing:
        #     raise DeprecationWarning('The current implementation of forward marginal passing is old and deprecated.')

        self._in_node = [
            PriorNode(GaussianNonInformativeMessage(self.n_in_dim[i], direction=PortMessageDirection.Backward,
                                                    inf_approx=None)) for i in range(self.number_of_in_ports)]
        self._out_node = [
            PriorNode(GaussianNonInformativeMessage(self.n_out_dim[i], direction=PortMessageDirection.Forward,
                                                    inf_approx=None)) for i in range(self.number_of_out_ports)]
        self._n_offset_prior = PriorNode(
            GaussianNonInformativeMessage(self.n_out_dim[0], direction=PortMessageDirection.Forward,
                                          inf_approx=None))
        self._M_node = [MatrixNode(self.M_matrix[i]) for i in range(self.number_of_in_ports)]
        self._add_n_offset_node = BigAdditionNode(self.number_of_in_ports + 1, self.number_of_out_ports)

        # Connect nodes
        for i in range(self.number_of_in_ports):
            self._in_node[i].port_a.connect(self._M_node[i].port_a)
            self._M_node[i].port_b.connect(self._add_n_offset_node.ports_in[i])

        self._n_offset_prior.port_a.connect(self._add_n_offset_node.ports_in[-1])

        for i in range(self.number_of_out_ports):
            self._add_n_offset_node.ports_out[i].connect(self._out_node[i].port_a)

    def get_backward_pass_method(self, in_port_index, backward_pass_mode):
        assert isinstance(backward_pass_mode, StatisticalLinearizationNode.BackwardPassMode)
        if backward_pass_mode == StatisticalLinearizationNode.BackwardPassMode.MeanCov:
            def backward_pass_method(i=in_port_index):
                return self.rts_mean_cov(i)
        elif backward_pass_mode == StatisticalLinearizationNode.BackwardPassMode.WeightedMeanInfo:
            def backward_pass_method(i=in_port_index):
                return self.rts_weighted_mean_info(i)
        elif backward_pass_mode == StatisticalLinearizationNode.BackwardPassMode.Tilde:
            def backward_pass_method(i=in_port_index):
                return self.rts_tilde(i)
        else:
            raise NotImplementedError
        return backward_pass_method

    def rts_mean_cov(self, in_port_index):
        # See Simo Särkkä, Unscented Rauch-Tung-Striebel Smoother.
        # Very naive implementation
        out_marginal_msg = self.ports_out[0].marginal(target_type=GaussianMeanCovMessage)
        out_fwd_msg = self.ports_out[0].out_msg.convert(GaussianMeanCovMessage)
        in_fwd_msg = self.ports_in[in_port_index].in_msg.convert(GaussianMeanCovMessage)

        D = in_fwd_msg.cov @ self.M_matrix[in_port_index].T @ inv(out_fwd_msg.cov)
        mean = in_fwd_msg.mean + D @ (out_marginal_msg.mean - out_fwd_msg.mean)
        cov = in_fwd_msg.cov + D @ (out_marginal_msg.cov - out_fwd_msg.cov) @ D.T

        a_margin = GaussianMeanCovMessage(mean, cov)
        a_out_messsage = a_margin / in_fwd_msg

        return a_out_messsage

    def rts_weighted_mean_info(self, in_port_index):
        # Naive copy & paste implementation
        # ToDo: Improve
        out_marginal_msg = self.ports_out[0].marginal(target_type=GaussianMeanCovMessage)
        out_fwd_msg = self.ports_out[0].out_msg.convert(GaussianMeanCovMessage)
        in_fwd_msg = self.ports_in[in_port_index].in_msg.convert(GaussianMeanCovMessage)

        D = in_fwd_msg.cov @ self.M_matrix[in_port_index].T @ inv(out_fwd_msg.cov)
        mean = in_fwd_msg.mean + D @ (out_marginal_msg.mean - out_fwd_msg.mean)
        cov = in_fwd_msg.cov + D @ (out_marginal_msg.cov - out_fwd_msg.cov) @ D.T

        a_margin = GaussianMeanCovMessage(mean, cov)
        a_out_messsage = a_margin / in_fwd_msg

        a_out_messsage = a_out_messsage.convert(GaussianWeightedMeanInfoMessage)

        return a_out_messsage

    def rts_tilde(self, in_port_index):
        # See Petersen, Hoffmann, Rostalski (2018): ON APPROXIMATE NONLINEAR GAUSSIAN MESSAGE PASSING ON FACTOR GRAPHS
        assert isinstance(self.ports_out[0].in_msg, GaussianTildeMessage)
        M_matrix = self.M_matrix[in_port_index]
        W_tilde_out_bwd = self.ports_out[0].in_msg.W
        xi_tilde_out_bwd = self.ports_out[0].in_msg.xi

        W_tilde_in_bwd = M_matrix.T @ W_tilde_out_bwd @ M_matrix
        xi_tilde_in_bwd = M_matrix.T @ xi_tilde_out_bwd

        return GaussianTildeMessage(xi_tilde_in_bwd, W_tilde_in_bwd)

    def _calc_msg_in(self, index, backward_pass_mode=None):
        if backward_pass_mode is not None:
            backwards_pass_method = self.get_backward_pass_method(index, backward_pass_mode)
        else:
            backwards_pass_method = self.backward_pass_method[index]
        # Backward direction
        if not callable(self.inverse_function):
            # If no inverse function is provided, either use the linearization from the forward pass to do the backward
            # message passing or use substitution
            if self.expectation_propagation is False:
                if self._xy_ccovs is None:
                    raise RuntimeError("Can't calculate reverse without previous forward pass or inverse function.")
                return backwards_pass_method()

            else:  # (self.expectation_propagation is True):
                if not (self.number_of_out_ports == 1):
                    raise NotImplementedError('Node without inverse function and expectation propagation is currently '
                                              'only functioning for a single output port!')

                port_out_msgs_to_linearize_about = [None] * self.number_of_out_ports
                port_in_msgs_to_linearize_about = [None] * self.number_of_in_ports

                # if self.linearization_about_marginal is True:
                #     for i in range(self.number_of_out_ports):
                #         if (self.ports_out[i].out_msg is None) | (
                #                 isinstance(self.ports_out[i].out_msg, GaussianNonInformativeMessage)):
                #             # Use the incoming message, if no marginal is available
                #             port_out_msgs_to_linearize_about[i] = self.ports_out[i].in_msg.convert(
                #                 GaussianMeanCovMessage)
                #         else:
                #             port_out_msgs_to_linearize_about[i] = self.ports_out[i].marginal(
                #                 target_type=GaussianMeanCovMessage)
                # else:
                #     for i in range(self.number_of_out_ports):
                #         # Always use the incoming messages to linearize about
                #         port_out_msgs_to_linearize_about[i] = self.ports_out[i].in_msg.convert(GaussianMeanCovMessage)

                for i in range(self.number_of_in_ports):
                    # Always use the incoming messages to linearize about
                    port_in_msgs_to_linearize_about[i] = self.ports_in[i].in_msg.convert(GaussianMeanCovMessage)

                noise_msg = GaussianMeanCovMessage([[0.0]], [[100.0]])
                port_in_msgs_to_linearize_about.append(noise_msg)

                for i in range(self.number_of_out_ports):
                    # Always use the incoming messages to linearize about
                    port_out_msgs_to_linearize_about[i] = self.ports_out[i].in_msg.convert(GaussianMeanCovMessage)

                # Produce one large message to linearize about by stacking all the input port marginals
                port_in_msg_to_linearize_about = self.stack_port_msgs(port_in_msgs_to_linearize_about,
                                                                      self.number_of_in_ports + 1)
                port_out_msg_to_linearize_about = self.stack_port_msgs(port_out_msgs_to_linearize_about,
                                                                       self.number_of_out_ports)

                def msg_out_in_pdf(y):
                    msg_out_in = port_out_msg_to_linearize_about
                    cov = msg_out_in.cov
                    mean = msg_out_in.mean
                    ny = shape(mean)[0]
                    nrmlz = det(cov) ** (-0.5) / sqrt((2 * pi) ** ny)

                    return nrmlz * exp(-0.5 * (y - mean).T @ inv(cov) @ (y - mean))

                def fun_aug_mean(x):

                    fun = self.function
                    z = squeeze(x)[1]
                    x = atleast_2d(x[0])

                    # x * p_y_bwd( fun(x) )
                    return x @ msg_out_in_pdf(fun(x + z))

                # self.x_marginal, _ = port_in_msg_to_linearize_about. \
                #     unscented_transform(fun_aug_mean,
                #                         sigma_point_scheme=self.sigma_point_scheme,
                #                         alpha=self.alpha)

                mean, _, _ = unscented_transform_gaussian(port_in_msg_to_linearize_about.mean,
                                                          port_in_msg_to_linearize_about.cov,
                                                          fun_aug_mean,
                                                          sigma_point_scheme=self.sigma_point_scheme,
                                                          alpha=self.alpha,
                                                          degree_of_exactness=10)

                def fun_aug_cov(x):

                    fun = self.function
                    z = squeeze(x)[1]
                    x = atleast_2d(x[0])

                    # (x - m)(x - m).T * p_y_bwd( fun(x) )
                    return (x - mean) @ (x - mean).T @ msg_out_in_pdf(fun(x + z))

                cov, _, _ = unscented_transform_gaussian(port_in_msg_to_linearize_about.mean,
                                                         port_in_msg_to_linearize_about.cov,
                                                         fun_aug_cov,
                                                         sigma_point_scheme=self.sigma_point_scheme,
                                                         alpha=self.alpha,
                                                         degree_of_exactness=10)

                self.x_marginal = GaussianMeanCovMessage(mean, cov)

                port_in_msg = self.x_marginal

                if self.linearization_about_marginal is True:
                    # Gaussian division to obtain forward message
                    # ToDo: Implement Gaussian Division for Tilde message?!
                    if isinstance(self.ports_in[0].in_msg, GaussianTildeMessage):
                        port_in_msg_in_msg = self.ports_in[0].in_msg.convert(target_type=GaussianMeanCovMessage,
                                                                             other_msg=self.ports_in[0].out_msg)
                    elif isinstance(self.ports_in[0].in_msg, GaussianWeightedMeanInfoMessage):
                        # ToDo: Probably not necessary
                        port_in_msg_in_msg = self.ports_in[0].in_msg.convert(GaussianMeanCovMessage)
                    elif isinstance(self.ports_out[0].in_msg, GaussianMeanCovMessage):
                        port_in_msg_in_msg = self.ports_in[0].in_msg
                    else:
                        raise NotImplementedError('The incoming message is supposed to be either a '
                                                  'GaussianTildeMessage, a GaussianWeightedMeanInfoMessage or a '
                                                  'GaussianMeanCovMessage!')

                    port_in_msg = self.x_marginal / port_in_msg_in_msg

                return port_in_msg

        # Using the inverse function perform sigma point-based calculation of mean and variance based on either the
        # directed message or the marginal message. In the end, compute directed message by Gaussian division.
        else:
            if not self.number_of_out_ports == 1:
                raise NotImplementedError(
                    'Node with inverse function is currently only functioning for a single output port!')
            if not self.number_of_in_ports == 1:
                raise NotImplementedError(
                    'Node with inverse function is currently only functioning for a single input port!')

            port_out_msgs_to_linearize_about = [None] * self.number_of_out_ports

            if self.linearization_about_marginal is True:
                for i in range(self.number_of_out_ports):
                    if (self.ports_out[i].out_msg is None) | (
                            isinstance(self.ports_out[i].out_msg, GaussianNonInformativeMessage)):
                        # Use the incoming message, if no marginal is available
                        port_out_msgs_to_linearize_about[i] = self.ports_out[i].in_msg.convert(GaussianMeanCovMessage)
                    else:
                        port_out_msgs_to_linearize_about[i] = self.ports_out[i].marginal(
                            target_type=GaussianMeanCovMessage)
            else:
                for i in range(self.number_of_out_ports):
                    # Always use the incoming messages to linearize about
                    port_out_msgs_to_linearize_about[i] = self.ports_out[i].in_msg.convert(GaussianMeanCovMessage)

            # Produce one large message to linearize about by stacking all the input port marginals
            port_out_msg_to_linearize_about = self.stack_port_msgs(port_out_msgs_to_linearize_about,
                                                                   self.number_of_out_ports)

            self.x_marginal, _ = port_out_msg_to_linearize_about. \
                unscented_transform(self.inverse_function,
                                    sigma_point_scheme=self.sigma_point_scheme,
                                    alpha=self.alpha)

            port_in_msg = self.x_marginal

            if self.linearization_about_marginal is True:
                # Gaussian division to obtain forward message
                # ToDo: Implement Gaussian Division for Tilde message?!
                if isinstance(self.ports_in[0].in_msg, GaussianTildeMessage):
                    port_in_msg_in_msg = self.ports_in[0].in_msg.convert(target_type=GaussianMeanCovMessage,
                                                                         other_msg=self.ports_in[0].out_msg)
                elif isinstance(self.ports_in[0].in_msg, GaussianWeightedMeanInfoMessage):
                    # ToDo: Probably not necessary
                    port_in_msg_in_msg = self.ports_in[0].in_msg.convert(GaussianMeanCovMessage)
                elif isinstance(self.ports_out[0].in_msg, GaussianMeanCovMessage):
                    port_in_msg_in_msg = self.ports_in[0].in_msg
                else:
                    raise NotImplementedError(
                        'The incoming message is supposed to be either a GaussianTildeMessage, a '
                        'GaussianWeightedMeanInfoMessage or a GaussianMeanCovMessage!')

                port_in_msg = self.x_marginal / port_in_msg_in_msg

            return port_in_msg

    def _calc_msg_out(self, index):
        # Forward direction

        port_in_msgs_to_linearize_about = [None] * self.number_of_in_ports

        if self.linearization_about_marginal:
            for i in range(self.number_of_in_ports):
                if (self.ports_in[i].out_msg is None) | (
                        isinstance(self.ports_in[i].out_msg, GaussianNonInformativeMessage)):
                    # Use the incoming message, if no marginal is available
                    port_in_msgs_to_linearize_about[i] = self.ports_in[i].in_msg.convert(GaussianMeanCovMessage)
                else:
                    port_in_msgs_to_linearize_about[i] = self.ports_in[i].marginal(target_type=GaussianMeanCovMessage)
        else:
            for i in range(self.number_of_in_ports):
                # Always use the incoming messages to linearize about
                port_in_msgs_to_linearize_about[i] = self.ports_in[i].in_msg.convert(GaussianMeanCovMessage)

        # new_list = [expression(i) for i in old_list if filter(i)]
        n_in = [shape(port_in_msgs_to_linearize_about[i].mean)[0] for i in range(self.number_of_in_ports)]
        nbegins = [sum(n_in[0:i]) for i in range(self.number_of_in_ports)]
        nends = [sum(n_in[0:i]) + n_in[i] for i in range(self.number_of_in_ports)]

        # Allow the function to segment the stacked msg
        def fun_arg_segmented(stacked_xs):
            xs = [stacked_xs[nbegins[i]:nends[i]] for i in range(self.number_of_in_ports)]
            # Use splat operator to expand list into several function arguments
            return self.function(*xs)

        # Produce one large message to linearize about by stacking all the input port marginals
        port_in_msg_to_linearize_about = self.stack_port_msgs(port_in_msgs_to_linearize_about, self.number_of_in_ports)

        self.y_marginal, self.xy_ccov = \
            port_in_msg_to_linearize_about.unscented_transform(fun_arg_segmented,
                                                               sigma_point_scheme=self.sigma_point_scheme,
                                                               alpha=self.alpha)

        self._xy_ccovs = [self.xy_ccov[nbegins[i]:nends[i], :] for i in range(self.number_of_in_ports)]
        self._M_marg_matrix = [self._xy_ccovs[i].T @ inv(port_in_msgs_to_linearize_about[i].cov) for i in
                               range(self.number_of_in_ports)]

        self._n_marg_offsets = [self._M_marg_matrix[i] @ port_in_msgs_to_linearize_about[i].mean for i in
                                range(self.number_of_in_ports)]

        self._n_marg_offset = self.y_marginal.mean - sum(self._n_marg_offsets)

        self._cov_marg_errs = [
            self._M_marg_matrix[i] @ port_in_msgs_to_linearize_about[i].cov @ self._M_marg_matrix[i].T for i in
            range(self.number_of_in_ports)]

        self._cov_marg_err = self.y_marginal.cov - sum(self._cov_marg_errs)

        # Check if this is the first time, the linearization is performed
        # or if linearization is not to be performed about the marginal
        if (self.n_offset is None) | \
                any([self.ports_out[i].in_msg is None for i in range(self.number_of_out_ports)]) | \
                (self.linearization_about_marginal is not True):
            self.M_matrix = [self._M_marg_matrix[i] for i in range(self.number_of_in_ports)]
            self.n_offsets = [self._n_marg_offsets[i] for i in range(self.number_of_in_ports)]
            self.n_offset = self._n_marg_offset
            self.cov_errs = self._cov_marg_errs
            self.cov_err = self._cov_marg_err

            port_out_msg = self.y_marginal

        # If EP mode is on, then transform the linearization parameters from the marginals to work
        # for the directed messages
        elif (self.expectation_propagation is True):
            # Gaussian division to obtain forward message
            # ToDo: Implement Gaussian Division for Tilde message?!
            if isinstance(self.ports_out[0].in_msg, GaussianTildeMessage):
                port_out_msg_in_msg = self.ports_out[0].in_msg.convert(target_type=GaussianMeanCovMessage,
                                                                       other_msg=self.ports_out[0].out_msg)
            elif isinstance(self.ports_out[0].in_msg, GaussianWeightedMeanInfoMessage):
                # ToDo: Probably not necessary
                port_out_msg_in_msg = self.ports_out[0].in_msg.convert(GaussianMeanCovMessage)
            elif isinstance(self.ports_out[0].in_msg, GaussianMeanCovMessage):
                port_out_msg_in_msg = self.ports_out[0].in_msg
            else:
                raise NotImplementedError(
                    'The incoming message is supposed to be either a GaussianTildeMessage, a '
                    'GaussianWeightedMeanInfoMessage or a GaussianMeanCovMessage!')

            port_out_msg = self.y_marginal / port_out_msg_in_msg

            # Moment matching-based update of the matrices
            # Formulae involving the parameters resulting from the linearization about the marginals
            self.M_matrix = [
                port_out_msg.cov @ inv(self.y_marginal.cov) @ self._M_marg_matrix[i] @ port_in_msgs_to_linearize_about[
                    i].cov @ inv(self.ports_in[i].in_msg.cov) for i in range(self.number_of_in_ports)]
            self.n_offsets = [
                self._M_marg_matrix[i] @ port_in_msgs_to_linearize_about[i].cov @ self.ports_in[i].out_msg.convert(
                    GaussianWeightedMeanInfoMessage).weighted_mean
                for i in range(self.number_of_in_ports)]
            self.n_offset = port_out_msg.cov @ inv(self.y_marginal.cov) @ \
                sum(self.n_offsets) + port_out_msg.cov \
                @ (inv(self.y_marginal.cov) @ self._n_marg_offset
                   - port_out_msg_in_msg.convert(GaussianWeightedMeanInfoMessage).weighted_mean)
            self.cov_errs = [self.M_matrix[i] @ self.ports_in[i].in_msg.cov @ self.M_matrix[i].T
                             for i in range(self.number_of_in_ports)]
            self.cov_err = port_out_msg.cov - sum(self.cov_errs)

            # Formulae that are simplified by inserting the closed-forms of statistical linearization-based M, n and V_E
            # self.M_matrix = [port_out_msg.cov @ inv(self.y_marginal.cov) @ self._xy_ccovs[i].T @ inv(
            #     self.ports_in[i].in_msg.cov) for i in range(self.number_of_in_ports)]
            # self.n_offsets = [self._xy_ccovs[i].T @ inv(self.ports_in[i].in_msg.cov) @ self.ports_in[i].in_msg.mean
            #                                  for i in range(self.number_of_in_ports)]
            # self.n_offset = port_out_msg.mean - port_out_msg.cov @ inv(self.y_marginal.cov) @ sum(self.n_offsets)
            # self.cov_errs = [self._xy_ccovs[i].T @ inv(self.ports_in[i].in_msg.cov) @ self._xy_ccovs[i]
            #                                  for i in range(self.number_of_in_ports)]
            # self.cov_err = port_out_msg.cov - port_out_msg.cov @ inv(self.y_marginal.cov) @ sum(self.cov_errs) \
            #                @ inv(self.y_marginal.cov) @ port_out_msg.cov

        else:
            self.M_matrix = [self._M_marg_matrix[i] for i in range(self.number_of_in_ports)]
            self.n_offsets = [self._n_marg_offsets[i] for i in range(self.number_of_in_ports)]
            self.n_offset = self._n_marg_offset
            self.cov_errs = self._cov_marg_errs
            self.cov_err = self._cov_marg_err

        # Update prior nodes, matrix nodes and do the message passing
        # Doing the message passing in forward direction is actually not necessary,
        # but it is performed here as a check of correctness
        for i in range(self.number_of_in_ports):
            self._M_node[i].matrix = self.M_matrix[i]
            self._in_node[i].update_prior(self.ports_in[i].in_msg)
            self._M_node[i].port_b.update(GaussianMeanCovMessage)

        for i in range(self.number_of_out_ports):
            if self.ports_out[i].in_msg is not None:
                self._out_node[i].update_prior(self.ports_out[i].in_msg)
            else:
                self._out_node[i].update_prior(
                    GaussianNonInformativeMessage(shape(self.n_offset)[0], direction=PortMessageDirection.Backward,
                                                  inf_approx=None))

        self._n_offset_prior.update_prior(GaussianMeanCovMessage(self.n_offset, self.cov_err))
        result = self._add_n_offset_node.ports_out[0].update(GaussianMeanCovMessage)

        # assert(allclose(result.mean, port_out_msg.mean))

        return result

    def stack_port_msgs(self, msgs, num_ports):
        assert (isinstance(msgs[0], GaussianMeanCovMessage))

        msgs_means = [col_vec(msgs[i].mean) for i in range(num_ports)]
        msgs_covs = [msgs[i].cov for i in range(num_ports)]
        mean = vstack(msgs_means)
        cov = block_diag(*msgs_covs)
        msg = GaussianMeanCovMessage(mean, cov)

        return msg

    def __repr__(self):
        if self.name is None:
            name = str(id(self))
        else:
            name = self.name

        return type(self).__name__ + "(" + name + ", Function:" + repr(self.function) + ")"

    def get_ports(self):
        return self.ports_in + self.ports_out


class Jacobian_or_Statistical_LinearizationNode(Node):
    """
          x1 +----------+ y
         --->|          |
         --->| function |--->
         --->|          |
          xm +----------+



                        | n ~ N(n, Vn)
                        V
         x1 +----+    +---+ y
        --->| M1 |--->| + |--->
            +----+    +---+
               :        A
               :        |
         xm +----+      |
        --->| Mm |------+
            +----+
    """

    class BackwardPassMode(Enum):
        MeanCov = 1  # Naive implementation of Särkkä's unscented RTS paper (2008)
        WeightedMeanInfo = 2  # Implementation of algorithm from Maximilian Pilz' thesis (2017)
        Tilde = 3  # Implementation of algorithm from Freiburg-paper (Petersen, Hoffmann, Rostalski 2018)

    def __init__(self, function, number_of_in_ports=1, number_of_out_ports=1,
                 init_in_port_out_msgs=None, init_out_port_out_msgs=None, name=None, inverse_func=None, deriv_func=None,
                 sigma_point_scheme=None,
                 alpha=None, backward_pass_mode=None, linearization_about_marginal=False, jacobian_linearization=False,
                 expectation_propagation=False, M_matrices_init=None, n_offset_init=None, cov_err_init=None):
        """

        :param function:
        :param name: Node name
        :param inverse_func: Inverse of nonlinear function to pass through, if available. If available, normal unscented
        pass is performed backwards. Otherweise, approximative RTS pass is perforemd in the bw direction.
        :param sigma_point_scheme: Selects method to choose sigma points and weights for the UT. See UT functions for
        details.
        :param alpha: Parameter for sigma point calculation in the UT (see there for details).
        :param backward_pass_mode: Selects default Rauch-Tung-Striebel (RTS) type backwards pass implementation. See
        UnscentedNode.BackwardsPassMode for options. Can be overriden explicitly when calling self._calc_msg_a_.
        :param linearization_about_marginal: If False (default), UT is performed for directed messages in the fwd
        direction, and those are also returned by port_b.calc_msg(). If True, UT is performed for marginals instead and
        those are also returned *AS A DIRECTED MESSAGE* (yes, this may be unexpected and should be changed in the
        future).
        """
        super().__init__(name=name)

        self.number_of_out_ports = number_of_out_ports
        self.number_of_in_ports = number_of_in_ports
        self.n_in_dim = [1] * number_of_in_ports
        self.n_out_dim = [1] * number_of_out_ports

        # RTS backwards pass implementation to use
        if backward_pass_mode is None:
            backward_pass_mode = Jacobian_or_Statistical_LinearizationNode.BackwardPassMode.MeanCov
        self.backward_pass_mode = backward_pass_mode
        self.backward_pass_method = self.get_backward_pass_method(lambda i: i, backward_pass_mode)

        self.ports_out = [NodePort(self, lambda i=i: self._calc_msg_out(i), NodePortType.OutPort) for i in
                          range(self.number_of_out_ports)]
        self.ports_in = [NodePort(self, lambda i=i: self._calc_msg_in(i, backward_pass_mode), NodePortType.InPort) for i
                         in range(self.number_of_in_ports)]

        # Make node compatible with two-port node
        if len(self.ports_in) == 1:
            self.port_a = self.ports_in[0]
        if len(self.ports_out) == 1:
            self.port_b = self.ports_out[0]

        # Handles for (nonlinear) functions to pass through
        self.function = function
        self.inverse_function = inverse_func
        self.derivative_function = deriv_func

        # Params for unscented transform
        self.alpha = alpha
        self.sigma_point_scheme = sigma_point_scheme

        for i in range(self.number_of_in_ports):
            # ToDo: Add assertions
            if init_in_port_out_msgs is not None:
                self.ports_in[i].out_msg = init_in_port_out_msgs[i]
            else:
                self.ports_in[i].out_msg = GaussianNonInformativeMessage(1, direction=PortMessageDirection.Backward,
                                                                         inf_approx=None)
        for i in range(self.number_of_out_ports):
            # ToDo: Add assertions
            if init_out_port_out_msgs is not None:
                self.ports_out[i].out_msg = init_out_port_out_msgs[i]
            else:
                self.ports_out[i].out_msg = GaussianNonInformativeMessage(1, direction=PortMessageDirection.Forward,
                                                                          inf_approx=None)

        self.M_matrix = [None] * self.number_of_in_ports
        self.n_offset = None
        self.n_offsets = [None] * self.number_of_in_ports
        self.cov_errs = [None] * self.number_of_in_ports
        self.cov_err = None

        # ToDo: Add assertions
        if (M_matrices_init is not None) & (n_offset_init is not None) & (self.cov_err is not None):
            for i in range(self.number_of_in_ports):
                assert (isinstance(M_matrices_init[i], matrix))
                self.M_matrix[i] = M_matrices_init[i]
            self.n_offset = n_offset_init
            self.cov_err = cov_err_init

        # Initialize helper matrices and vectors from linearization about marginal
        self._M_marg_matrix = [None] * self.number_of_in_ports
        self._n_marg_offsets = [None] * self.number_of_in_ports
        self._n_marg_offset = None
        self._cov_marg_errs = [None] * self.number_of_in_ports
        self._cov_marg_err = None
        self._xy_ccovs = [None] * self.number_of_in_ports

        # Forward message passing type
        self.linearization_about_marginal = linearization_about_marginal
        self.expectation_propagation = expectation_propagation
        self.jacobian_linearization = jacobian_linearization
        self.y_marginal = None
        self.x_marginal = None
        # if marginal_passing:
        #     raise DeprecationWarning('The current implementation of forward marginal passing is old and deprecated.')

        self._in_node = [
            PriorNode(GaussianNonInformativeMessage(self.n_in_dim[i], direction=PortMessageDirection.Backward,
                                                    inf_approx=None)) for i in range(self.number_of_in_ports)]
        self._out_node = [
            PriorNode(GaussianNonInformativeMessage(self.n_out_dim[i], direction=PortMessageDirection.Forward,
                                                    inf_approx=None)) for i in range(self.number_of_out_ports)]
        self._n_offset_prior = PriorNode(
            GaussianNonInformativeMessage(self.n_out_dim[0], direction=PortMessageDirection.Forward,
                                          inf_approx=None))
        self._M_node = [MatrixNode(self.M_matrix[i]) for i in range(self.number_of_in_ports)]
        self._add_n_offset_node = BigAdditionNode(self.number_of_in_ports + 1, self.number_of_out_ports)

        # Connect nodes
        for i in range(self.number_of_in_ports):
            self._in_node[i].port_a.connect(self._M_node[i].port_a)
            self._M_node[i].port_b.connect(self._add_n_offset_node.ports_in[i])

        self._n_offset_prior.port_a.connect(self._add_n_offset_node.ports_in[-1])

        for i in range(self.number_of_out_ports):
            self._add_n_offset_node.ports_out[i].connect(self._out_node[i].port_a)

    def get_backward_pass_method(self, in_port_index, backward_pass_mode):
        assert isinstance(backward_pass_mode, Jacobian_or_Statistical_LinearizationNode.BackwardPassMode)
        if backward_pass_mode == Jacobian_or_Statistical_LinearizationNode.BackwardPassMode.MeanCov:
            def backward_pass_method(i=in_port_index):
                return self.rts_mean_cov(i)
        elif backward_pass_mode == Jacobian_or_Statistical_LinearizationNode.BackwardPassMode.WeightedMeanInfo:
            def backward_pass_method(i=in_port_index):
                return self.rts_weighted_mean_info(i)
        elif backward_pass_mode == Jacobian_or_Statistical_LinearizationNode.BackwardPassMode.Tilde:
            def backward_pass_method(i=in_port_index):
                return self.rts_tilde(i)
        else:
            raise NotImplementedError
        return backward_pass_method

    def rts_mean_cov(self, in_port_index):
        # See Simo Särkkä, Unscented Rauch-Tung-Striebel Smoother.
        # Very naive implementation
        out_marginal_msg = self.ports_out[0].marginal(target_type=GaussianMeanCovMessage)
        out_fwd_msg = self.ports_out[0].out_msg.convert(GaussianMeanCovMessage)
        in_fwd_msg = self.ports_in[in_port_index].in_msg.convert(GaussianMeanCovMessage)

        D = in_fwd_msg.cov @ self.M_matrix[in_port_index].T @ inv(out_fwd_msg.cov)
        mean = in_fwd_msg.mean + D @ (out_marginal_msg.mean - out_fwd_msg.mean)
        cov = in_fwd_msg.cov + D @ (out_marginal_msg.cov - out_fwd_msg.cov) @ D.T

        a_margin = GaussianMeanCovMessage(mean, cov)
        a_out_messsage = a_margin / in_fwd_msg

        return a_out_messsage

    def rts_weighted_mean_info(self, in_port_index):
        # Naive copy & paste implementation
        # ToDo: Improve
        out_marginal_msg = self.ports_out[0].marginal(target_type=GaussianMeanCovMessage)
        out_fwd_msg = self.ports_out[0].out_msg.convert(GaussianMeanCovMessage)
        in_fwd_msg = self.ports_in[in_port_index].in_msg.convert(GaussianMeanCovMessage)

        D = in_fwd_msg.cov @ self.M_matrix[in_port_index].T @ inv(out_fwd_msg.cov)
        mean = in_fwd_msg.mean + D @ (out_marginal_msg.mean - out_fwd_msg.mean)
        cov = in_fwd_msg.cov + D @ (out_marginal_msg.cov - out_fwd_msg.cov) @ D.T

        a_margin = GaussianMeanCovMessage(mean, cov)
        a_out_messsage = a_margin / in_fwd_msg

        a_out_messsage = a_out_messsage.convert(GaussianWeightedMeanInfoMessage)

        return a_out_messsage

    def rts_tilde(self, in_port_index):
        # See Petersen, Hoffmann, Rostalski (2018): ON APPROXIMATE NONLINEAR GAUSSIAN MESSAGE PASSING ON FACTOR GRAPHS
        assert isinstance(self.ports_out[0].in_msg, GaussianTildeMessage)
        M_matrix = self.M_matrix[in_port_index]
        W_tilde_out_bwd = self.ports_out[0].in_msg.W
        xi_tilde_out_bwd = self.ports_out[0].in_msg.xi

        W_tilde_in_bwd = M_matrix.T @ W_tilde_out_bwd @ M_matrix
        xi_tilde_in_bwd = M_matrix.T @ xi_tilde_out_bwd

        return GaussianTildeMessage(xi_tilde_in_bwd, W_tilde_in_bwd)

    def _calc_msg_in(self, index, backward_pass_mode=None):
        if backward_pass_mode is not None:
            backwards_pass_method = self.get_backward_pass_method(index, backward_pass_mode)
        else:
            backwards_pass_method = self.backward_pass_method[index]
        # Backward direction
        if not (callable(self.inverse_function)):
            # If no inverse function is provided, either use the linearization from the forward pass to do the backward
            # message passing or use substitution
            if (self.expectation_propagation is False):
                if self._xy_ccovs is None:
                    raise RuntimeError("Can't calculate reverse without previous forward pass or inverse function.")
                return backwards_pass_method()

            else:  # (self.expectation_propagation is True):
                if not (self.number_of_out_ports == 1):
                    raise NotImplementedError(
                        'Node without inverse function and expectation propagation is currently only functioning for a '
                        'single output port!')
                if not (self.number_of_in_ports == 1):
                    raise NotImplementedError(
                        'Node without inverse function and expectation propagation  is currently only functioning for '
                        'a single input port!')

                port_out_msgs_to_linearize_about = [None] * self.number_of_out_ports
                port_in_msgs_to_linearize_about = [None] * self.number_of_in_ports

                # if self.linearization_about_marginal is True:
                #     for i in range(self.number_of_out_ports):
                #         if (self.ports_out[i].out_msg is None) | (
                #                 isinstance(self.ports_out[i].out_msg, GaussianNonInformativeMessage)):
                #             # Use the incoming message, if no marginal is available
                #             port_out_msgs_to_linearize_about[i] = self.ports_out[i].in_msg.convert(
                #                 GaussianMeanCovMessage)
                #         else:
                #             port_out_msgs_to_linearize_about[i] = self.ports_out[i].marginal(
                #                 target_type=GaussianMeanCovMessage)
                # else:
                #     for i in range(self.number_of_out_ports):
                #         # Always use the incoming messages to linearize about
                #         port_out_msgs_to_linearize_about[i] = self.ports_out[i].in_msg.convert(GaussianMeanCovMessage)

                for i in range(self.number_of_in_ports):
                    # Always use the incoming messages to linearize about
                    port_in_msgs_to_linearize_about[i] = self.ports_in[i].in_msg.convert(GaussianMeanCovMessage)

                noise_msg = GaussianMeanCovMessage(0.0, 100.0)
                port_in_msgs_to_linearize_about.append(noise_msg)

                for i in range(self.number_of_out_ports):
                    # Always use the incoming messages to linearize about
                    port_out_msgs_to_linearize_about[i] = self.ports_out[i].in_msg.convert(GaussianMeanCovMessage)

                # Produce one large message to linearize about by stacking all the input port marginals
                port_in_msg_to_linearize_about = self.stack_port_msgs(port_in_msgs_to_linearize_about,
                                                                      self.number_of_in_ports + 1)
                port_out_msg_to_linearize_about = self.stack_port_msgs(port_out_msgs_to_linearize_about,
                                                                       self.number_of_out_ports)

                def msg_out_in_pdf(y):
                    msg_out_in = port_out_msg_to_linearize_about
                    cov = msg_out_in.cov
                    mean = msg_out_in.mean
                    ny = shape(mean)[0]
                    nrmlz = det(cov) ** (-0.5) / sqrt((2 * pi) ** ny)

                    return nrmlz * exp(-0.5 * (y - mean).T @ inv(cov) @ (y - mean))

                def fun_aug_mean(x):

                    fun = self.function
                    z = squeeze(x)[1]
                    x = atleast_2d(x[0])

                    # x * p_y_bwd( fun(x) )
                    return x @ msg_out_in_pdf(fun(x + z))

                # self.x_marginal, _ = port_in_msg_to_linearize_about.unscented_transform(
                #     fun_aug_mean,
                #     sigma_point_scheme=self.sigma_point_scheme,
                #     alpha=self.alpha)

                mean, _, _ = unscented_transform_gaussian(port_in_msg_to_linearize_about.mean,
                                                          port_in_msg_to_linearize_about.cov,
                                                          fun_aug_mean,
                                                          sigma_point_scheme=self.sigma_point_scheme,
                                                          alpha=self.alpha,
                                                          degree_of_exactness=10)

                def fun_aug_cov(x):

                    fun = self.function
                    z = squeeze(x)[1]
                    x = atleast_2d(x[0])

                    # (x - m)(x - m).T * p_y_bwd( fun(x) )
                    return (x - mean) @ (x - mean).T @ msg_out_in_pdf(fun(x + z))

                cov, _, _ = unscented_transform_gaussian(port_in_msg_to_linearize_about.mean,
                                                         port_in_msg_to_linearize_about.cov,
                                                         fun_aug_cov,
                                                         sigma_point_scheme=self.sigma_point_scheme,
                                                         alpha=self.alpha,
                                                         degree_of_exactness=10)

                self.x_marginal = GaussianMeanCovMessage(mean, cov)

                port_in_msg = self.x_marginal

                if self.linearization_about_marginal is True:
                    # Gaussian division to obtain forward message
                    # ToDo: Implement Gaussian Division for Tilde message?!
                    if isinstance(self.ports_in[0].in_msg, GaussianTildeMessage):
                        port_in_msg_in_msg = self.ports_in[0].in_msg.convert(target_type=GaussianMeanCovMessage,
                                                                             other_msg=self.ports_in[0].out_msg)
                    elif isinstance(self.ports_in[0].in_msg, GaussianWeightedMeanInfoMessage):
                        # ToDo: Probably not necessary
                        port_in_msg_in_msg = self.ports_in[0].in_msg.convert(GaussianMeanCovMessage)
                    elif isinstance(self.ports_out[0].in_msg, GaussianMeanCovMessage):
                        port_in_msg_in_msg = self.ports_in[0].in_msg
                    else:
                        raise NotImplementedError(
                            'The incoming message is supposed to be either a GaussianTildeMessage, a '
                            'GaussianWeightedMeanInfoMessage or a GaussianMeanCovMessage!')

                    port_in_msg = self.x_marginal / port_in_msg_in_msg

                return port_in_msg

        # Using the inverse function perform sigma point-based calculation of mean and variance based on either the
        # directed message or the marginal message. In the end, compute directed message by Gaussian division.
        else:
            if not (self.number_of_out_ports == 1):
                raise NotImplementedError(
                    'Node with inverse function is currently only functioning for a single output port!')
            if not (self.number_of_in_ports == 1):
                raise NotImplementedError(
                    'Node with inverse function is currently only functioning for a single input port!')

            port_out_msgs_to_linearize_about = [None] * self.number_of_out_ports

            if self.linearization_about_marginal is True:
                for i in range(self.number_of_out_ports):
                    if (self.ports_out[i].out_msg is None) | (
                            isinstance(self.ports_out[i].out_msg, GaussianNonInformativeMessage)):
                        # Use the incoming message, if no marginal is available
                        port_out_msgs_to_linearize_about[i] = self.ports_out[i].in_msg.convert(GaussianMeanCovMessage)
                    else:
                        port_out_msgs_to_linearize_about[i] = self.ports_out[i].marginal(
                            target_type=GaussianMeanCovMessage)
            else:
                for i in range(self.number_of_out_ports):
                    # Always use the incoming messages to linearize about
                    port_out_msgs_to_linearize_about[i] = self.ports_out[i].in_msg.convert(GaussianMeanCovMessage)

            # Produce one large message to linearize about by stacking all the input port marginals
            port_out_msg_to_linearize_about = self.stack_port_msgs(port_out_msgs_to_linearize_about,
                                                                   self.number_of_out_ports)

            self.x_marginal, _ = port_out_msg_to_linearize_about.unscented_transform(
                self.inverse_function,
                sigma_point_scheme=self.sigma_point_scheme,
                alpha=self.alpha)

            port_in_msg = self.x_marginal

            if (self.linearization_about_marginal is True):
                # Gaussian division to obtain forward message
                # ToDo: Implement Gaussian Division for Tilde message?!
                if isinstance(self.ports_in[0].in_msg, GaussianTildeMessage):
                    port_in_msg_in_msg = self.ports_in[0].in_msg.convert(target_type=GaussianMeanCovMessage,
                                                                         other_msg=self.ports_in[0].out_msg)
                elif isinstance(self.ports_in[0].in_msg, GaussianWeightedMeanInfoMessage):
                    # ToDo: Probably not necessary
                    port_in_msg_in_msg = self.ports_in[0].in_msg.convert(GaussianMeanCovMessage)
                elif isinstance(self.ports_out[0].in_msg, GaussianMeanCovMessage):
                    port_in_msg_in_msg = self.ports_in[0].in_msg
                else:
                    raise NotImplementedError(
                        'The incoming message is supposed to be either a GaussianTildeMessage, a '
                        'GaussianWeightedMeanInfoMessage or a GaussianMeanCovMessage!')

                port_in_msg = self.x_marginal / port_in_msg_in_msg

            return port_in_msg

    def _calc_msg_out(self, index):
        # Forward direction

        port_in_msgs_to_linearize_about = [None] * self.number_of_in_ports

        if self.linearization_about_marginal:
            for i in range(self.number_of_in_ports):
                if (self.ports_in[i].out_msg is None) | (
                        isinstance(self.ports_in[i].out_msg, GaussianNonInformativeMessage)):
                    # Use the incoming message, if no marginal is available
                    port_in_msgs_to_linearize_about[i] = self.ports_in[i].in_msg.convert(GaussianMeanCovMessage)
                else:
                    port_in_msgs_to_linearize_about[i] = self.ports_in[i].marginal(target_type=GaussianMeanCovMessage)
        else:
            for i in range(self.number_of_in_ports):
                # Always use the incoming messages to linearize about
                port_in_msgs_to_linearize_about[i] = self.ports_in[i].in_msg.convert(GaussianMeanCovMessage)

        # new_list = [expression(i) for i in old_list if filter(i)]
        n_in = [shape(port_in_msgs_to_linearize_about[i].mean)[0] for i in range(self.number_of_in_ports)]
        nbegins = [sum(n_in[0:i]) for i in range(self.number_of_in_ports)]
        nends = [sum(n_in[0:i]) + n_in[i] for i in range(self.number_of_in_ports)]

        # Allow the function to segment the stacked msg
        def fun_arg_segmented(stacked_xs):
            xs = [stacked_xs[nbegins[i]:nends[i]] for i in range(self.number_of_in_ports)]
            # Use splat operator to expand list into several function arguments
            return self.function(*xs)

        if (self.jacobian_linearization is True):
            def deriv_fun_arg_segmented(stacked_xs):
                xs = [stacked_xs[nbegins[i]:nends[i]] for i in range(self.number_of_in_ports)]
                # Use splat operator to expand list into several function arguments
                return self.derivative_function(*xs)

        # Produce one large message to linearize about by stacking all the input port marginals
        port_in_msg_to_linearize_about = self.stack_port_msgs(port_in_msgs_to_linearize_about, self.number_of_in_ports)

        if (self.jacobian_linearization is False):
            self.y_marginal, self.xy_ccov = \
                port_in_msg_to_linearize_about.unscented_transform(fun_arg_segmented,
                                                                   sigma_point_scheme=self.sigma_point_scheme,
                                                                   alpha=self.alpha)

            self._xy_ccovs = [self.xy_ccov[nbegins[i]:nends[i], :] for i in range(self.number_of_in_ports)]
            self._M_marg_matrix = [self._xy_ccovs[i].T @ inv(port_in_msgs_to_linearize_about[i].cov) for i in
                                   range(self.number_of_in_ports)]

            self._n_marg_offsets = [self._M_marg_matrix[i] @ port_in_msgs_to_linearize_about[i].mean for i in
                                    range(self.number_of_in_ports)]

            self._n_marg_offset = self.y_marginal.mean - sum(self._n_marg_offsets)

            self._cov_marg_errs = [
                self._M_marg_matrix[i] @ port_in_msgs_to_linearize_about[i].cov @ self._M_marg_matrix[i].T for i in
                range(self.number_of_in_ports)]

            self._cov_marg_err = self.y_marginal.cov - sum(self._cov_marg_errs)
        else:
            self._big_M = deriv_fun_arg_segmented(port_in_msg_to_linearize_about.mean)
            self._big_n = fun_arg_segmented(
                port_in_msg_to_linearize_about.mean) - self._big_M @ port_in_msg_to_linearize_about.mean
            mean = self._big_M @ port_in_msg_to_linearize_about.mean + self._big_n
            cov = self._big_M @ port_in_msg_to_linearize_about.cov @ self._big_M.T

            self.y_marginal = GaussianMeanCovMessage(mean, cov)

            self._M_marg_matrix = [self._big_M[:, nbegins[i]:nends[i]] for i in
                                   range(self.number_of_in_ports)]

            self._n_marg_offsets = [self._big_n[nbegins[i]:nends[i], :] for i in
                                    range(self.number_of_in_ports)]

            self._n_marg_offset = self._big_n  # self.y_marginal.mean - sum(self._n_marg_offsets)

            self._cov_marg_errs = [
                self._M_marg_matrix[i] @ port_in_msgs_to_linearize_about[i].cov @ self._M_marg_matrix[i].T for i in
                range(self.number_of_in_ports)]

            self._cov_marg_err = self.y_marginal.cov - sum(self._cov_marg_errs)

        # Check if this is the first time, the linearization is performed
        # or if linearization is not to be performed about the marginal
        if (self.n_offset is None) | \
                any([self.ports_out[i].in_msg is None for i in range(self.number_of_out_ports)]) | \
                (self.linearization_about_marginal is not True):
            self.M_matrix = [self._M_marg_matrix[i] for i in range(self.number_of_in_ports)]
            self.n_offsets = [self._n_marg_offsets[i] for i in range(self.number_of_in_ports)]
            self.n_offset = self._n_marg_offset
            self.cov_errs = self._cov_marg_errs
            self.cov_err = self._cov_marg_err

            port_out_msg = self.y_marginal

        # If EP mode is on, then transform the linearization parameters from the marginals to work
        # for the directed messages
        elif (self.expectation_propagation is True):
            # Gaussian division to obtain forward message
            # ToDo: Implement Gaussian Division for Tilde message?!
            if isinstance(self.ports_out[0].in_msg, GaussianTildeMessage):
                port_out_msg_in_msg = self.ports_out[0].in_msg.convert(target_type=GaussianMeanCovMessage,
                                                                       other_msg=self.ports_out[0].out_msg)
            elif isinstance(self.ports_out[0].in_msg, GaussianWeightedMeanInfoMessage):
                # ToDo: Probably not necessary
                port_out_msg_in_msg = self.ports_out[0].in_msg.convert(GaussianMeanCovMessage)
            elif isinstance(self.ports_out[0].in_msg, GaussianMeanCovMessage):
                port_out_msg_in_msg = self.ports_out[0].in_msg
            else:
                raise NotImplementedError(
                    'The incoming message is supposed to be either a GaussianTildeMessage, a '
                    'GaussianWeightedMeanInfoMessage or a GaussianMeanCovMessage!')

            port_out_msg = self.y_marginal / port_out_msg_in_msg

            # Moment matching-based update of the matrices
            # Formulae involving the parameters resulting from the linearization about the marginals
            self.M_matrix = [
                port_out_msg.cov @ inv(self.y_marginal.cov) @ self._M_marg_matrix[i] @ port_in_msgs_to_linearize_about[
                    i].cov @ inv(self.ports_in[i].in_msg.cov) for i in range(self.number_of_in_ports)]
            self.n_offsets = [
                self._M_marg_matrix[i] @ port_in_msgs_to_linearize_about[i].cov @ self.ports_in[i].out_msg.convert(
                    GaussianWeightedMeanInfoMessage).weighted_mean
                for i in range(self.number_of_in_ports)]
            self.n_offset = port_out_msg.cov @ inv(self.y_marginal.cov) @ sum(self.n_offsets) + \
                port_out_msg.cov @ (inv(self.y_marginal.cov) @ self._n_marg_offset
                                    - port_out_msg_in_msg.convert(GaussianWeightedMeanInfoMessage).weighted_mean)
            self.cov_errs = [self.M_matrix[i] @ self.ports_in[i].in_msg.cov @ self.M_matrix[i].T
                             for i in range(self.number_of_in_ports)]
            self.cov_err = port_out_msg.cov - sum(self.cov_errs)

            # Formulae that are simplified by inserting the closed-forms of statistical linearization-based M, n and V_E
            # self.M_matrix = [port_out_msg.cov @ inv(self.y_marginal.cov) @ self._xy_ccovs[i].T @ inv(
            #     self.ports_in[i].in_msg.cov) for i in range(self.number_of_in_ports)]
            # self.n_offsets = [self._xy_ccovs[i].T @ inv(self.ports_in[i].in_msg.cov) @ self.ports_in[i].in_msg.mean
            #                                  for i in range(self.number_of_in_ports)]
            # self.n_offset = port_out_msg.mean - port_out_msg.cov @ inv(self.y_marginal.cov) @ sum(self.n_offsets)
            # self.cov_errs = [self._xy_ccovs[i].T @ inv(self.ports_in[i].in_msg.cov) @ self._xy_ccovs[i]
            #                                  for i in range(self.number_of_in_ports)]
            # self.cov_err = port_out_msg.cov - port_out_msg.cov @ inv(self.y_marginal.cov) @ sum(self.cov_errs) @ \
            #                inv(self.y_marginal.cov) @ port_out_msg.cov

        else:
            self.M_matrix = [self._M_marg_matrix[i] for i in range(self.number_of_in_ports)]
            self.n_offsets = [self._n_marg_offsets[i] for i in range(self.number_of_in_ports)]
            self.n_offset = self._n_marg_offset
            self.cov_errs = self._cov_marg_errs
            self.cov_err = self._cov_marg_err

        # Update prior nodes, matrix nodes and do the message passing
        # Doing the message passing in forward direction is actually not necessary,
        # but it is performed here as a check of correctness
        for i in range(self.number_of_in_ports):
            self._M_node[i].matrix = self.M_matrix[i]
            self._in_node[i].update_prior(self.ports_in[i].in_msg)
            self._M_node[i].port_b.update(GaussianMeanCovMessage)

        for i in range(self.number_of_out_ports):
            if self.ports_out[i].in_msg is not None:
                self._out_node[i].update_prior(self.ports_out[i].in_msg)
            else:
                self._out_node[i].update_prior(
                    GaussianNonInformativeMessage(shape(self.n_offset)[0], direction=PortMessageDirection.Backward,
                                                  inf_approx=None))

        self._n_offset_prior.update_prior(GaussianMeanCovMessage(self.n_offset, self.cov_err))
        result = self._add_n_offset_node.ports_out[0].update(GaussianMeanCovMessage)

        # assert(allclose(result.mean, port_out_msg.mean))

        return result

    def stack_port_msgs(self, msgs, num_ports):
        assert (isinstance(msgs[0], GaussianMeanCovMessage))

        msgs_means = [col_vec(msgs[i].mean) for i in range(num_ports)]
        msgs_covs = [msgs[i].cov for i in range(num_ports)]
        mean = vstack(msgs_means)
        cov = block_diag(*msgs_covs)
        msg = GaussianMeanCovMessage(mean, cov)

        return msg

    def __repr__(self):
        if self.name is None:
            name = str(id(self))
        else:
            name = self.name

        return type(self).__name__ + "(" + name + ", Function:" + repr(self.function) + ")"

    def get_ports(self):
        return self.ports_in + self.ports_out


class ForgettingFactorNode(Node):
    def __init__(self, factor, name=None):
        super().__init__(name=name)

        self.factor = atleast_2d(asarray(factor))
        assert ndim(self.factor) == 2 and self.factor.shape == (1, 1)

        self.port_a = NodePort(self, self._calc_msg_a, NodePortType.InPort)
        self.port_b = NodePort(self, self._calc_msg_b, NodePortType.OutPort)

    def _calc_msg_a(self):
        msg_b = self.port_b.in_msg
        return self._forget_msg(msg_b)

    def _calc_msg_b(self):
        msg_a = self.port_a.in_msg
        return self._forget_msg(msg_a)

    def _forget_msg(self, msg):
        if type(msg) is GaussianMeanCovMessage:
            return GaussianMeanCovMessage(msg.mean, msg.cov / self.factor)
        elif type(msg) is GaussianWeightedMeanInfoMessage:
            return GaussianWeightedMeanInfoMessage(msg.weighted_mean * self.factor, msg.info * self.factor)
        else:
            return NotImplemented

    def get_ports(self):
        return [self.port_a, self.port_b]
