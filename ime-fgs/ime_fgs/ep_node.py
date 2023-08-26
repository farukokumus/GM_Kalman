# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
from ime_fgs.base import NodePort
from numpy import asarray, atleast_2d, linspace, inf, identity, transpose, kron
from numpy.random import normal, seed

import time
import matplotlib.pyplot as plt

from ime_fgs.basic_nodes import *
from ime_fgs.compound_nodes import *
from ime_fgs.messages import *
from ime_fgs.divergence_measures import moment_matched_mean_cov_of_doubly_truncated_gaussian

"""
Implementation of general EP-Node
"""


class EPNode(Node):
    """

    X   +--------+   Y
    --->| f(x,y) |--->
        +--------+

    X   +------+    +------+   Y
    --->| f(x) |    | f(y) |--->
        +------+    +------+

    f(x,y)  : Real node function, potentially not totally factorizable
    f(x)    : Part of (approximately) factorized node function
    f(y)    : Part of (approximately) factorized node function
    X       : R^(N)
    Y       : R^(M)

    """
    # ToDo: Implement EP-way of calculating output messages
    # ToDo: Implement initialization of approximate output messages

    def __init__(self, init_port_a_out_msg=None, init_port_b_out_msg=None, node_function_ab=None, node_function_a=None,
                 node_function_b=None, name=None):
        super().__init__(name=name)
        # init connections of node frame
        self.port_a = NodePort(self, self._calc_msg_a)
        self.port_b = NodePort(self, self._calc_msg_b)

        self._node_function_ab = node_function_ab
        self._node_function_a = node_function_a
        self._node_function_b = node_function_b

        self.port_a.out_msg = init_port_a_out_msg
        self.port_b.out_msg = init_port_b_out_msg

    @property
    def node_function_ab(self):
        return self._node_function_ab

    @node_function_ab.setter
    def node_function_ab(self, node_function_ab):
        self._node_function_ab = 0

    @property
    def node_function_a(self):
        return self._node_function_a

    @node_function_a.setter
    def node_function_a(self, node_function_a):
        self._node_function_a = node_function_a

    @property
    def node_function_b(self):
        return self._node_function_b

    @node_function_b.setter
    def node_function_b(self, node_function_b):
        self._node_function_b = node_function_b

    def init_msg_port_a(self, init_port_a_out_msg=None):
        # Init port a output by neglecting the marginal once
        if (init_port_a_out_msg is None) & self.port_a.connected:
            self.port_a.out_msg = self._node_function_a(self.port_b.in_msg)
    # Else simply use the argument provided
        else:
            self.port_a.out_msg = init_port_a_out_msg

    def init_msg_port_b(self, init_port_b_out_msg=None):
        # Init port b output by neglecting the marginal once
        if (init_port_b_out_msg is None) & self.port_b.connected:
            self.port_b.out_msg = self._node_function_b(self.port_a.in_msg)
    # Else simply use the argument provided
        else:
            self.port_b.out_msg = init_port_b_out_msg

    def _calc_msg_a(self):
        # Incoming msg equals cavity message as Gaussian division of the marginal by the approximate message simply
        # yields the incoming message
        if self.port_a.in_msg is not None:
            cavity_msg = self.port_a.in_msg.convert(GaussianWeightedMeanInfoMessage)

            # ToDo: Check that there is a difference in letting port b msg pass through, form marginal and then project
            # ToDo: instead of taking the exact function of the port b msg and then project the correct marginal
            marginal_msg = self.port_b.in_msg.convert(GaussianWeightedMeanInfoMessage)
            # marginal_msg = self.port_b.in_msg.convert( GaussianMeanCovMessage )
            # marginal_msg = self.node_function_a(marginal_msg)
            # marginal_msg = marginal_msg.convert( GaussianWeightedMeanInfoMessage )
            marginal_msg = marginal_msg.combine(cavity_msg)
            marginal_msg = marginal_msg.convert(GaussianMeanCovMessage)

            # Computes approximate marginal via node function
            approximate_marginal = self.node_function_a(marginal_msg)
            self.port_a.out_msg = approximate_marginal.convert(GaussianWeightedMeanInfoMessage) / cavity_msg

            return self.port_a.out_msg.convert(GaussianMeanCovMessage)
        else:
            self.init_msg_port_a()
            return self.port_a.out_msg.convert(GaussianMeanCovMessage)

    def _calc_msg_b(self):
        # Incoming msg equals cavity message as Gaussian division of the marginal by the approximate message simply
        # yields the incoming message
        if self.port_b.in_msg is not None:
            cavity_msg = self.port_b.in_msg.convert(GaussianWeightedMeanInfoMessage)

            # ToDo: Check that there is a difference in letting port a msg pass through, form marginal and then project
            # ToDo: instead of taking the exact function of the port a msg and then project the correct marginal
            marginal_msg = self.port_a.in_msg.convert(GaussianWeightedMeanInfoMessage)
            # marginal_msg = self.port_a.in_msg.convert( GaussianMeanCovMessage )
            # marginal_msg = self.node_function_b(marginal_msg)
            # marginal_msg = marginal_msg.convert( GaussianWeightedMeanInfoMessage )
            marginal_msg = marginal_msg.combine(cavity_msg)
            marginal_msg = marginal_msg.convert(GaussianMeanCovMessage)

            # Computes approximate marginal via node function
            approximate_marginal = self.node_function_b(marginal_msg)
            self.port_b.out_msg = approximate_marginal.convert(GaussianWeightedMeanInfoMessage) / cavity_msg

            return self.port_b.out_msg.convert(GaussianMeanCovMessage)
        else:
            self.init_msg_port_b()
            return self.port_b.out_msg.convert(GaussianMeanCovMessage)

    def __repr__(self):
        if self.name is None:
            name = str(id(self))
        else:
            name = self.name
        # return type(self).__name__ + "(" + name + ", Function f(x,y):" + repr(self.node_function_xy.tolist( ) ) \
        #                                         + ", Function f(x):" + repr(self.node_function_x.tolist( ) ) \
        #                                         + ", Function f(y):" + repr( self.node_function_y.tolist( ) )+ ")"
        return type(self).__name__ + "(" + name + ")"

    def get_ports(self):
        return [self.port_a, self.port_b]


class EPNodeTruncation(Node):

    def __init__(self, init_port_a_out_msg, init_port_b_out_msg, hyperplane_normal, lower_bounds, upper_bounds,
                 name=None):
        """
        :param init_port_a_out_msg: GaussianMeanCovMessage as initial guess for port a out message
        :param init_port_b_out_msg: GaussianMeanCovMessage as initial guess for port b out message
        :param init_port_b_out_msg: GaussianMeanCovMessage as initial guess for port b out message
        """
        super().__init__(name=name)

        self.port_a = NodePort(self, self._calc_msg_a)
        self.port_b = NodePort(self, self._calc_msg_b)
        self.port_a.out_msg = init_port_a_out_msg
        self.port_b.out_msg = init_port_b_out_msg

        self._lower_bounds = atleast_2d(asarray(lower_bounds))
        self._upper_bounds = atleast_2d(asarray(upper_bounds))
        assert ndim(self._lower_bounds) == ndim(self._upper_bounds)
        self._hyperplane_normal = atleast_2d(asarray(hyperplane_normal))

        def _node_function(self, cavity_msg=None, in_msg=None):
            moment_matched_mean, moment_matched_cov = \
                moment_matched_mean_cov_of_doubly_truncated_gaussian(in_msg.mean, in_msg.cov, self._hyperplane_normal,
                                                                     self._upper_bounds, self._lower_bounds)

            return GaussianMeanCovMessage(moment_matched_mean, moment_matched_cov)

        self.EPNode = EPNode(init_port_a_out_msg=init_port_a_out_msg,
                             init_port_b_out_msg=init_port_b_out_msg,
                             node_function_a=_node_function,
                             node_function_b=_node_function)

    def get_ports(self):
        return [self.port_a, self.port_b]

    def _calc_msg_a(self):
        self.port_a.out_msg = self.EPNode._calc_msg_a
        return self.port_a.out_msg

    def _calc_msg_b(self):
        self.port_b.out_msg = self.EPNode._calc_msg_b
        return self.port_b.out_msg


class EPNodeUnscentedTransformation(EPNode):
    def __init__(self,
                 function,
                 node_function_ab=None, node_function_a=None, node_function_b=None,
                 init_port_a_out_msg=GaussianMeanCovMessage([[0]], [[1e9]]),
                 init_port_b_out_msg=GaussianMeanCovMessage([[0]], [[1e9]]),
                 inverse_func=None, method=1, alpha=0.999, scaled_sigma_points=True,
                 name=None):
        """
        :param init_port_a_out_msg: GaussianMeanCovMessage as initial guess for port a out message
        :param init_port_b_out_msg: GaussianMeanCovMessage as initial guess for port b out message

          a +----------+ b
        --->| function |--->
            +----------+
        """
        super().__init__(name=name)

        self.function = function
        self.inverse_function = inverse_func
        self.alpha = alpha
        self.scale = scaled_sigma_points
        self.method = method

        self.sigma_points = None
        self.weights = None
        self.sigma_points_transformed = None

        self.port_a_marginal_fwd = None
        self.port_a_marginal_bwd = None
        self.port_b_marginal_fwd = None
        self.port_b_marginal_bwd = None
        self.port_a_cov_fwd = None
        self.port_b_cov_fwd = None
        self.port_ab_cov_fwd = None

        self.node_function_a = self.unscented_backward
        self.node_function_b = self.unscented_forward

        # self.EPNode = EPNode( init_port_a_out_msg=init_port_a_out_msg,
        #                       init_port_b_out_msg=init_port_b_out_msg,
        #                       node_function_a=_node_function_a,
        #                       node_function_b=_node_function_b )

        self.port_a = NodePort(self, self._calc_msg_a)
        self.port_b = NodePort(self, self._calc_msg_b)
        self.port_a.out_msg = init_port_a_out_msg
        self.port_b.out_msg = init_port_b_out_msg

    def unscented_backward(self, port_b_in_msg):
        if not (callable(self.inverse_function)):

            if self.sigma_points is None or self.weights is None or self.sigma_points_transformed is None:
                raise RuntimeError("Can't calculate reverse")

            # use the unscented rts smoother
            # see Simo Särkkä, Unscented Rauch--Tung--Striebel Smoother

            self.port_a_marginal_fwd = self.port_a.marginal().convert(GaussianMeanCovMessage)
            self.port_b_marginal_bwd = self.port_b.marginal().convert(GaussianMeanCovMessage)

            D_a = self.port_ab_cov_fwd * inv(self.port_b_cov_fwd)
            mean = self.port_a_marginal_fwd.mean + D_a * (self.port_b_marginal_bwd.mean - self.port_b_marginal_fwd.mean)
            cov = self.port_a_marginal_fwd.cov + D_a * (self.port_b_marginal_bwd.cov - self.port_b_cov_fwd) * D_a.T

            self.port_a_marginal_bwd = GaussianMeanCovMessage(mean, cov)

            return self.port_a_marginal_bwd

        else:
            msg, _, _, _ = self.port_a.in_msg.unscented_transform(self.function,
                                                                  sigma_point_scheme=self.method,
                                                                  a=self.alpha,
                                                                  scale_arg=self.scale)
            return msg

    def unscented_forward(self, port_a_in_msg):
        # forward direction
        # If the backward message is already available, a pass through the filter has already been made. We now want
        # to pass the marginals instead of only the forward messages.

        self.port_a_marginal_fwd = self.port_a.marginal(target_type=GaussianMeanCovMessage)

        marg, self.sigma_points, self.weights, self.sigma_points_transformed \
            = self.port_a_marginal_fwd.unscented_transform(self.function,
                                                           sigma_point_scheme=self.method,
                                                           a=self.alpha,
                                                           scale_arg=self.scale)

        # We have now computed the posterior marginal and need to divide this by the backward message for the
        # marginal to obtain the new forward message.
        self.port_b_marginal_fwd = marg

        self.port_ab_cov_fwd = sum([weight * (point - self.port_a_marginal_fwd.mean)
                                    * (point_t - self.port_b_marginal_fwd.mean).T
                                    for weight, point, point_t
                                    in zip(self.weights, self.sigma_points, self.sigma_points_transformed)])

        self.port_b_cov_fwd = sum([weight * (point_t - self.port_b_marginal_fwd.mean)
                                   * (point_t - self.port_b_marginal_fwd.mean).T
                                   for weight, point, point_t
                                   in zip(self.weights, self.sigma_points, self.sigma_points_transformed)])

        return self.port_b_marginal_fwd

    def get_ports(self):
        return [self.port_a, self.port_b]

    # def _calc_msg_a(self):
    #     self.port_a.out_msg = self.EPNode._calc_msg_a()
    #     return self.port_a.out_msg
    #
    # def _calc_msg_b(self):
    #     self.port_b.out_msg = self.EPNode._calc_msg_b()
    #     return self.port_b.out_msg

    # class UnscentedEPNode( Node ):
    #     """
    #       a +----------+ b
    #     --->| function |--->
    #         +----------+
    #     """
    #
    #     def __init__(self, function, name=None, inverse_func=None, method=1, alpha=0.999, scaled_sigma_points=True, \
    #                  port_a_out_msg_init=GaussianMeanCovMessage( 0, 1e9 ), \
    #                  port_b_in_msg_init=GaussianMeanCovMessage( 0, 1e9 )):
    #         super().__init__( name=name )
    #
    #         self.port_a = NodePort( self, self._calc_msg_a )
    #         self.port_b = NodePort( self, self._calc_msg_b )
    #         self.function = function
    #         self.inverse_function = inverse_func
    #         self.alpha = alpha
    #         self.scale = scaled_sigma_points
    #         self.method = method
    #
    #         self.sigma_points = None
    #         self.weights = None
    #         self.sigma_points_transformed = None
    #
    #         # Init backward messages (default is unknown)
    #         self.port_a.out_msg = port_a_out_msg_init
    #         # self.port_b.in_msg  = port_b_in_msg_init
    #         self.port_a_out_msg_init = port_a_out_msg_init
    #         self.port_b_in_msg_init = port_b_in_msg_init
    #
    #         self.msg_a_marginal = GaussianMeanCovMessage( 0, 1e9 )
    #         self.msg_b_marginal = GaussianMeanCovMessage( 0, 1e9 )
    #
    #     def _calc_msg_a(self):
    #         if not (callable( self.inverse_function )):
    #             if self.sigma_points is None or self.weights is None or self.sigma_points_transformed is None:
    #                 raise RuntimeError( "Can't calculate reverse" )
    #
    #             # todo add assert for sigma points and weights dimensions
    #
    #             # use the unscented rts smoother
    #             # see Simo Särkkä, Unscented Rauch--Tung--Striebel Smoother
    #
    #             a_marginal_message = self.port_a.marginal().convert( GaussianMeanCovMessage )
    #             b_marginal_message = self.port_b.marginal().convert( GaussianMeanCovMessage )
    #             b_margin = self.port_b.marginal().convert( GaussianMeanCovMessage )
    #
    #             b_out_message = self.msg_b_marginal / self.port_b.in_msg
    #
    #             # The sigma points originate from previous forward pass
    #             # If the forward pass has been performed based on marginals, this is probably not working!
    #             C_b = sum( [weight * (point - a_marginal_message.mean) * (point_t - b_marginal_message.mean).T
    #                         for weight, point, point_t
    #                         in zip( self.weights, self.sigma_points, self.sigma_points_transformed )] )
    #
    #             # compute port b marginals
    #             D_a = C_b * inv( b_marginal_message.cov )
    #             mean = a_marginal_message.mean + D_a * (b_marginal_message.mean - b_out_message.mean)
    #
    #             b_margin_out_cov_diff = (b_marginal_message.cov - b_out_message.cov)
    #
    #             cov = a_marginal_message.cov + D_a * (b_marginal_message.cov - b_out_message.cov) * D_a.T
    #
    #             self.msg_a_marginal = GaussianMeanCovMessage( mean, cov )
    #             a_out_messsage = self.msg_a_marginal / self.port_a.in_msg
    #
    #             return a_out_messsage
    #
    #         else:
    #             msg, _, _, _ = self.port_a.in_msg.unscented_transform( self.function,
    #                                                                    method=self.method,
    #                                                                    a=self.alpha,
    #                                                                    scale_arg=self.scale )
    #             return msg
    #
    #     def _calc_msg_b(self):
    #         # forward direction
    #         # If the backward message is already available, a pass through the filter has already been made. We now
    #         # want to pass the marginals instead of only the forward messages.
    #         # Probably wrong
    #         marg, self.sigma_points, self.weights, self.sigma_points_transformed \
    #             = self.port_a.marginal( target_type=GaussianMeanCovMessage ).unscented_transform(self.function,
    #                                                                                              method=self.method,
    #                                                                                              a=self.alpha,
    #                                                                                              scale_arg=self.scale)
    #         # We have now computed the posterior marginal and need to divide this by the backward message for the
    #         # marginal to obtain the new forward message.
    #         self.msg_b_marginal = marg
    #
    #         # Since during init the port is not connected, provide if-clause here.
    #         if self.port_b.in_msg is None:
    #             msg = marg / self.port_b_in_msg_init
    #         else:
    #             msg = marg / self.port_b.in_msg
    #
    #         return msg
    #
    #     def __repr__(self):
    #         if self.name is None:
    #             name = str( id( self ) )
    #         else:
    #             name = self.name
    #
    #         return type( self ).__name__ + "(" + name + ", Function:" + repr( self.function ) + ")"
    #
    #     def get_ports(self):
    #         return [self.port_a, self.port_b]

########################################################################################################################
# Test
########################################################################################################################


if __name__ == '__main__':

    def node_function(msg=None, hyperplane_normal=1, upper_bounds=1, lower_bounds=-1):

        moment_matched_mean, moment_matched_cov = \
            moment_matched_mean_cov_of_doubly_truncated_gaussian(msg.mean, msg.cov, hyperplane_normal,
                                                                 upper_bounds, lower_bounds)

        return GaussianMeanCovMessage(moment_matched_mean, moment_matched_cov)

    EPNodeTest1 = EPNode(init_port_b_out_msg=GaussianMeanCovMessage(1, 1), node_function_a=node_function,
                         node_function_b=node_function, name='trunc')

    EPNodeTest2 = EPNode(init_port_a_out_msg=GaussianMeanCovMessage(1, 1), node_function_a=node_function,
                         node_function_b=node_function, name='trunc')

    # EPNodeTest = EPNodeTruncation(GaussianMeanCovMessage(1, 2), GaussianMeanCovMessage(3, 4), 1, -1, 1)
    Prior_a = PriorNode(GaussianMeanCovMessage(5, 1e0), name='a')
    Prior_b = PriorNode(GaussianMeanCovMessage(0, 1e0), name='b')

    Prior_a.port_a.connect(EPNodeTest1.port_a)
    EPNodeTest1.port_b.connect(EPNodeTest2.port_a)
    EPNodeTest2.port_b.connect(Prior_b.port_a)

    meana = np.zeros([20])
    meanb = np.zeros([20])

    EPNodeTest1.init_msg_port_a()
    EPNodeTest1.init_msg_port_b()
    EPNodeTest2.init_msg_port_a()
    EPNodeTest2.init_msg_port_b()

    for ii in range(20):
        # EPNodeTest1.port_a.update( )
        EPNodeTest1.port_b.update()
        meanb[ii] = EPNodeTest1.port_b.out_msg.mean
        EPNodeTest2.port_a.update()
        # EPNodeTest2.port_b.update( )
        meana[ii] = EPNodeTest2.port_a.out_msg.mean

    moment_matched_mean, moment_matched_cov = \
        moment_matched_mean_cov_of_doubly_truncated_gaussian(2, 1e0, 1,
                                                             1, -1)
    print(moment_matched_mean)

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.plot(meana)
    ax.plot(meanb)
    plt.show()
