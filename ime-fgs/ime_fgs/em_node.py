# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
from ime_fgs.base import NodePort
from numpy import asarray, atleast_2d, linspace, inf, identity, transpose, kron, trace, shape
from numpy.random import normal, seed

import time
import matplotlib.pyplot as plt

from ime_fgs.basic_nodes import *
from ime_fgs.compound_nodes import *
from ime_fgs.messages import *

import numpy as np

"""
implementation of Matrix-EM-Node based on Table II (Loeliger Expectation Max., 2015)

inputs:

X on port_a is column vector
Theta on port_theta is column vector
S on port_b is scalar --> S = Theta^T * X

initial_state: GaussianWeightedMeanInfoMessage

"""


class EMNodeScalar(Node):
    def __init__(self, initial_state):
        """

        :param initial_state: GaussianWeightedMeanInfoMessage as initial guess for factor
        """
        # auxiliary variables
        super().__init__()
        self.input_in = GaussianWeightedMeanInfoMessage(0, 0)
        self.input_out = GaussianWeightedMeanInfoMessage(0, 0)
        self.wtheta = 0
        self.wmtheta = 0

        self.port_a = NodePort(self, self.__calc_msg_a__)
        self.port_b = NodePort(self, self.__calc_msg_b__)
        self.port_theta = NodePort(self, self.__calc_msg_theta__)

        self.thetEM_node = PriorNode(initial_state)
        self.__factor__ = self.thetEM_node.get_prior().convert(GaussianMeanCovMessage).mean

        self.port_theta.connect(self.thetEM_node.port_a)

    # right to left
    def __calc_msg_a__(self):
        return self.port_b.in_msg.convert(GaussianWeightedMeanInfoMessage).multiply_deterministic(self.__factor__,
                                                                                                  inverse=True)

    # left to right
    def __calc_msg_b__(self):
        return self.port_a.in_msg.multiply_deterministic(self.__factor__)

    def __calc_msg_theta__(self):
        raise NotImplementedError('nothing here yet')

    # update step of em-algorithm
    def update_theta(self):
        input_in = self.port_a.in_msg.convert(GaussianWeightedMeanInfoMessage)
        input_out = self.port_a.out_msg.convert(GaussianWeightedMeanInfoMessage)
        input_marginal = input_in.combine(input_out).convert(GaussianMeanCovMessage)

        wtheta = (input_marginal.cov + (input_marginal.mean * input_marginal.mean.transpose())) / (
            self.port_b.in_msg.convert(GaussianMeanCovMessage).cov)
        wmtheta = (input_marginal.mean * self.port_b.in_msg.convert(GaussianMeanCovMessage).mean) / (
            self.port_b.in_msg.convert(GaussianMeanCovMessage).cov)
        # update values for theta
        self.thetEM_node.update_prior(GaussianWeightedMeanInfoMessage(wmtheta, wtheta))
        self.__factor__ = self.thetEM_node.get_prior().convert(GaussianMeanCovMessage).mean

    def get_theta(self):
        return self.__factor__

    def get_ports(self):
        return [self.port_a, self.port_b]


class EmMatrixNodeAffineParametrized(Node):
    """     | Theta
            |
            | A(Theta)
            v
    X     +---+    Y
    ----->| x |------->
          +---+

    A(Theta)   : R^(MxN)
    X       : R^(N)
    Y       : R^(M)
    V_z     : uncertainty variance

    Matrix A (depending on Theta) times column vector X leading to column vector Y
    A = A_0 + Theta_1*A_1 + ... + Theta_n*A_n
    """

    def __init__(self, A_0, A_theta=None, theta=None, matrix_parameterization='full', name=None):
        """
        initialize EM-Node

        :param A_0:          A_0   : R^(MxN) Matrix of known factors of A
        :param A_theta       A_theta: Concatenation of unknown factors
        :param theta         theta (row vector)

        * port_a:       incoming port - left side of matrix node
        * port_b:       outgoing port - right side of matrix node
        * port_theta:   use this to connect equality-cycle - use fake-prior to prevent exception due to port-connection
        check
        """
        super().__init__(name=name)

        self.n = np.shape(A_0)[1]

        if theta is None:
            theta = A_0.reshape(np.size(A_0), 1)

        self.rows = shape(A_0)[0]
        self.cols = shape(A_0)[1]
        self.n_theta = np.shape(theta)[0]
        self.ident_n = np.identity(self.n)
        self.ident_theta = np.identity(self.n_theta)

        self.matrix_parameterization = matrix_parameterization

        self.A_0 = atleast_2d(A_0)
        self.A_theta = atleast_2d(A_theta)
        self.theta = atleast_2d(theta)
        self._theta_up_msg = GaussianWeightedMeanInfoMessage(np.zeros([self.n_theta, 1]),
                                                             np.zeros([self.n_theta, self.n_theta]))
        self._theta_marg_msg = GaussianWeightedMeanInfoMessage(np.zeros([self.n_theta, 1]), np.identity(self.n_theta))

        self._matrix = 0.0

        # init connections of node frame
        self.port_a = NodePort(self, self._calc_msg_a)
        self.port_b = NodePort(self, self._calc_msg_b)
        self.port_theta = NodePort(self, self._calc_msg_theta)
        self.update_matrix()

        # set auxiliaries and helpers
        self.aux_marginal_x_GMC = GaussianMeanCovMessage([[0]], [[0]])

    def update_matrix(self):
        ident_theta = self.ident_n

        if self.matrix_parameterization == 'affine':
            for i in range(0, self.n_theta):
                ident_theta = np.vstack([ident_theta, kron(self.ident_n, self.theta[i])])

            self._matrix = np.hstack([self.A_0, self.A_theta]) @ ident_theta

        elif self.matrix_parameterization == 'full':
            self._matrix = self.theta.reshape(self.rows, self.cols)

    # calculate out_msg port_a
    def _calc_msg_a(self):
        # from right through the 'matrix_node'
        return self.port_b.in_msg.convert(GaussianWeightedMeanInfoMessage).multiply_deterministic(self._matrix,
                                                                                                  inverse=True)

    # calculate out_msg port_b
    def _calc_msg_b(self):
        # from port_a through the 'matrix_node'
        return self.port_a.in_msg.multiply_deterministic(self._matrix)

    def _calc_msg_theta(self):
        # 1. check for marginal @ port_a and incomming msg @ port_b
        if self.port_a.in_msg is None:
            raise Exception("No incoming message on port_a!")
        elif self.port_a.out_msg is None:
            raise Exception("No outgoing message on port_a!")
        elif self.port_b.in_msg is None:
            raise Exception("No incoming message on port_b!")
        elif self.port_b.out_msg is None:
            raise Exception("No outgoing message on port_b!")
        else:

            if self.matrix_parameterization == 'affine':
                # 2. calculate new weighted_mean & info
                # 2.1 calculate auxiliary variables
                # calculate marginal V_x; m_x from left side
                self.aux_marginal_x_GMC = self.port_a.in_msg.convert(GaussianWeightedMeanInfoMessage).combine(
                    self.port_a.out_msg.convert(GaussianWeightedMeanInfoMessage)).convert(GaussianMeanCovMessage)
                # Wy_my from right side
                self.y_msg_GWM = self.port_b.in_msg.convert(GaussianWeightedMeanInfoMessage)

                # 2.2. Info_theta: W_theta = tr(A_i^T*W_y*A_j*V_x)+(kron(I,m_x^T)+A_theta^T+W_y*A_theta*(kron(I,m_x))
                # tmp = tr(A_i^T*W_y*A_j*V_x)
                tmp = np.empty([self.n_theta, self.n_theta])
                # tr_wm = tr(A_i^T*W_y*A_0*V_x) (used for Wm_theta)
                tr_wm = np.empty([self.n_theta, 1])
                for i in range(0, self.n_theta):
                    tr_wm[i] = trace(transpose(self.A_theta[:, self.n * i:self.n * i + self.n]) @ self.y_msg_GWM.info @
                                     self.A_0 @ self.aux_marginal_x_GMC.cov)

                    for j in range(0, self.n_theta):
                        tmp[i, j] = trace(transpose(
                            self.A_theta[:, self.n * i:self.n * i + self.n]) @ self.y_msg_GWM.info @
                            self.A_theta[:, self.n * j:self.n * j + self.n] @ self.aux_marginal_x_GMC.cov)

                self._theta_up_msg.info = tmp + (kron(self.ident_theta, transpose(self.aux_marginal_x_GMC.mean)) @
                                                 transpose(self.A_theta) @ self.y_msg_GWM.info @ self.A_theta @
                                                 kron(self.ident_theta, self.aux_marginal_x_GMC.mean))

                # 2.3 Weighted Mean: W_theta_m_theta = (kron(I,m_x^T)*A_theta^T*W_ym_y  - tr(A_i^T*W_y*A_0*V_x)
                #                                      - (kron(I, m_x^T)*A_theta^T*W_y*A_0*m_x
                wm = kron(self.ident_theta, transpose(self.aux_marginal_x_GMC.mean)) @ transpose(
                    self.A_theta) @ self.y_msg_GWM.weighted_mean
                self._theta_up_msg.weighted_mean = \
                    wm - tr_wm - (kron(self.ident_theta, transpose(self.aux_marginal_x_GMC.mean)) @
                                  transpose(self.A_theta) @ self.y_msg_GWM.info @ self.A_0 @
                                  self.aux_marginal_x_GMC.mean)

            elif self.matrix_parameterization == 'full':
                if self.port_b.in_msg.is_non_informative():
                    self._theta_up_msg = GaussianWeightedMeanInfoMessage.non_informative(self.n_theta)
                else:
                    # 2. calculate new weighted_mean & info
                    self.aux_marginal = self.port_a.in_msg.convert(GaussianWeightedMeanInfoMessage).combine(
                        self.port_a.out_msg)
                    self.aux_marginal = self.aux_marginal.convert(GaussianMeanCovMessage)
                    self._theta_up_msg.info = kron(self.port_b.in_msg.convert(GaussianWeightedMeanInfoMessage).info, (
                        self.aux_marginal.cov + (self.aux_marginal.mean @ transpose(self.aux_marginal.mean))))
                    self._theta_up_msg.weighted_mean = \
                        kron(self.port_b.in_msg.convert(GaussianWeightedMeanInfoMessage).info, self.ident_n) @ \
                        transpose((self.aux_marginal.mean @
                                   transpose(self.port_b.in_msg.convert(GaussianMeanCovMessage).mean)).flatten('F'))

                    self._theta_up_msg.weighted_mean = np.atleast_2d(self._theta_up_msg.weighted_mean).T

        # return as out_msg for port_theta --> for possible equality-cycle
        return self._theta_up_msg

    # calculate new A(Theta) matrix
    def update_theta(self):
        self.port_theta.out_msg = self.port_theta.out_msg.convert(GaussianWeightedMeanInfoMessage)
        self.theta = self.port_theta.marginal().convert(GaussianMeanCovMessage).mean
        self.update_matrix()

    def get_matrix(self):
        return self._matrix

    def __repr__(self):
        if self.name is None:
            name = str(id(self))
        else:
            name = self.name
        return type(self).__name__ + "(" + name + ", Matrix:" + repr(self._matrix.tolist()) + ")"

    def get_ports(self):
        return [self.port_a, self.port_b]


class NUVPrior(Node):
    """
                            +---+
                            +   + mean_offset
                            +---+
                 | Theta      |
                 |            |
                 | A(Theta)   |
      Nx         v            v
    +---+   X  +---+    Y   +---+    Z
    +   +----->| x |------->+ + +------->
    +---+      +---+        +---+

    Nx          : Normal, zero mean, identity variance (NZMIV) prior
    mean_offset : Mean offset, to allow for non-zero mean priors
    A(Theta)    : R^(MxN)
    X           : R^(N)
    Y           : R^(M)
    V_z         : uncertainty variance

    Matrix A (depending on Theta) times column vector X leading to column vector Y
    A = A_0 + Theta_1*A_1 + ... + Theta_n*A_n
    """

    def __init__(self, A_0, A_theta=None, theta=None, matrix_parameterization='full', mean_offset=None,
                 non_informative_prior=False, name=None):
        """
        initialize EM-Node

        :param A_0           R^(MxN) matrix of known factors of A
        :param A_theta       R^(Mx(N*n_theta)) matrix with concatenated gains of of unknown factors
        :param theta         theta (row vector)

        * port_a:       incoming port - left side of matrix node
        * port_b:       outgoing port - right side of matrix node
        * port_theta:   use this to connect equality-cycle - use fake-prior to prevent exception due to port-connection

        The structure of the A matrix to be estimated is as follows:

        A = A_0 + theta_1 * A_1 + ... + theta_n * A_n
          = [A_0 A_1 A_2 ... A_n] * [ I, kron(I, theta) ]'
          = [A_0 A_theta] * [ I, kron(I, theta) ]'
        """
        super().__init__(name=name)

        self.theta = theta
        self.A_0 = A_0
        self.A_theta = A_theta

        self.n = shape(self.A_0)[1]
        self.rows = shape(self.A_0)[0]
        self.cols = shape(self.A_0)[1]

        # If no structure is provided assume that all matrix entries are to be estimated.
        if self.A_theta is None:
            self.n_theta = self.rows * self.cols

            # Construct structure matrix capturing every single matrix entry with a factor of 1.
            # Thetas will be interpreted to correspond to the matrix entries reading row-wise from left to right.
            # Example:
            # A_0 is 2x2
            # A_theta = array([[1., 0., 0., 1., 0., 0., 0., 0.],
            #                  [0., 0., 0., 0., 1., 0., 0., 1.]])
            # that means that
            # A = A_0 + [[theta_1, theta_2], [theta_3, theta_4]]
            # temp_vec = list()
            # for ii in range(self.rows*self.cols):
            #     temp_vec.append(np.zeros([self.rows*self.cols, 1]))
            #     temp_vec[ii][ii] = 1.0
            #     temp_vec[ii] = temp_vec[ii].reshape(self.rows, self.cols)
            #
            # self.A_theta = np.concatenate( temp_vec, axis=1 )
            self.matrix_parameterization = 'full'

        # Expect a full matrix to be estimated, if no structure is provided and no initial theta is given
        if (self.A_theta is None) and (self.theta is None):
            # Caution: if theta is initialized with zeros, no estimation will happen
            self.theta = np.ones(shape(self.A_0.reshape(np.size(A_0), 1)))
            self.n_theta = shape(self.theta)[0]
            self.matrix_parameterization = 'full'

        # Expect a full matrix to be estimated, if no structure is provided
        elif (self.A_theta is not None) and (self.theta is None):
            self.n_theta = int(shape(self.A_theta)[1] / self.cols)
            # Caution: if theta is initialized with zeros, no estimation will happen
            self.theta = np.ones((self.n_theta, 1))
            self.matrix_parameterization = 'affine'

        else:
            self.matrix_parameterization = 'affine'
            self.n_theta = shape(self.theta)[0]
            assert self.n_theta == shape(self.A_theta)[
                1] / self.cols, "The dimension of the initial theta doesn't match the structure matrix A_theta!"

        # init connections of node frame
        self.port_b = NodePort(self, self._calc_msg_b)

        self.non_informative_prior = non_informative_prior
        self.port_theta = NodePort(self, self._calc_msg_theta)

        self.Nx_prior = PriorNode(GaussianMeanCovMessage(np.zeros([self.rows, 1]),
                                                         np.identity(self.rows)))

        self.signal_out_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))
        self.theta_out_node = PriorNode(GaussianWeightedMeanInfoMessage(np.zeros([self.n_theta, 1]),
                                                                        np.zeros([self.n_theta, self.n_theta])))

        self.EMNode = EmMatrixNodeAffineParametrized(A_0=A_0, A_theta=self.A_theta, theta=self.theta,
                                                     matrix_parameterization=self.matrix_parameterization)

        self._mean_offset = col_vec(mean_offset)
        self.mean_offset_prior = PriorNode(GaussianMeanCovMessage(self._mean_offset,
                                                                  np.zeros([self.rows, self.rows])))

        self.add_mean_node = AdditionNode()

        self.Nx_prior.port_a.connect(self.EMNode.port_a)
        self.EMNode.port_b.connect(self.add_mean_node.port_a)
        self.EMNode.port_theta.connect(self.theta_out_node.port_a)
        self.mean_offset_prior.port_a.connect(self.add_mean_node.port_b)
        self.add_mean_node.port_c.connect(self.signal_out_node.port_a)

        # Compute forward message once, s.t. the node can directly function as a normal prior
        self.port_b.update()

    # update offset
    def update_prior(self, msg, target_type=None):
        self.mean_offset_prior = msg
        self.port_b.target_type = target_type
        self.port_b.update()

    # calculate new A(Theta) matrix
    def _calc_msg_theta(self):
        if isinstance(self.port_b.in_msg, GaussianTildeMessage):
            self.signal_out_node.update_prior(
                self.port_b.in_msg.convert(target_type=GaussianMeanCovMessage, other_msg=self.port_b.out_msg))
        else:
            self.signal_out_node.update_prior(self.port_b.in_msg.convert(target_type=GaussianMeanCovMessage))
        if self.non_informative_prior is False:
            self.theta_out_node.update_prior(self.port_theta.in_msg, target_type=GaussianWeightedMeanInfoMessage)
        self.add_mean_node.port_a.update(target_type=GaussianWeightedMeanInfoMessage)
        self.EMNode.port_a.update(target_type=GaussianWeightedMeanInfoMessage)
        self.EMNode.port_theta.update()
        self.EMNode.update_theta()
        self.theta = self.EMNode.theta
        return self.EMNode.port_theta.update()

    def update_theta(self):
        if self.non_informative_prior is False:
            self.theta_out_node.update_prior(self.port_theta.in_msg, target_type=GaussianWeightedMeanInfoMessage)
        self.EMNode.update_theta()
        self.theta = self.EMNode.theta

    def get_matrix(self):
        return self.EMNode._matrix

    def get_theta(self):
        return self.theta

    def get_mean_offset(self):
        return self._mean_offset

    def _calc_msg_b(self):
        self.EMNode.port_b.update(GaussianMeanCovMessage)
        return self.add_mean_node.port_c.update(GaussianMeanCovMessage)

    def __repr__(self):
        if self.name is None:
            name = str(id(self))
        else:
            name = self.name
        return type(self).__name__ + "(" + name + ", Matrix:" + repr(self.get_matrix().tolist()) + ")"

    def get_ports(self):
        return [self.port_b]


########################################################################################################################
# Test
########################################################################################################################


"""
implementation of State Space Model (SSM) with matrices
"""


class EMMatrixSlice(Node):
    def __init__(self, A, B, C, name=None):
        super().__init__(name=name)
        # init external ports of this slice
        self.port_state_in = NodePort(self, self.calc_msg_state_in)
        self.port_state_out = NodePort(self, self.calc_msg_state_out)
        self.port_process_noise = NodePort(self, self.calc_msg_process_noise)
        self.port_input = NodePort(self, self.calc_msg_input)
        self.port_meas = NodePort(self, self.calc_msg_meas)

        # init PriorNodes as interfaces
        self.state_in_node = PriorNode(GaussianMeanCovMessage(0, inf))
        self.state_out_node = PriorNode(GaussianMeanCovMessage(0, inf))
        self.process_noise_node = PriorNode(GaussianMeanCovMessage(0, inf))
        self.input_node = PriorNode(GaussianMeanCovMessage(0, inf))
        self.meas_node = PriorNode(GaussianMeanCovMessage(0, inf))

        # init remaining SSM-nodes
        # 1. matrices
        self.A_node = EmMatrixNode(A)
        self.fake_prior = PriorNode(GaussianMeanCovMessage(0, inf))
        self.B_node = MatrixNode(B)
        self.C_node = MatrixNode(C)
        # 2. connectors
        self.add_process_noise_node = AdditionNode()
        self.add_input_node = AdditionNode()
        self.equality_node = EqualityNode()

        # connect the model starting from state_in
        self.state_in_node.port_a.connect(self.A_node.port_a)
        self.A_node.port_theta.connect(self.fake_prior.port_a)
        self.A_node.port_b.connect(self.add_process_noise_node.port_a)
        self.process_noise_node.port_a.connect(self.add_process_noise_node.port_b)
        self.add_process_noise_node.port_c.connect(self.add_input_node.port_a)
        self.input_node.port_a.connect(self.B_node.port_a)
        self.B_node.port_b.connect(self.add_input_node.port_b)
        self.add_input_node.port_c.connect(self.equality_node.ports[0])
        self.meas_node.port_a.connect(self.C_node.port_b)
        self.C_node.port_a.connect(self.equality_node.ports[1])
        self.equality_node.ports[2].connect(self.state_out_node.port_a)

    def calc_msg_state_in(self):
        # 1. update all the priors
        # self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.process_noise_node.update_prior(self.port_process_noise.in_msg)
        self.input_node.update_prior(self.port_input.in_msg)
        self.meas_node.update_prior(self.port_meas.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        # 2. reverse through the branches to get to state_in
        self.C_node.port_b.update()
        self.equality_node.ports[0].update()
        self.B_node.port_b.update()
        self.add_input_node.port_a.update()
        self.add_process_noise_node.port_a.update()
        return self.A_node.port_a.update()

    def calc_msg_state_out(self):
        # 1. update all the priors
        print(1.)
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        print(2.)
        self.process_noise_node.update_prior(self.port_process_noise.in_msg)
        print(3.)
        self.input_node.update_prior(self.port_input.in_msg)
        print(4.)
        self.meas_node.update_prior(self.port_meas.in_msg, GaussianWeightedMeanInfoMessage)
        # print(self.meas_node.port_a.out_msg)
        # self.state_out_node.update_prior(self.port_state_out.in_msg)
        # 2. forward through the branches to get to state_out
        print(5.)
        self.A_node.port_b.update()
        print(6.)
        self.add_process_noise_node.port_c.update()
        print(7.)
        self.B_node.port_b.update()
        print(8.)
        self.add_input_node.port_c.update(GaussianWeightedMeanInfoMessage)
        print(9.)
        print(self.C_node.port_b.in_msg)
        self.C_node.port_a.update(GaussianWeightedMeanInfoMessage)

        print(np.linalg.matrix_rank(self.C_node.port_b.out_msg.cov))
        print(self.C_node.port_b.out_msg.cov)

        print(10.)
        print("equality_node.ports[0].in_msg")
        print(self.equality_node.ports[0].in_msg)
        print("equality_node.ports[1].in_msg")
        print(self.equality_node.ports[1].in_msg)
        print("equality_node.ports[2].out_msg")
        print(self.equality_node.ports[2].update())
        return self.equality_node.ports[2].update(GaussianMeanCovMessage)

    def calc_msg_process_noise(self):
        # 1. update all the priors
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        # self.process_noise_node.update_prior(self.port_process_noise.in_msg)
        self.input_node.update_prior(self.port_input.in_msg)
        self.meas_node.update_prior(self.port_meas.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        # 2. iterate through branches
        self.A_node.port_b.update()
        self.C_node.port_b.update()
        self.equality_node.ports[0].update()
        self.B_node.port_b.update()
        self.add_input_node.port_a.update()
        return self.add_process_noise_node.port_b.update()

    def calc_msg_input(self):
        # 1. update all the priors
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.process_noise_node.update_prior(self.port_process_noise.in_msg)
        # self.input_node.update_prior(self.port_input.in_msg)
        self.meas_node.update_prior(self.port_meas.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        # 2. iterate through branches
        self.A_node.port_b.update()
        self.add_process_noise_node.port_c.update()
        self.C_node.port_b.update()
        self.equality_node.ports[0].update()
        self.add_input_node.port_b.update()
        self.B_node.port_a.update()

        return self.B_node.port_a.out_msg

    def calc_msg_meas(self):
        # 1. update all the priors
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.process_noise_node.update_prior(self.port_process_noise.in_msg)
        self.input_node.update_prior(self.port_input.in_msg)
        # self.meas_node.update_prior(self.port_meas.in_msg)
        self.state_out_node.update_prior(self.port_state_out.in_msg)
        # 2. iterate through branches
        self.A_node.port_b.update()
        self.add_process_noise_node.port_c.update()
        self.B_node.port_b.update()
        self.add_input_node.port_c.update()
        self.C_node.port_a.update()

        return self.C_node.port_a.out_msg

    def em_step(self):
        # call update functions in EMMatrixNode
        self.A_node.port_theta.update()
        # in this simple example there is no equality cycle, so just update the matrix
        self.A_node.update_matrix()

    def get_ports(self):
        return [self.port_state_in, self.port_state_out, self.port_meas, self.port_input, self.port_process_noise]


class EMMatrixFilter(object):
    def __init__(self, A, B, C, initial_state_msg, input_noise_cov=None, process_noise_cov=None, meas_noise_cov=None,
                 slice_type=EMMatrixSlice):
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
        :param slice_type: The Kalman filter time slice model to be used.
        """
        # Store default parameter values
        self.A = A
        self.B = B
        self.C = C
        self.slice_type = slice_type
        self.input_noise_cov = input_noise_cov
        self.process_noise_cov = process_noise_cov
        self.meas_noise_cov = meas_noise_cov

        # Initialize factor graph
        self.slices = []
        self.initial_state = PriorNode(initial_state_msg)

    def add_slice(self, input_val, meas_val, A=None, B=None, C=None, meas_noise_msg=None):

        if A is None:
            A = self.A
        if B is None:
            B = self.B
        if C is None:
            C = self.C
        if meas_noise_msg is None:
            meas_noise_msg = GaussianMeanCovMessage(0, self.meas_noise_cov)
        # construct a new slice
        new_slice = self.slice_type(A, B, C)
        # add input & measurements to the mix
        # ToDo: creation of PriorNode needs to be adaptable!
        new_slice.port_meas.connect(PriorNode(GaussianMeanCovMessage(meas_val, np.matrix([[1, 0], [0, 1]]))).port_a)
        new_slice.port_input.connect(PriorNode(GaussianMeanCovMessage(input_val, 0) + meas_noise_msg).port_a)
        new_slice.port_process_noise.connect(PriorNode(GaussianMeanCovMessage(0, 1e-1)).port_a)
        # if it is the first slice - set an initial state - else connect to preceding slice
        if len(self.slices) == 0:
            self.initial_state.port_a.connect(new_slice.port_state_in)
        else:
            self.slices[-1].port_state_out.disconnect()
            self.slices[-1].port_state_out.connect(new_slice.port_state_in)

        # need a final node dto do backwards iteration
        final_node = PriorNode(GaussianMeanCovMessage(0, inf))
        new_slice.port_state_out.connect(final_node.port_a)

        # just build the model - no calculation at this point
        # new_slice.port_state_out.update()
        self.slices.append(new_slice)

        # there is no return
        # return new_slice.port_state_out.out_msg

    def get_state_msgs(self):
        return [slice.port_state_out.out_msg for slice in self.slices]

    def do_forward(self):
        for slice in self.slices:
            # print("do forward try")
            # print(slice.port_state_in.in_msg)
            slice.port_state_out.update()

    def do_backward(self):
        for slice in reversed(self.slices):
            slice.port_state_in.update()

    def do_update(self):
        for slice in self.slices:
            slice.em_step()

    def get_state_msgs(self):
        return [slice.port_state_out.out_message() for slice in self.slices]

    def get_A_value(self):
        return [slice.get_matrix() for slice in self.slices]


class EMSliceNaive(Node):
    def __init__(self, A, B, C, name=None):
        super().__init__(name=name)

        # Initialize ports of the macro (slice) node
        self.port_state_in = NodePort(self, self.calc_msg_state_in)
        self.port_state_out = NodePort(self, self.calc_msg_state_out)
        self.port_meas_in = NodePort(self, self.calc_msg_meas)
        self.port_input_in = NodePort(self, self.calc_msg_input)

        # Initialize all relevant nodes
        self.EM_node = EMNodeScalar(GaussianMeanCovMessage(A, 1).convert(GaussianWeightedMeanInfoMessage))
        self.B_node = MatrixNode(B)
        self.C_node = MatrixNode(C)
        self.equality_node = EqualityNode()
        self.add_node = AdditionNode()

        self.state_in_node = PriorNode(GaussianMeanCovMessage(0, inf))
        self.state_out_node = PriorNode(GaussianMeanCovMessage(0, inf))
        self.meas_in_node = PriorNode(GaussianMeanCovMessage(0, inf))
        self.input_in_node = PriorNode(GaussianMeanCovMessage(0, inf))

        # Connect the nodes
        self.state_in_node.port_a.connect(self.EM_node.port_b)
        self.EM_node.port_a.connect(self.add_node.port_a)

        self.input_in_node.port_a.connect(self.B_node.port_a)
        self.B_node.port_b.connect(self.add_node.port_b)
        self.add_node.port_c.connect(self.equality_node.ports[0])

        self.meas_in_node.port_a.connect(self.C_node.port_a)
        self.C_node.port_b.connect(self.equality_node.ports[1])
        self.equality_node.ports[2].connect(self.state_out_node.port_a)

    def calc_msg_input(self):
        raise NotImplementedError('404 Error - not implemented at this point')

    def calc_msg_state_out(self):
        self.state_in_node.update_prior(self.port_state_in.in_msg)
        self.EM_node.port_a.update(GaussianMeanCovMessage)

        self.input_in_node.update_prior(self.port_input_in.in_msg)
        self.B_node.port_b.update()
        self.add_node.port_c.update(GaussianWeightedMeanInfoMessage)

        self.meas_in_node.update_prior(self.port_meas_in.in_msg)
        self.C_node.port_b.update(GaussianWeightedMeanInfoMessage)
        return self.equality_node.ports[2].update(GaussianMeanCovMessage)
        # return self.equality_node.ports[2].out_msg.convert(GaussianMeanCovMessage)

    def calc_msg_state_in(self):
        self.meas_in_node.update_prior(self.port_meas_in.in_msg)
        self.C_node.port_b.update(GaussianWeightedMeanInfoMessage)

        self.state_out_node.update_prior(self.port_state_out.in_msg, GaussianWeightedMeanInfoMessage)
        self.equality_node.ports[0].update(GaussianMeanCovMessage)

        self.input_in_node.update_prior(self.port_input_in.in_msg)
        self.B_node.port_b.update()
        self.add_node.port_a.update()

        self.EM_node.port_b.update()

        return self.EM_node.port_b.out_msg

    def calc_msg_meas(self):
        self.EM_node.update_theta()

    def get_matrix(self):
        return self.EM_node.get_theta()

    def get_ports(self):
        return [self.port_state_in, self.port_state_out, self.port_meas_in]


class EMFilter(object):
    def __init__(self, A, B, C, initial_state_msg, meas_noise_cov=None,
                 slice_type=EMSliceNaive):
        """
        Initialize a EM filter object given an initial state.

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
        :param slice_type: The Kalman filter time slice model to be used.
        """
        # Store default parameter values
        self.A = A
        self.B = B
        self.C = C
        self.slice_type = slice_type
        self.meas_noise_cov = meas_noise_cov

        # Initialize factor graph
        self.slices = []
        self.initial_state = PriorNode(initial_state_msg)

    def add_slice(self, input_val, meas_val, A=None, B=None, C=None, meas_noise_msg=None):

        if A is None:
            A = self.A
        if B is None:
            B = self.B
        if C is None:
            C = self.C
        if meas_noise_msg is None:
            meas_noise_msg = GaussianMeanCovMessage(0, self.meas_noise_cov)

        new_slice = self.slice_type(A, B, C)

        new_slice.port_meas_in.connect(PriorNode(GaussianMeanCovMessage(meas_val, 0) + meas_noise_msg).port_a)
        new_slice.port_input_in.connect(PriorNode(GaussianMeanCovMessage(input_val, 0) + meas_noise_msg).port_a)

        if len(self.slices) == 0:
            self.initial_state.port_a.connect(new_slice.port_state_in)
        else:
            self.slices[-1].port_state_out.disconnect()
            self.slices[-1].port_state_out.connect(new_slice.port_state_in)

        # need a final node dto do backwards iteration
        final_node = PriorNode(GaussianMeanCovMessage(0, inf))
        new_slice.port_state_out.connect(final_node.port_a)

        # just build the model - no calculation at this point
        # new_slice.port_state_out.update()
        self.slices.append(new_slice)

        # there is no return
        # return new_slice.port_state_out.out_msg

    def get_state_msgs(self):
        return [slice.port_state_out.out_msg for slice in self.slices]

    def do_forward(self):
        for slice in self.slices:
            slice.port_state_out.update()

    def do_backward(self):
        for slice in reversed(self.slices):
            slice.port_state_in.update()

    def do_update(self):
        for slice in self.slices:
            slice.port_meas_in.update()

    def get_A_value(self):
        return [slice.get_matrix() for slice in self.slices]


if __name__ == '__main__':

    # bla = GaussianMeanCovMessage(np.matrix([[1],[2]]), np.matrix([[1],[2]]))
    #
    # Example due to matlab https://de.mathworks.com/help/control/ref/lsim.html

    # just to lazy to make this nice right now
    measurements1 = []
    measurements2 = []
    input1 = []

    file = open("meas.txt", "r")
    for line in file:
        measurements1.append(float(line))

    file2 = open("meas2.txt", "r")
    for line in file2:
        measurements2.append(float(line))

    file3 = open("input.txt", "r")
    for line in file3:
        input1.append(float(line))

    # Define system model
    A = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.matrix([[2], [0], [1], [0]])
    C = transpose(np.matrix([[0.5, -1.25, 0, 0], [0, 0, 1, -0.5]]))

    num_meas = len(input1)

    start = time.perf_counter()

    # initialize the system
    initial_mean = np.matrix([[1], [1], [1], [1]])
    initial_cov = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    initial_state = GaussianMeanCovMessage(initial_mean, initial_cov)
    embase = EMMatrixFilter(A, B, C, initial_state, input_noise_cov=0.01, process_noise_cov=1e-1, meas_noise_cov=0.01)

    # print(np.matrix([[measurements1[1]], [measurements2[1]]]))

    for ii in range(num_meas):
        embase.add_slice(input1[ii], np.matrix([[measurements1[1]], [measurements2[1]]]))

    for i in range(100):
        embase.do_forward()
        embase.do_backward()
        embase.do_update()
    #
    # muh = kf.get_A_value()
    # print(muh)
    # av_value = sum(muh) / num_meas
    # print(av_value)
    #
    # end = time.perf_counter()
    #
    # print('Time elapsed with naive Kalman slice: ' + str(end - start))
    #
    # plt.figure()
    # plt.plot(z, 'k+', label='noisy measurements')
    # plt.plot(squeeze([estimate.mean for estimate in kf.get_state_msgs()]), 'b-', label='a posteriori estimate')
    # plt.plot(z_real, color='g', label='truth value')
    # plt.legend()
    # plt.title('Estimate vs. iteration step', fontweight='bold')
    # plt.xlabel('Iteration')
    # plt.ylabel('')
