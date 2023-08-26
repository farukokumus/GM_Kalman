# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
from ime_fgs.base import NodePort
from numpy import asarray, atleast_2d, squeeze, identity
from numpy.linalg import inv

from ime_fgs.messages import GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage, GaussianTildeMessage
from ime_fgs.base import Node, NodePortType


class CompoundEqualityMatrixNode(Node):
    """
    Implements combination of Equality node and matrix multiplication node.

      a +---+ c
    --->| = |--->
        +---+
          |
          v
        +---+
        | A |
        +---+
          | b
          v

    Equivalent to factor delta(x-(A^-1)*y)*delta(x-z).

    See Loeliger (2016): On Sparsity by NUV-EM...
    """

    def __init__(self, matrix, name=None):
        super().__init__(name=name)
        self.port_a = NodePort(self, self._calc_msg_a, NodePortType.InPort)
        self.port_b = NodePort(self, self._calc_msg_b, NodePortType.OutPort)
        self.port_c = NodePort(self, self._calc_msg_c, NodePortType.OutPort)
        self.matrix = atleast_2d(squeeze(asarray(matrix)))
        self.matrix_h = self.matrix.transpose().conj()

    def _calc_msg_a(self):
        msg_a = self.port_a.in_msg
        msg_b = self.port_b.in_msg
        msg_c = self.port_c.in_msg

        if isinstance(msg_b, GaussianWeightedMeanInfoMessage) and isinstance(msg_c, GaussianWeightedMeanInfoMessage):
            msg_Ab_weighted_mean = self.matrix_h @ msg_b.weighted_mean
            msg_Ab_info = self.matrix_h @ msg_b.info @ self.matrix
            return GaussianWeightedMeanInfoMessage(msg_Ab_weighted_mean + msg_c.weighted_mean, msg_Ab_info + msg_c.info)

        elif isinstance(msg_b, GaussianMeanCovMessage) and isinstance(msg_c, GaussianTildeMessage) and \
                isinstance(msg_a, GaussianMeanCovMessage):
            G = inv(msg_b.cov + self.matrix @ msg_a.cov @ self.matrix_h)
            F = identity(msg_a.cov.shape[0]) - msg_a.cov @ self.matrix_h @ G @ self.matrix
            F_h = F.transpose().conjugate()
            xi_a = F_h @ msg_c.xi + self.matrix_h @ G @ (self.matrix @ msg_a.mean - msg_b.mean)
            W_a = F_h @ msg_c.W @ F + self.matrix_h @ G @ self.matrix
            return GaussianTildeMessage(xi_a, W_a)

        else:
            raise NotImplementedError()

    @staticmethod
    def _calc_msg_b():
        raise NotImplementedError()

    def _calc_msg_c(self):
        msg_a = self.port_a.in_msg
        msg_b = self.port_b.in_msg
        if isinstance(msg_a, GaussianMeanCovMessage) and isinstance(msg_b, GaussianMeanCovMessage):
            G = inv(msg_b.cov + self.matrix @ msg_a.cov @ self.matrix_h)
            mean = msg_a.mean + msg_a.cov @ self.matrix_h @ G @ (msg_b.mean - self.matrix @ msg_a.mean)
            cov = msg_a.cov - msg_a.cov @ self.matrix_h @ G @ self.matrix @ msg_a.cov
            return GaussianMeanCovMessage(mean, cov)
        else:
            raise NotImplementedError()

    def get_ports(self):
        return [self.port_a, self.port_b, self.port_c]
