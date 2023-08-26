# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
import unittest
import numpy as np
from ime_fgs.basic_nodes import PriorNode, AdditionNode
from ime_fgs.em_node import NUVPrior
from ime_fgs.messages import GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage


class NUVPriorNodeTest(unittest.TestCase):
    def test_NUVPrior_node(self):
        """ Test
                                +---+
                                +   + mean_offset
                                +---+
                     | Theta      |
                     |            |
                     | A(Theta)   |
          Nx         v            v
        +---+      +---+        +---+    X   +---+        +---+
        +   +----->| x |------->+ + +------->+ + +------->+   + Nz
        +---+      +---+        +---+        +---+        +---+
                                               A
                                               |
                                             +---+
                                             +   + Nd
                                             +---+
        """
        NUVPrior_node = NUVPrior(name="node_NUV", A_0=0 * np.identity(1), A_theta=[[1]], mean_offset=np.array([[0]]),
                                 non_informative_prior=True)
        msg_z = GaussianMeanCovMessage([[1]], np.identity(1) * 0.0)
        msg_d = GaussianMeanCovMessage([[0]], np.identity(1) * 0.5)

        # From Loeliger, Bruderer, Malmberg, Radehn, Zalmai 2016
        # On Sparsity by NUV-EM, Gaussian Message Passing, and Kalman Smoothing
        #
        # var_x = argmax mu_marg_z = max(0, m_z^2 - var_d)
        #   m_x = (mz^2 - var_d) / m_z for var_x = m_z^2 - var_d, or 0 otherwise

        # expect var_x = 0.5
        #          m_x = 0.5

        p_node_d = PriorNode(msg_d)
        p_node_z = PriorNode(msg_z)
        add_node = AdditionNode()
        NUVPrior_node.port_b.connect(add_node.port_a)
        add_node.port_c.connect(p_node_z.port_a)
        add_node.port_b.connect(p_node_d.port_a)

        for ii in range(0, 20):
            add_node.port_a.update()
            NUVPrior_node.port_theta.update()
            add_node.port_b.update()
            add_node.port_c.update()
            # print(NUVPrior_node.get_theta())
            # print(NUVPrior_node.port_b.marginal(target_type=GaussianMeanCovMessage))

        assert (np.allclose(NUVPrior_node.get_theta() ** 2, 0.5))


if __name__ == '__main__':
    unittest.main()
