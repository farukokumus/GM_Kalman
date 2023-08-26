# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
import unittest

from ime_fgs.basic_nodes import *
import numpy as np
from ime_fgs.messages import GaussianWeightedMeanInfoMessage, GaussianMeanCovMessage


class EqualizationTests(unittest.TestCase):
    """Tests for equalize_messages."""

    def test_GaussianWeightedMeanInfoScalar(self):
        """
        Does message equalization work for simple, scalar messages in weighted mean and information matrix
        parameterization?
        """

        msg_a = GaussianWeightedMeanInfoMessage([[0]], [[2]])
        msg_b = GaussianWeightedMeanInfoMessage([[2]], [[2]])
        msg_c = msg_a.combine(msg_b)
        self.assertEqual(msg_c.weighted_mean, [[2]])
        self.assertEqual(msg_c.info, [[4]])

        msg_a = GaussianWeightedMeanInfoMessage([[3]], [[3]])
        msg_b = GaussianWeightedMeanInfoMessage([[-12]], [[6]])
        msg_c = msg_a.combine(msg_b)
        self.assertEqual(msg_c.weighted_mean, [[-9]])
        self.assertEqual(msg_c.info, [[9]])
        msg_c = msg_c.convert(GaussianMeanCovMessage)

    def test_GaussianWeightedMeanInfoVector(self):
        """
        Does message equalization work for simple, vector messages in weighted mean and information matrix
        parameterization?
        """

        msg_a = GaussianWeightedMeanInfoMessage(weighted_mean=[[1], [0]], info=[[2, 0], [0, 3]])
        msg_b = GaussianWeightedMeanInfoMessage(weighted_mean=[[2], [0]], info=[[3, 0], [0, 1]])
        msg_c = msg_a.combine(msg_b)
        np.testing.assert_allclose(msg_c.weighted_mean, [[3], [0]])
        np.testing.assert_allclose(msg_c.info, [[5, 0], [0, 4]])


class BaseTests(unittest.TestCase):

    def test_construction(self):
        """Can we create and connect nodes and update message values?"""

        prior_1 = PriorNode(GaussianWeightedMeanInfoMessage(weighted_mean=[[2], [4]], info=[[3, 0], [0, 5]]))
        prior_2 = PriorNode(GaussianWeightedMeanInfoMessage(weighted_mean=[[1], [3]], info=[[2, 2], [2, 2]]))
        prior_3 = PriorNode(GaussianWeightedMeanInfoMessage(weighted_mean=[[0], [2]], info=[[4, 0], [0, 15]]))
        eq = EqualityNode()

        prior_1.port_a.connect(eq.ports[0])
        prior_2.port_a.connect(eq.ports[1])
        prior_3.port_a.connect(eq.ports[2])
        with self.assertRaises(ConnectionRefusedError):
            prior_2.port_a.connect(eq.ports[0])

        prior_1.port_a.update()
        prior_2.port_a.update()
        prior_3.port_a.update()
        eq.ports[0].update()
        eq.ports[1].update()
        eq.ports[2].update()

        prior_1_mar = prior_1.port_a.marginal()
        prior_2_mar = prior_2.port_a.marginal()
        prior_3_mar = prior_3.port_a.marginal()
        eq_a_mar = eq.ports[0].marginal()
        eq_b_mar = eq.ports[1].marginal()
        eq_c_mar = eq.ports[2].marginal()

        self.assertTrue(prior_1_mar == prior_2_mar == prior_3_mar == eq_a_mar == eq_b_mar == eq_c_mar)

    def test_construction_gaussian_weighted_mean_info_message(self):
        """Can we create and connect nodes and update message values?"""

        prior_1 = PriorNode(GaussianWeightedMeanInfoMessage(weighted_mean=[[4], [1]], info=[[3, 0], [0, 2]]))
        prior_2 = PriorNode(GaussianWeightedMeanInfoMessage(weighted_mean=[[1], [4]], info=[[1, 0], [0, 1]]))
        prior_3 = PriorNode(GaussianWeightedMeanInfoMessage(weighted_mean=[[4], [5]], info=[[3, 0], [0, 2]]))
        eq = EqualityNode()

        prior_1.port_a.connect(eq.ports[0])
        prior_2.port_a.connect(eq.ports[1])
        prior_3.port_a.connect(eq.ports[2])
        with self.assertRaises(ConnectionRefusedError):
            prior_2.port_a.connect(eq.ports[0])

        prior_1.port_a.update()
        prior_2.port_a.update()
        prior_3.port_a.update()
        eq.ports[0].update()
        eq.ports[1].update()
        eq.ports[2].update()

        prior_1_mar = prior_1.port_a.marginal()
        prior_2_mar = prior_2.port_a.marginal()
        prior_3_mar = prior_3.port_a.marginal()
        eq_a_mar = eq.ports[0].marginal()
        eq_b_mar = eq.ports[1].marginal()
        eq_c_mar = eq.ports[2].marginal()

        prior_1_mar = prior_1_mar.convert(GaussianWeightedMeanInfoMessage)
        prior_2_mar = prior_2_mar.convert(GaussianWeightedMeanInfoMessage)
        prior_3_mar = prior_3_mar.convert(GaussianWeightedMeanInfoMessage)
        eq_a_mar = eq_a_mar.convert(GaussianWeightedMeanInfoMessage)
        eq_b_mar = eq_b_mar.convert(GaussianWeightedMeanInfoMessage)
        eq_c_mar = eq_c_mar.convert(GaussianWeightedMeanInfoMessage)

        self.assertTrue(prior_1_mar == prior_2_mar == prior_3_mar == eq_a_mar == eq_b_mar == eq_c_mar)

    def test_is_non_informative(self):
        """
        Some basic tests for the non informative test
        """
        self.assertFalse(GaussianMeanCovMessage([[0], [0]], [[1, 0],
                                                             [0, 1]]).is_non_informative())
        self.assertFalse(GaussianWeightedMeanInfoMessage([[0], [0]], [[1, 0],
                                                                      [0, 1]]).is_non_informative())
        self.assertTrue(GaussianMeanCovMessage([[0], [0]], [[np.inf, 0],
                                                            [0, np.inf]]).is_non_informative())
        self.assertTrue(GaussianWeightedMeanInfoMessage([[0], [0]], [[0, 0],
                                                                     [0, 0]]).is_non_informative())

    def test_in_msg(self):
        eq_node = EqualityNode()
        m_node = MatrixNode(1)

        eq_node.ports[0].connect(m_node.port_a)

        msg = GaussianMeanCovMessage([[0]], [[42]])
        eq_node.ports[0].in_msg = msg

        self.assertIs(eq_node.ports[0].in_msg, m_node.port_a.out_msg)


if __name__ == '__main__':
    unittest.main()
