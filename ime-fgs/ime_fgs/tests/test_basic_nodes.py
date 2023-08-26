# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
import copy
import unittest
import numpy as np
from ime_fgs.basic_nodes import EqualityNode, PriorNode, AdditionNode, BigAdditionNode, MatrixNode
from ime_fgs.messages import GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage


class EqualityNodeTest(unittest.TestCase):
    """
                +---+
     ports[0]   |   |
    ------------|   |
                | = |
     ports[1]   |   |
    ------------|   |
                |   |
                |   |
    """

    def test_equality_node(self):
        """ test update and _calc_msg"""
        e_node = EqualityNode(name="node_a", number_of_ports=4)
        msg_a = GaussianWeightedMeanInfoMessage([[1], [2]], np.identity(2) * 2)
        msg_b = GaussianWeightedMeanInfoMessage([[1], [2]], np.identity(2) * 2)
        msg_c = GaussianWeightedMeanInfoMessage([[2], [3]], np.identity(2))
        p_node_a = PriorNode(msg_a)
        p_node_b = PriorNode(msg_b)
        p_node_c = PriorNode(msg_c)
        p_node_a.port_a.connect(e_node.ports[0])
        p_node_b.port_a.connect(e_node.ports[1])
        p_node_c.port_a.connect(e_node.ports[2])
        msg_d = e_node.ports[3].update(GaussianWeightedMeanInfoMessage)
        msg_e = e_node._calc_msg(3)
        msg_f = msg_a.combine(msg_b.combine(msg_c))
        self.assertEqual(msg_d, msg_e, msg_f)

    def test_init(self):
        # test possibility to initialize EqualityNodes with different number_of_ports
        with self.assertRaises(AssertionError):
            EqualityNode(number_of_ports=0)
        with self.assertRaises(AssertionError):
            EqualityNode(number_of_ports=1)
        e_node_2 = EqualityNode(number_of_ports=2)
        e_node_5 = EqualityNode(number_of_ports=5)

    def test_unconnected_ports(self):
        # test unconnected ports
        e_node = EqualityNode("e_node", 3, True)
        msg_a = GaussianWeightedMeanInfoMessage([[1], [0]], np.identity(2) * 2)
        p_node = PriorNode(msg_a)
        p_node.port_a.connect(e_node.ports[0])
        msg_b = e_node.ports[1].update(GaussianWeightedMeanInfoMessage)
        msg_c = e_node._calc_msg_unconnected(2)
        self.assertEqual(msg_b, msg_a, msg_c)

    def test_unconnected_ports_assertion(self):
        e_node = EqualityNode(number_of_ports=3)
        msg_a = GaussianWeightedMeanInfoMessage([[1], [0]], np.identity(2) * 2)
        p_node = PriorNode(msg_a)
        p_node.port_a.connect(e_node.ports[0])
        with self.assertRaises(ConnectionError):
            e_node.ports[2].update(GaussianWeightedMeanInfoMessage)

    def test_get_ports(self):
        """ test if size of returned list matches with number_of_ports """
        e_node = EqualityNode(number_of_ports=4)
        msg_a = GaussianWeightedMeanInfoMessage([[2], [3]], np.identity(2) * 4)
        p_node = PriorNode(msg_a, name="p_node")
        p_node.port_a.connect(e_node.ports[0])
        self.assertTrue(len(e_node.get_ports()) == 4)

    def test_ports_unconnected(self):
        e_node = EqualityNode(number_of_ports=5, allow_unconnected_ports=True)
        msg_a = GaussianWeightedMeanInfoMessage([[2], [3]], np.identity(2) * 4)
        p_node = PriorNode(msg_a, name="p_node")
        p_node.port_a.connect(e_node.ports[0])
        self.assertTrue(len(e_node.get_ports()) == 1)


class AdditionNodeTest(unittest.TestCase):
    """
      a +---+ c
    --->| + |--->
        +---+
          ^
          |
          | b
    """

    def test_calc_msg_a(self):
        a_node = AdditionNode(name="a_node")
        msg_b = GaussianMeanCovMessage([[1], [4]], np.identity(2) * 2)
        msg_c = GaussianMeanCovMessage([[1], [1]], np.identity(2) * 4)
        p_node_b = PriorNode(msg_b)
        p_node_c = PriorNode(msg_c)
        p_node_b.port_a.connect(a_node.port_b)
        p_node_c.port_a.connect(a_node.port_c)
        msg_d = a_node.port_a.update(GaussianMeanCovMessage)
        msg_a = msg_c - msg_b
        msg_e = a_node._calc_msg_a()
        self.assertEqual(msg_d, msg_a, msg_e)

    def test_calc_msg_b(self):
        a_node = AdditionNode(name="a_node")
        msg_a = GaussianMeanCovMessage([[1], [2]], np.identity(2) * 2)
        msg_c = GaussianMeanCovMessage([[1], [1]], np.identity(2) * 1)
        p_node_a = PriorNode(msg_a)
        p_node_c = PriorNode(msg_c)
        p_node_a.port_a.connect(a_node.port_a)
        p_node_c.port_a.connect(a_node.port_c)
        msg_b = a_node.port_b.update(GaussianMeanCovMessage)
        msg_d = - msg_a + msg_c
        msg_e = a_node._calc_msg_b()
        self.assertEqual(msg_b, msg_d, msg_e)

    def test_calc_msg_c(self):
        a_node = AdditionNode("a_node")
        msg_a = GaussianMeanCovMessage([[1], [2]], np.identity(2) * 2)
        msg_b = GaussianMeanCovMessage([[1], [1]], np.identity(2) * 1)
        p_node_a = PriorNode(msg_a)
        p_node_b = PriorNode(msg_b)
        p_node_a.port_a.connect(a_node.port_a)
        p_node_b.port_a.connect(a_node.port_b)
        msg_c = a_node.port_c.update(GaussianMeanCovMessage)
        msg_d = msg_a + msg_b
        msg_e = a_node._calc_msg_c_()
        self.assertEqual(msg_c, msg_d, msg_e)

    def test_get_ports(self):
        a_node = AdditionNode(name="Knoten")
        self.assertTrue(len(a_node.get_ports()) == 3)


class BigAdditionNodeTest(unittest.TestCase):
    """
                   +---+
     ports_in[0]   |   | ports_out[0]
    -------------->|   |------------->
                   | + |
     ports_in[1]   |   | ports_out[1]
    -------------->|   |------------->
                   |   |
                   |   |
    """

    def test_calc_msg_in(self):
        b_node = BigAdditionNode(3, 3)
        # initialize in-ports
        msg_a_in = GaussianMeanCovMessage([[1], [4]], np.identity(2) * 2)
        msg_b_in = GaussianMeanCovMessage([[1], [1]], np.identity(2) * 4)
        msg_c_in = GaussianMeanCovMessage([[0], [4]], np.identity(2) * 2)
        p_node_a = PriorNode(msg_a_in)
        p_node_b = PriorNode(msg_b_in)
        p_node_c = PriorNode(msg_c_in)
        p_node_a.port_a.connect(b_node.ports_in[0])
        p_node_b.port_a.connect(b_node.ports_in[1])
        p_node_c.port_a.connect(b_node.ports_in[2])
        # initialize out-ports
        msg_a_out = GaussianMeanCovMessage([[1], [4]], np.identity(2) * 2)
        msg_b_out = GaussianMeanCovMessage([[1], [1]], np.identity(2) * 4)
        p_node_d = PriorNode(msg_a_out)
        p_node_e = PriorNode(msg_b_out)
        p_node_d.port_a.connect(b_node.ports_out[0])
        p_node_e.port_a.connect(b_node.ports_out[1])
        # calculate out_msg ports[2]
        msg_d = b_node._calc_msg_out(2)
        msg_e = msg_a_in + msg_b_in + msg_c_in - msg_a_out - msg_b_out
        msg_f = b_node.ports_out[2].update(GaussianMeanCovMessage)
        self.assertEqual(msg_d, msg_e, msg_f)

    def test_calc_msg_out(self):
        b_node = BigAdditionNode(3, 2)
        # initialize 2 in-ports
        msg_a_in = GaussianMeanCovMessage([[1], [2 + 1j], [3]], np.identity(3) * 6)
        msg_b_in = GaussianMeanCovMessage([[1], [1], [1]], np.identity(3) * 4)
        p_node_a = PriorNode(msg_a_in)
        p_node_b = PriorNode(msg_b_in)
        p_node_a.port_a.connect(b_node.ports_in[0])
        p_node_b.port_a.connect(b_node.ports_in[1])
        # initialize out-ports
        msg_a_out = GaussianMeanCovMessage([[1], [4], [4]], np.identity(3) * 2)
        msg_b_out = GaussianMeanCovMessage([[1], [1], [9]], np.identity(3) * (3 + 5j))
        p_node_d = PriorNode(msg_a_out)
        p_node_e = PriorNode(msg_b_out)
        p_node_d.port_a.connect(b_node.ports_out[0])
        p_node_e.port_a.connect(b_node.ports_out[1])
        # calculate ports-in[2]
        msg_d = b_node._calc_msg_in(2)
        msg_e = - msg_a_in - msg_b_in + msg_a_out + msg_b_out
        msg_f = b_node.ports_in[2].update(GaussianMeanCovMessage)
        self.assertEqual(msg_d, msg_f, msg_e)

    def test_get_ports(self):
        b_node = BigAdditionNode(3, 7)
        self.assertTrue(len(b_node.get_ports()) == (3 + 7))


class MatrixNodeTest(unittest.TestCase):
    """
      a +--------+ b
    --->| matrix |--->
        +--------+
    """

    def test_matrix_1(self):
        node_a = MatrixNode(np.identity(7) * 9, "nod_a")
        m_a = np.identity(7) * 9
        m_b = node_a.matrix
        self.assertTrue(np.allclose(m_a, m_b))

    def test_matrix_2(self):
        node_a = MatrixNode(np.identity(7) * 9, "nod_a")
        node_a.matrix = np.identity(2) * 1
        m_b = np.identity(2)
        self.assertTrue(np.allclose(m_b, node_a.matrix))

    def test_matrix_dimensions(self):
        matrix = 3
        m_node = MatrixNode(matrix)
        p_node = PriorNode(GaussianMeanCovMessage([[2], [3]], np.identity(2) * 3))
        p_node.port_a.connect(m_node.port_a)
        with self.assertRaises(ValueError):
            m_node.port_b.update(GaussianMeanCovMessage)

    def test_calc_msg_b(self):
        """ test update, _calc_msg_b"""
        matrix = np.identity(2) * 3
        m_node = MatrixNode(matrix)
        msg_a = GaussianMeanCovMessage([[1], [2]], np.identity(2) * 2)
        p_node = PriorNode(msg_a)
        p_node.port_a.connect(m_node.port_a)
        msg_b = m_node._calc_msg_b()
        msg_c = m_node.port_b.update(GaussianMeanCovMessage)
        msg_d = msg_a.multiply_deterministic(matrix)
        self.assertEqual(msg_b, msg_d, msg_c)

    def test_calc_msg_a(self):
        """ test update, calc_msg_a"""
        matrix = np.identity(2) * 3
        m_node_a = MatrixNode(matrix)
        msg_a = GaussianWeightedMeanInfoMessage([[1], [2]], np.identity(2) * 2)
        p_node_a = PriorNode(msg_a)
        p_node_a.port_a.connect(m_node_a.port_b)
        msg_b = m_node_a._calc_msg_a()
        msg_c = m_node_a.port_a.update(GaussianWeightedMeanInfoMessage)
        msg_d = msg_a.multiply_deterministic(matrix, 1)
        self.assertEqual(msg_b, msg_d, msg_c)

    def get_ports(self):
        m_node = MatrixNode(np.identity(2))
        self.assertTrue(len(m_node.get_ports()) == 2)


class PriorNodeTest(unittest.TestCase):
    """
    +---+ a
    |   |---
    +---+
     msg
    """

    def test_update_prior(self):
        msg_a = GaussianMeanCovMessage([[1], [2]], np.identity(2) * 3)
        p_node = PriorNode(msg_a, "p_node")
        msg_b = GaussianWeightedMeanInfoMessage([[1], [3]], np.identity(2) * 2)
        # target_type egal??
        p_node.update_prior(msg_b, GaussianMeanCovMessage)
        self.assertEqual(p_node._prior, msg_b)

    def test_update_1(self):
        msg_a = GaussianMeanCovMessage([[1], [2]], np.identity(2) * 3)
        p_node = PriorNode(msg_a, name="p_node")
        msg_b = p_node.port_a.update(GaussianWeightedMeanInfoMessage)
        msg_c = msg_a.convert(GaussianWeightedMeanInfoMessage)
        self.assertEqual(msg_c, msg_b)

    def test_update_2(self):
        msg_a = GaussianMeanCovMessage([[1], [2]], np.identity(2) * 3)
        p_node = PriorNode(msg_a, name="p_node")
        msg_d = p_node.port_a.update(GaussianMeanCovMessage)
        self.assertEqual(msg_d, msg_a)

    def test_get_ports(self):
        msg_a = GaussianMeanCovMessage([[1]], [[7]])
        p_node = PriorNode(msg_a)
        self.assertTrue(len(p_node.get_ports()) == 1)

    def test_deep_copy(self):
        msg1 = GaussianMeanCovMessage([[1]], [[42]])
        msg2 = GaussianMeanCovMessage([[0]], [[42]])
        p_node1 = PriorNode(msg1)
        p_node2 = copy.deepcopy(p_node1)
        p_node2.update_prior(msg2)
        self.assertEqual(p_node2.port_a.out_msg, msg2)


if __name__ == '__main__':
    unittest.main()
