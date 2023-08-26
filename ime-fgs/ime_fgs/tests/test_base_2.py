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
from ime_fgs.basic_nodes import EqualityNode, PriorNode, AdditionNode, BigAdditionNode, MatrixNode
from ime_fgs.messages import GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage


class NodePortTest(unittest.TestCase):
    def test_init_False(self):
        msg_a = GaussianMeanCovMessage([[1], [2]], np.identity(2))
        p_node = PriorNode(msg_a)
        self.assertFalse(p_node.port_a._value_in_cache, p_node.port_a._connected)

    def test_apply_cached_out_message_1(self):
        e_node = EqualityNode(number_of_ports=3)
        msg_a = GaussianWeightedMeanInfoMessage([[3]], [[7]])
        msg_b = GaussianWeightedMeanInfoMessage([[8]], [[9]])
        p_node_a = PriorNode(msg_a)
        p_node_b = PriorNode(msg_b)
        e_node.ports[0].connect(p_node_a.port_a)
        e_node.ports[1].connect(p_node_b.port_a)
        e_node.ports[2].update(cached=True)
        msg = e_node.ports[2].cached_out_msg
        e_node.ports[2].apply_cached_out_message()
        self.assertEqual(msg, e_node.ports[2].out_msg)

    def test_apply_cached_out_message_2(self):
        e_node = EqualityNode(number_of_ports=3)
        msg_a = GaussianWeightedMeanInfoMessage([[3]], [[7]])
        msg_b = GaussianWeightedMeanInfoMessage([[8]], [[9]])
        p_node_a = PriorNode(msg_a)
        p_node_b = PriorNode(msg_b)
        e_node.ports[0].connect(p_node_a.port_a)
        e_node.ports[1].connect(p_node_b.port_a)
        e_node.ports[2].update(cached=True)
        e_node.ports[2].apply_cached_out_message()
        self.assertEqual(None, e_node.ports[2].cached_out_msg)

    def test_apply_cached_out_message_Exception(self):
        """ test by initializing AdditionNode """
        a_node = AdditionNode(name="a_node")
        with self.assertRaises(Exception):
            a_node.port_a.apply_cached_out_message()

    def test_connected_if_connected(self):
        e_node = EqualityNode(number_of_ports=5, allow_unconnected_ports=True)
        p_node = PriorNode(GaussianMeanCovMessage([[1]], [[3]]))
        p_node.port_a.connect(e_node.ports[0])
        self.assertTrue(e_node.ports[0].connected)

    def test_connected_if_unconnected(self):
        e_node = EqualityNode(number_of_ports=5, allow_unconnected_ports=True)
        self.assertFalse(e_node.ports[0].connected)

    def test_in_msg(self):
        a_node = AdditionNode()
        msg_a = GaussianMeanCovMessage([[4], [6]], np.identity(2) * 5)
        p_node = PriorNode(msg_a)
        p_node.port_a.connect(a_node.port_a)
        self.assertEqual(a_node.port_a.in_msg, msg_a, p_node.port_a.in_msg)

    def test_in_msg_Error(self):
        a_node = AdditionNode()
        with self.assertRaises(ConnectionError):
            a_node.port_a.in_msg

    def test_update_RuntimeError(self):
        a_node_1 = AdditionNode()
        a_node_2 = AdditionNode()
        a_node_1.port_a.connect(a_node_2.port_a)
        msg_a = GaussianMeanCovMessage([[2]], [[1]])
        p_node_1 = PriorNode(msg_a)
        p_node_1.port_a.connect(a_node_2.port_b)
        with self.assertRaises(RuntimeError):
            a_node_2.port_c.update()

    def test_update_out_msg(self):
        m_node = MatrixNode(np.identity(2) * 7)
        p_node = PriorNode(GaussianMeanCovMessage([[2], [3]], np.identity(2) * 5))
        m_node.port_a.connect(p_node.port_a)
        msg = m_node.port_b.update()
        self.assertEqual(msg, p_node._prior.multiply_deterministic(m_node.matrix))

    def test_update_target_type_1(self):
        a_node = AdditionNode()
        msg_a = GaussianMeanCovMessage([[2]], [[7]])
        msg_b = GaussianMeanCovMessage([[4]], [[6]])
        p_node_a = PriorNode(msg_a)
        p_node_b = PriorNode(msg_b)
        a_node.port_a.connect(p_node_a.port_a)
        a_node.port_b.connect(p_node_b.port_a)
        a_node.port_c.update(GaussianWeightedMeanInfoMessage)
        self.assertEqual(GaussianWeightedMeanInfoMessage, a_node.port_c.target_type)

    def test_update_target_type_2(self):
        a_node = AdditionNode()
        msg_a = GaussianMeanCovMessage([[2]], [[7]])
        msg_b = GaussianMeanCovMessage([[4]], [[6]])
        p_node_a = PriorNode(msg_a)
        p_node_b = PriorNode(msg_b)
        a_node.port_a.connect(p_node_a.port_a)
        a_node.port_b.connect(p_node_b.port_a)
        msg_c = a_node.port_c.update(GaussianWeightedMeanInfoMessage)
        msg_d = msg_a + msg_b
        self.assertEqual(msg_d.convert(GaussianWeightedMeanInfoMessage), msg_c)

    def test_update_cached_True_1(self):
        e_node = EqualityNode(number_of_ports=3)
        msg_a = GaussianWeightedMeanInfoMessage([[3]], [[7]])
        msg_b = GaussianWeightedMeanInfoMessage([[8]], [[9]])
        p_node_a = PriorNode(msg_a)
        p_node_b = PriorNode(msg_b)
        e_node.ports[0].connect(p_node_a.port_a)
        e_node.ports[1].connect(p_node_b.port_a)
        e_node.ports[2].update(cached=True)
        self.assertTrue(e_node.ports[2]._value_in_cache)

    def test_update_cached_True_2(self):
        e_node = EqualityNode(number_of_ports=3)
        msg_a = GaussianWeightedMeanInfoMessage([[2], [4]], np.identity(2) * 7)
        msg_b = GaussianWeightedMeanInfoMessage([[8], [9]], np.identity(2) * 2)
        p_node_a = PriorNode(msg_a)
        p_node_b = PriorNode(msg_b)
        e_node.ports[0].connect(p_node_a.port_a)
        e_node.ports[1].connect(p_node_b.port_a)
        msg = e_node.ports[2].update(cached=True)
        self.assertEqual(msg, e_node.ports[2].cached_out_msg)

    def test_update_cached_False(self):
        e_node = EqualityNode(number_of_ports=3)
        msg_a = GaussianWeightedMeanInfoMessage([[2], [14]], np.identity(2) * 17)
        msg_b = GaussianWeightedMeanInfoMessage([[18], [9]], np.identity(2) * 2)
        p_node_a = PriorNode(msg_a)
        p_node_b = PriorNode(msg_b)
        e_node.ports[0].connect(p_node_a.port_a)
        e_node.ports[1].connect(p_node_b.port_a)
        msg = e_node.ports[2].update()
        self.assertEqual(msg, e_node.ports[2].out_msg)

    def test_connect_ConnectionRefusedError(self):
        a_node = AdditionNode()
        msg_a = GaussianMeanCovMessage([[2], [7]], np.identity(2) * 17)
        p_node = PriorNode(msg_a)
        p_node.port_a.connect(a_node.port_a)
        with self.assertRaises(ConnectionRefusedError):
            p_node.port_a.connect(a_node.port_b)

    def test_connect_other_port_1(self):
        a_node = AdditionNode()
        msg_a = GaussianMeanCovMessage([[7], [7]], np.identity(2) * 11)
        p_node = PriorNode(msg_a)
        p_node.port_a.connect(a_node.port_a)
        self.assertEqual(p_node.port_a.other_port, a_node.port_a)

    def test_connect_other_port_2(self):
        e_node = EqualityNode(number_of_ports=3)
        msg_a = GaussianMeanCovMessage([[7], [7]], np.identity(2) * 11)
        p_node = PriorNode(msg_a)
        p_node.port_a.connect(e_node.ports[0])
        self.assertEqual(p_node.port_a, e_node.ports[0].other_port)

    def test_connect_connected_1(self):
        matrix = np.identity(2) * 17
        m_node = MatrixNode(matrix)
        msg_a = GaussianMeanCovMessage([[2], [4]], np.identity(2) * 5)
        p_node = PriorNode(msg_a)
        m_node.port_a.connect(p_node.port_a)
        self.assertTrue(m_node.port_a._connected)

    def test_connect_connected_2(self):
        matrix = np.identity(2) * 17
        m_node = MatrixNode(matrix)
        msg_a = GaussianMeanCovMessage([[2], [4]], np.identity(2) * 5)
        p_node = PriorNode(msg_a)
        m_node.port_a.connect(p_node.port_a)
        self.assertTrue(p_node.port_a._connected)

    def test_connect_target_type_1(self):
        a_node = AdditionNode()
        msg_a = GaussianMeanCovMessage([[2], [5]], np.identity(2) * 9)
        p_node = PriorNode(msg_a)
        p_node.port_a.connect(a_node.port_a, out_type=GaussianWeightedMeanInfoMessage)
        self.assertEqual(GaussianWeightedMeanInfoMessage, p_node.port_a.target_type)

    def test_connect_target_type_2(self):
        a_node = AdditionNode()
        msg_a = GaussianMeanCovMessage([[2], [5]], np.identity(2) * 9)
        p_node = PriorNode(msg_a)
        p_node.port_a.connect(a_node.port_a, in_type=GaussianWeightedMeanInfoMessage)
        self.assertEqual(GaussianWeightedMeanInfoMessage, a_node.port_a.target_type)

    def test_disconnect_ConnectionRefusedError(self):
        m_node = MatrixNode(np.identity(2) * 101)
        with self.assertRaises(ConnectionRefusedError):
            m_node.port_a.disconnect()

    def test_disconnect_connected_1(self):
        a_node = AdditionNode()
        msg_a = GaussianMeanCovMessage([[2], [17]], np.identity(2) * 22)
        p_node = PriorNode(msg_a)
        p_node.port_a.connect(a_node.port_a)
        a_node.port_a.disconnect()
        self.assertFalse(a_node.port_a._connected)

    def test_disconnect_connected_2(self):
        a_node = AdditionNode()
        msg_a = GaussianMeanCovMessage([[2], [17]], np.identity(2) * 22)
        p_node = PriorNode(msg_a)
        p_node.port_a.connect(a_node.port_a)
        a_node.port_a.disconnect()
        self.assertFalse(p_node.port_a._connected)

    def test_disconnect_out_msg(self):
        m_node = MatrixNode(np.identity(2) * 2)
        msg_a = GaussianMeanCovMessage([[8], [0]], np.identity(2))
        p_node = PriorNode(msg_a)
        m_node.port_a.connect(p_node.port_a)
        m_node.port_a.disconnect()
        self.assertEqual(msg_a, p_node.port_a.out_msg)

    def test_marginal(self):
        e_node = EqualityNode(number_of_ports=3)
        msg_a = GaussianWeightedMeanInfoMessage([[3], [7]], np.identity(2) * 4)
        msg_b = GaussianWeightedMeanInfoMessage([[1], [0]], np.identity(2) * 2)
        msg_c = GaussianWeightedMeanInfoMessage([[2], [3]], np.identity(2) * 6)
        p_node_a = PriorNode(msg_a)
        p_node_b = PriorNode(msg_b)
        p_node_c = PriorNode(msg_c)
        p_node_a.port_a.connect(e_node.ports[0])
        p_node_b.port_a.connect(e_node.ports[1])
        p_node_c.port_a.connect(e_node.ports[2])
        msg_d = e_node.ports[2].update()
        msg_f = e_node.ports[2].marginal()
        msg_g = msg_d.combine(msg_c)
        self.assertEqual(msg_f, msg_g)


if __name__ == '__main__':
    unittest.main()
