# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
import unittest

from ime_fgs.basic_nodes import EqualityNode, AdditionNode, MatrixNode
from ime_fgs.plot import get_all_nodes_and_edges


class TestPlot(unittest.TestCase):
    def test_single_node(self):
        node = EqualityNode(name="=")
        nodes, edges = get_all_nodes_and_edges(node)

        self.assertEqual(nodes, {node})
        self.assertEqual(edges, set())

    def test_multiple_nodes(self):
        node_eq = EqualityNode(name="=")
        node_add = AdditionNode(name="+")
        node_multi = MatrixNode(1, name="M")

        node_eq.ports[0].connect(node_add.port_a)
        node_add.port_b.connect(node_multi.port_b)

        nodes, edges = get_all_nodes_and_edges(node_add)

        self.assertEqual(nodes, {node_eq, node_add, node_multi})
        self.assertEqual(edges, {frozenset((node_eq, node_add)), frozenset((node_add, node_multi))})

    def test_loopy_nodes(self):
        node_eq = EqualityNode(name="=")
        node_add = AdditionNode(name="+")
        node_multi = MatrixNode(1, name="M")

        node_eq.ports[0].connect(node_add.port_a)
        node_add.port_b.connect(node_multi.port_b)
        node_multi.port_a.connect(node_eq.ports[1])

        nodes, edges = get_all_nodes_and_edges(node_add)

        self.assertEqual(nodes, {node_eq, node_add, node_multi})
        self.assertEqual(edges, {frozenset((node_eq, node_add)),
                                 frozenset((node_add, node_multi)),
                                 frozenset((node_multi, node_eq))})


if __name__ == '__main__':
    unittest.main()
