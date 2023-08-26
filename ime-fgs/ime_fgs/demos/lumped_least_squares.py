# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
from numpy import linspace, inf, squeeze, concatenate
import numpy as np
import matplotlib.pyplot as plt

from ime_fgs.base import NodePort, Node
from ime_fgs.basic_nodes import MatrixNode, AdditionNode, PriorNode, EqualityNode
from ime_fgs.messages import GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage

"""
Simple least squares example, written in Berlin, 10.10.2017
min J = e'Qe + u'Ru
s.t. e = Au + y
"""

if __name__ == '__main__':
    # Initialize variables
    A = [[1, 0], [0, 1]]

    # Initialize all relevant nodes
    A_node = MatrixNode(A)
    meas_node = PriorNode(GaussianMeanCovMessage([[1], [1]], 1e-1 * np.eye(2)))
    input_node = PriorNode(GaussianWeightedMeanInfoMessage([[0], [0]], 0 * np.eye(2)))
    output_node = PriorNode(GaussianMeanCovMessage([[0], [0]], 1e1 * np.eye(2)))
    addition_node = AdditionNode()

    # Connect the nodes
    input_node.port_a.connect(A_node.port_a)
    A_node.port_b.connect(addition_node.port_a)
    meas_node.port_a.connect(addition_node.port_b)
    addition_node.port_c.connect(output_node.port_a)

    # Perform message passing
    addition_node.port_a.update(GaussianWeightedMeanInfoMessage)
    A_node.port_a.update(GaussianWeightedMeanInfoMessage)

    # Marginalize
    result = input_node.port_a.out_msg.combine(
        input_node.port_a.in_msg)
    print(result.convert(GaussianMeanCovMessage))
