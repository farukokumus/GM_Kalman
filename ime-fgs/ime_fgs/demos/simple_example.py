# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
from numpy import inf

from ime_fgs.basic_nodes import MatrixNode, PriorNode, EqualityNode
from ime_fgs.messages import GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage

"""
Simple example, written in Berlin, 10.10.2017
"""

if __name__ == '__main__':

    # Initialize variables
    A = [[2]]

    # Initialize all relevant nodes
    A_node = MatrixNode(A)
    meas_node = PriorNode(GaussianMeanCovMessage([[0]], [[1e-1]]).convert(GaussianWeightedMeanInfoMessage))
    input_node = PriorNode(GaussianMeanCovMessage([[1]], [[2]]))
    output_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))
    equality_node = EqualityNode()

    # Connect the nodes
    input_node.port_a.connect(A_node.port_a)
    A_node.port_b.connect(equality_node.ports[0])
    meas_node.port_a.connect(equality_node.ports[1])
    equality_node.ports[2].connect(output_node.port_a)

    # Perform message passing
    A_node.port_b.update(GaussianWeightedMeanInfoMessage)
    equality_node.ports[2].update(GaussianWeightedMeanInfoMessage)

    # Marginalize
    result = equality_node.ports[2].out_msg.combine(
        equality_node.ports[2].in_msg.convert(GaussianWeightedMeanInfoMessage))
    print(result.convert(GaussianMeanCovMessage))
