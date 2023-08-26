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
Simple example, written in Berlin, 10.10.2017
"""


class Slice(Node):

    def __init__(self, A):

        self.A_node = MatrixNode(A)
        self.equality_node = EqualityNode()

        # Initialize ports of the macro (slice) node
        self.port_in = NodePort(self, self.calc_msg_in)
        self.port_out = NodePort(self, self.calc_msg_out)
        self.port_meas = NodePort(self, self.calc_msg_meas)

        self.in_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))
        self.out_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))
        self.meas_node = PriorNode(GaussianMeanCovMessage([[0]], [[inf]]))

        # Connect nodes
        self.in_node.port_a.connect(self.A_node.port_a)
        self.A_node.port_b.connect(self.equality_node.ports[0])
        self.equality_node.ports[1].connect(self.meas_node.port_a)
        self.equality_node.ports[2].connect(self.out_node.port_a)

    def calc_msg_in(self):
        self.out_node.update_prior(self.port_out.in_msg, target_type=GaussianWeightedMeanInfoMessage)
        self.meas_node.update_prior(self.port_meas.in_msg, target_type=GaussianWeightedMeanInfoMessage)
        self.equality_node.ports[0].update(GaussianWeightedMeanInfoMessage)
        return self.A_node.port_a.update(GaussianMeanCovMessage)

    def calc_msg_out(self):
        self.in_node.update_prior(self.port_in.in_msg)
        self.meas_node.update_prior(self.port_meas.in_msg, target_type=GaussianWeightedMeanInfoMessage)
        self.A_node.port_b.update(GaussianWeightedMeanInfoMessage)
        return self.equality_node.ports[2].update(GaussianMeanCovMessage)

    def calc_msg_meas(self):
        self.in_node.update_prior(self.port_in.in_msg)
        self.out_node.update_prior(self.port_out.in_msg, target_type=GaussianWeightedMeanInfoMessage)
        self.A_node.port_b.update(GaussianWeightedMeanInfoMessage)
        return self.equality_node.ports[1].update(GaussianMeanCovMessage)

    def get_ports(self):
        return [self.port_in, self.port_out, self.port_meas]


if __name__ == '__main__':

    # Initialize variables
    A = np.matrix([2])

    # Initialize all relevant nodes
    meas_node = PriorNode(GaussianMeanCovMessage([[0]], [[1]]))
    input_node = PriorNode(GaussianMeanCovMessage([[1]], [[2]]))
    output_node = PriorNode(GaussianMeanCovMessage([[0]], [[1]]))
    slice_node = Slice(A)

    # Connect the nodes
    input_node.port_a.connect(slice_node.port_in)
    meas_node.port_a.connect(slice_node.port_meas)
    output_node.port_a.connect(slice_node.port_out)

    # Perform message passing
    result = slice_node.port_out.update()

    # Marginalize
    print(result.convert(GaussianWeightedMeanInfoMessage))
