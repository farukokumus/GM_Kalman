# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
import numpy as np

from ime_fgs.basic_nodes import PriorNode, BigAdditionNode, MatrixNode
from ime_fgs.messages import GaussianMeanCovMessage, GaussianWeightedMeanInfoMessage, Message
from ime_fgs.base import NodePort, Node


class PredictionNode(Node):
    """
              +--------------------------+
              |         +---+            |
              |         |   | noise      |
              |         +---+            |
              |           |              |
              |           v              |
     state_in |         +---+            | state_out
    ----------|-------->| + |------------|---------->
              |         +---+            |
              |           ^              |
              |           |              |
              | speed_times_elapsed_time |
              +--------------------------+
    """

    def __init__(self, speed_times_elapsed_time, noise, name=None):
        super().__init__(name=name)

        # todo allow non 2d case

        # node ports
        self.port_state_in = NodePort(self, self.calc_msg_state_in)
        self.port_state_out = NodePort(self, self.calc_msg_state_out, Message)

        # internal nodes
        self.__noise_node__ = PriorNode(noise)
        self.cov_speed_times_elapsed_time = 1e-9 * np.identity(len(speed_times_elapsed_time))
        self.__speed_times_elapsed_time_node__ = PriorNode(GaussianMeanCovMessage(speed_times_elapsed_time,
                                                                                  self.cov_speed_times_elapsed_time))

        self.cov_state_node = 1e9 * np.identity(len(speed_times_elapsed_time))
        self.mean_state_node = len(speed_times_elapsed_time) * [0]
        self.__state_in_node__ = PriorNode(GaussianMeanCovMessage(self.mean_state_node, self.cov_state_node))
        self.__state_out_node__ = PriorNode(GaussianMeanCovMessage(self.mean_state_node, self.cov_state_node))

        self.__add_node__ = BigAdditionNode(3, 1)

        # connect internal nodes
        self.__add_node__.ports_in[2].connect(self.__noise_node__.port_a)
        self.__add_node__.ports_in[1].connect(self.__speed_times_elapsed_time_node__.port_a)
        self.__add_node__.ports_in[0].connect(self.__state_in_node__.port_a)

        self.__add_node__.ports_out[0].connect(self.__state_out_node__.port_a)

    def calc_msg_state_in(self):
        self.__state_out_node__.update_prior(self.port_state_out.in_msg)
        return self.__add_node__.ports_in[0].update()

    def calc_msg_state_out(self):
        self.__state_in_node__.update_prior(self.port_state_in.in_msg)
        return self.__add_node__.ports_out[0].update(GaussianMeanCovMessage)

    def update_speed(self, speed_times_elapsed_time):
        self.__speed_times_elapsed_time_node__.update_prior(GaussianMeanCovMessage(speed_times_elapsed_time,
                                                                                   self.cov_speed_times_elapsed_time))

    def get_ports(self):
        return [self.port_state_in, self.port_state_out]


class DistanceMeasurementNode(Node):
    """
                | port_state_xi
    +---------------------------+
    |           |               |
    |           v               |
    |        +-----+            |
    |        | Bji |            |
    |        +-----+            |
    |          1|               |
    |           v               |
    | +---+ 2 +---+             |
    | |   |-->| + |             |
    | +---+   |   | 0           |
    | noise   |   |--> distance |
    |       0 |   |             |
    | Dji  -->|   |             |
    |         +---+             |
    |           ^               |
    |          3|               |
    |        +-----+            |
    |        |-Bji |            |
    |        +-----+            |
    |           ^               |
    |           |               |
    +---------------------------+
                | port_state_xj
    """

    def __init__(self, distance: object, noise: object, name: object = None) -> object:
        super().__init__(name=name)

        # node ports
        self.port_state_xi = NodePort(self, self.calc_port_state_xi)
        self.port_state_xj = NodePort(self, self.calc_port_state_xj)

        # internal nodes
        self.state_node_xi = PriorNode(GaussianMeanCovMessage([0, 0], [[10e9, 0], [0, 10e9]]))
        self.state_node_xj = PriorNode(GaussianMeanCovMessage([0, 0], [[10e9, 0], [0, 10e9]]))

        self.distance_node = PriorNode(GaussianMeanCovMessage(distance, 1e-9))  # todo set to 0
        self.noise_node = PriorNode(noise)
        self.constant_node = PriorNode(GaussianMeanCovMessage(0, 1e-9))  # todo set to 0

        self.matrix_node_Bji = MatrixNode(np.identity(2))
        self.matrix_node_nBji = MatrixNode(np.identity(2))

        self.add_node = BigAdditionNode(4, 1)

        # connect internal nodes
        self.add_node.ports_in[0].connect(self.constant_node.port_a)
        self.add_node.ports_in[1].connect(self.matrix_node_Bji.port_b)
        self.add_node.ports_in[2].connect(self.noise_node.port_a)
        self.add_node.ports_in[3].connect(self.matrix_node_nBji.port_b)

        self.add_node.ports_out[0].connect(self.distance_node.port_a)

        self.matrix_node_Bji.port_a.connect(self.state_node_xi.port_a)
        self.matrix_node_nBji.port_a.connect(self.state_node_xj.port_a)

    def update_constants(self, predicted_xi, predicted_xj):
        # calculate updated constants
        d = np.atleast_2d(np.asarray(predicted_xi) - np.asarray(predicted_xj))
        dji = np.linalg.norm(d)
        Bji = d.T / dji
        Dji = dji - Bji @ d

        # update constants
        self.constant_node.update_prior(GaussianMeanCovMessage(Dji, 10e-9))  # todo set 0
        self.matrix_node_Bji.matrix = Bji
        self.matrix_node_nBji.matrix = -Bji

    def update_distance(self, distance):
        self.distance_node.update_prior(GaussianMeanCovMessage(distance, 1e-9))  # todo set to 0

    def calc_port_state_xi(self):
        self.state_node_xj.update_prior(self.port_state_xj.in_msg)
        self.matrix_node_nBji.port_b.update()
        self.add_node.ports_in[1].update(GaussianWeightedMeanInfoMessage)
        return self.matrix_node_Bji.port_a.update()

    def calc_port_state_xj(self):
        self.state_node_xi.update_prior(self.port_state_xi.in_msg)
        self.matrix_node_Bji.port_b.update()
        self.add_node.ports_in[3].update(GaussianWeightedMeanInfoMessage)
        return self.matrix_node_nBji.port_a.update()

    def get_ports(self):
        return [self.port_state_xi, self.port_state_xj]
