# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
from ime_fgs.base import NodePort
from numpy import atleast_2d

from ime_fgs.base import Node, NodePortType
from ime_fgs.messages import GaussianMixtureMeanCovMessage, MultipleCombineMessage
from ime_fgs.gaussian_mixture_reduction import reduction_algorithm


class EqualityNode(Node):
    """
                +---+
     ports[0]   | = |
    ------------|   |
                |   |
     ports[1]   |   |
    ------------|   |
                |   |
                |   |
    """

    def __init__(self, name=None, number_of_ports=3, allow_unconnected_ports=False):
        super().__init__(name=name)

        # prohibit degenerated equal nodes
        assert number_of_ports >= 2
        self.allow_unconnected_ports = allow_unconnected_ports
        # TODO: Assign in or out direction to ports. How?
        if self.allow_unconnected_ports:
            self.ports = [NodePort(self, lambda i=i: self._calc_msg_unconnected(i)) for i in range(number_of_ports)]
        else:
            self.ports = [NodePort(self, lambda i=i: self._calc_msg(i)) for i in range(number_of_ports)]

    def _calc_msg(self, index):
        message_list = [port.in_msg for port in self.ports if (port is not self.ports[index])]
        return self._calc_result(message_list)

    def _calc_msg_unconnected(self, index):
        message_list = [port.in_msg for port in self.ports if
                        ((port is not self.ports[index]) and (port.connected))]
        return self._calc_result(message_list)

    @staticmethod
    def _calc_result(message_list):
        # try to use shortcut to reduce object creations
        if all(isinstance(msg, type(message_list[0])) for msg in message_list) and\
                isinstance(message_list[0], MultipleCombineMessage):
            return message_list[0].combine_multiple(message_list)
        else:
            it = iter(message_list)

            result = next(it)
            for msg in it:
                result = result.combine(msg)

            return result

    def get_ports(self):
        if not self.allow_unconnected_ports:
            return list(self.ports)
        else:
            return [port for port in self.ports if port.connected]


class AdditionNode(Node):
    """
      a +---+ c
    --->| + |--->
        +---+
          ^
          |
          | b
    """

    def __init__(self, name=None):
        super().__init__(name=name)

        self.port_a = NodePort(self, self._calc_msg_a, NodePortType.InPort)
        self.port_b = NodePort(self, self._calc_msg_b, NodePortType.InPort)
        self.port_c = NodePort(self, self._calc_msg_c_, NodePortType.OutPort)

    def _calc_msg_a(self):
        return self.port_c.in_msg - self.port_b.in_msg

    def _calc_msg_b(self):
        return self.port_c.in_msg - self.port_a.in_msg

    def _calc_msg_c_(self):
        return self.port_a.in_msg + self.port_b.in_msg

    def get_ports(self):
        return [self.port_a, self.port_b, self.port_c]


class BigAdditionNode(Node):
    """
                   +---+
     ports_in[0]   | + | ports_out[0]
    -------------->|   |------------->
                   |   |
     ports_in[1]   |   | ports_out[1]
    -------------->|   |------------->
                   |   |
                   |   |
    """

    def __init__(self, number_of_in_ports, number_of_out_ports, name=None):
        super().__init__(name=name)

        self.ports_in = [NodePort(self, lambda i=i: self._calc_msg_in(i), NodePortType.InPort)
                         for i in range(number_of_in_ports)]
        self.ports_out = [NodePort(self, lambda i=i: self._calc_msg_out(i), NodePortType.OutPort)
                          for i in range(number_of_out_ports)]
        # prohibit degenerated addition nodes
        assert number_of_in_ports + number_of_out_ports >= 3

    def _calc_msg_in(self, index):
        msg_list = [-port.in_msg for port in self.ports_in if port is not self.ports_in[index]] + \
                   [port.in_msg for port in self.ports_out]

        return self._msg_calc(msg_list)

    def _calc_msg_out(self, index):
        msg_list = [port.in_msg for port in self.ports_in] + \
                   [-port.in_msg for port in self.ports_out if port is not self.ports_out[index]]

        return self._msg_calc(msg_list)

    @staticmethod
    def _msg_calc(msg_list):
        it = iter(msg_list)

        result = next(it)

        for msg in it:
            result += msg
        return result

    def get_ports(self):
        return self.ports_in + self.ports_out


class MatrixNode(Node):
    """
      a +--------+ b
    --->| matrix |--->
        +--------+
    """

    def __init__(self, matrix, name=None):
        super().__init__(name=name)

        self.port_a = NodePort(self, self._calc_msg_a, NodePortType.InPort)
        self.port_b = NodePort(self, self._calc_msg_b, NodePortType.OutPort)
        self._matrix = atleast_2d(matrix)

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        self._matrix = atleast_2d(matrix)

    def _calc_msg_a(self):
        return self.port_b.in_msg.multiply_deterministic(self.matrix, inverse=True)

    def _calc_msg_b(self):
        return self.port_a.in_msg.multiply_deterministic(self.matrix)

    def __repr__(self):
        if self.name is None:
            name = str(id(self))
        else:
            name = self.name

        return type(self).__name__ + "(" + name + ", Matrix:" + repr(self.matrix.tolist()) + ")"

    def get_ports(self):
        return [self.port_a, self.port_b]

class ReductionNode(Node):
    """
      a +---------------+ b
    --->| ReductionNode |--->
        +---------------+
    """

    def __init__(self, num_weights, name=None):
        super().__init__(name=name)

        self.port_a = NodePort(self, self._calc_msg_a, NodePortType.InPort)
        self.port_b = NodePort(self, self._calc_msg_b, NodePortType.OutPort)
        self._num_weights = num_weights

    @property
    def num_weights(self):
        return self._num_weights

    @num_weights.setter
    def num_weights(self, num_weights):
        self._num_weights = num_weights

    def _calc_msg_a(self):
        if isinstance(self.port_b.in_msg, GaussianMixtureMeanCovMessage):
            return reduction_algorithm(self.port_b.in_msg, self._num_weights)
        else:
            return self.port_b.in_msg

    def _calc_msg_b(self):
        if isinstance(self.port_a.in_msg, GaussianMixtureMeanCovMessage):
            return reduction_algorithm(self.port_a.in_msg, self._num_weights)
        else:
            return self.port_a.in_msg

    def __repr__(self):
        if self.name is None:
            name = str(id(self))
        else:
            name = self.name

        return type(self).__name__ + "(" + name + ", Num Weights:" + self._num_weights + ")"

    def get_ports(self):
        return [self.port_a, self.port_b]


class PriorNode(Node):
    """
    +---+ a
    |   |---
    +---+
     msg
    """

    def __init__(self, msg=None, name=None, target_type=None):
        super().__init__(name=name)

        self._prior = msg
        self.port_a = NodePort(self, self._update_port_a, target_type=target_type)
        if msg is not None:
            self.port_a.update()

    def update_prior(self, msg, target_type=None):
        self._prior = msg
        self.port_a.target_type = target_type
        self.port_a.update()

    def __repr__(self):
        if self.name is None:
            name = str(id(self))
        else:
            name = self.name

        return type(self).__name__ + '("' + name + '", Prior:' + repr(self._prior) + ")"

    def _update_port_a(self):
        return self._prior

    def get_ports(self):
        return [self.port_a]
