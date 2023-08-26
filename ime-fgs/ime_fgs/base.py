# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
from abc import ABC, abstractmethod
from enum import Enum
from ime_fgs.messages import PortMessageDirection


class NodePortType(Enum):
    InPort = 0
    OutPort = 1
    InOutPort = 2


class NodePort(object):

    def __init__(self, node, update_method, port_type=NodePortType.InOutPort, target_type=None, name=None):
        self.name = name
        self.parent_node = node
        self.other_port = None
        self.out_msg = None
        self.cached_out_msg = None
        self.update_method = update_method
        self._connected = False
        self._value_in_cache = False
        self._target_type = None

        self.target_type = target_type

        assert isinstance(port_type, NodePortType)
        self.port_type = port_type

    def apply_cached_out_message(self):
        if self._value_in_cache:
            self.out_msg = self.cached_out_msg
            self.cached_out_msg = None
        else:
            raise Exception("No value in cache!")

    @property
    def target_type(self):
        return self._target_type

    @target_type.setter
    def target_type(self, target_type):
        if target_type != self._target_type:
            self.out_msg = None
        self._target_type = target_type

    @property
    def connected(self):
        return self._connected

    @property
    def in_msg(self):
        if self._connected:
            return self.other_port.out_msg
        else:
            raise ConnectionError('Unconnected port ' + repr(self.name) + 'of node ' + repr(self.parent_node))

    @in_msg.setter
    def in_msg(self, msg):
        if self._connected:
            self.other_port.out_msg = msg
        else:
            raise ConnectionError('Unconnected port ' + repr(self.name) + 'of node ' + repr(self.parent_node))

    def _update(self, **kwargs):
        return self.update_method(**kwargs)

    def update(self, target_type=None, cached=False, **kwargs):
        if __debug__:
            self._check_other_ports()

        out_msg = self._update(**kwargs)

        if out_msg.direction == PortMessageDirection.Undefined:
            if self.port_type == NodePortType.InPort:
                out_msg.direction = PortMessageDirection.Backward
            elif self.port_type == NodePortType.OutPort:
                out_msg.direction = PortMessageDirection.Forward

        if target_type is not None:
            self.target_type = target_type

        if self.target_type is not None:
            out_msg = out_msg.convert(self.target_type)

        if not cached:
            self.out_msg = out_msg
        else:
            self._value_in_cache = True
            self.cached_out_msg = out_msg

        return out_msg

    def _check_other_ports(self):
        """
        check whether all other ports of the node have incoming messages. Throws Runtime Error if not.
        """
        for port in self.parent_node.get_ports():
            if port is not self and port.in_msg is None:
                raise RuntimeError('Missing incoming message for port ' + repr(self)
                                   + ' of node' + repr(self.parent_node))

    def connect(self, other_port, out_type=None, in_type=None):
        """
        Connect port with another port together. The forward and backward message types can be specified. If the message
        type changes, the corresponding message is set to None. If one or both of the two ports does not have a
        predefined directionality (in/out), this action also assigns one. If neither of the two ports has a predefined
        directionality (i.e., both are NodePortType.InOutPorts), then the directionality of the edge is set to "self ->
        other".
        :param other_port: the other port to connect to
        :param out_type: optional type parameter
        :param in_type: optional type parameter
        """
        if self.connected or other_port.connected:
            raise ConnectionRefusedError('Trying to connect already connected port ' + repr(self)
                                         + ' of node' + repr(self.parent_node))

        self.other_port = other_port
        other_port.other_port = self

        self._connected = True
        self.other_port._connected = True

        if out_type is not None:
            self.target_type = out_type
        if in_type is not None:
            self.other_port.target_type = in_type

    def disconnect(self):
        if not self.connected:
            raise ConnectionRefusedError('Trying to disconnect already disconnected port ' + repr(self)
                                         + ' of node' + repr(self.parent_node))

        self.other_port.other_port = None
        self.other_port._connected = False
        self.other_port = None
        self._connected = False

    def __repr__(self):
        parent = repr(self.parent_node)

        try:
            in_msg = repr(self.in_msg)
        except ConnectionError:
            in_msg = repr(None)

        out_msg = repr(self.out_msg)

        if self.name is None:
            name = "NodePort"
        else:
            name = self.name

        return name + "(" + str(self.port_type) + ", Parent:" + parent + ", In: " + in_msg + ", Out: " + out_msg + ")"

    def marginal(self, target_type=None):
        if self.in_msg is not None:
            if self.out_msg is not None:
                marg = self.in_msg.combine(self.out_msg, auto_convert=True)
            else:
                marg = self.in_msg
        else:
            if self.out_msg is not None:
                marg = self.out_msg
            else:
                raise RuntimeError('Marginal requested but both edge messages are None.')

        if target_type is None:
            return marg
        else:
            return marg.convert(target_type)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


class Node(ABC):
    def __init__(self, name=None):
        self.name = name

    @abstractmethod
    def get_ports(self):
        pass

    def __repr__(self):
        if self.name is None:
            name = str(id(self))
        else:
            name = self.name

        return type(self).__name__ + "(" + name + ")"
