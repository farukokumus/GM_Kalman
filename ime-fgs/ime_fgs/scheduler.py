# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#
import random

# TODO this is WIP and contains a lot of broken stuff!!!


def get_port_list(node):
    unvisited_nodes = set([node])
    visited_nodes = set()
    ports = []

    # search the tree
    while unvisited_nodes:
        node = unvisited_nodes.pop()
        visited_nodes.add(node)

        new_ports = node.get_ports()
        new_nodes = set([port.other_port.parent_node for port in new_ports])

        unvisited_nodes |= new_nodes - visited_nodes

        ports += new_ports

    return ports


# flooding algorithm
class ScheduleSynchronous(object):
    def __init__(self, port_list):
        self.port_list = port_list

    def run(self, iterations):  # todo add other exit condition (e.g. function)
        for _ in range(iterations):
            updated = []
            for port in self.port_list:
                try:
                    port.update(cached=True)
                    updated.append(port)
                except RuntimeError as e:  # todo too broad
                    pass

            for port in updated:
                port.apply_cached_out_message()


class ScheduleAsynchronous(object):
    def __init__(self, port_list):
        self.port_list = port_list

    def run(self, iterations):  # todo add other exit condition (e.g. function)
        for _ in range(iterations):
            for port in self.port_list:
                try:
                    port.update()
                except RuntimeError:  # todo too broad
                    pass


class ScheduleRandom(object):
    def __init__(self, port_list):
        self.port_list = list(port_list)
        random.shuffle(self.port_list)

    def run(self, iterations):  # todo add other exit condition (e.g. function)
        for _ in range(iterations):
            for port in self.port_list:
                try:
                    port.update()
                except RuntimeError:  # todo too broad
                    pass

# Residual Belief Propagation
# todo: WIP


class ScheduleRBP(object):
    def __init__(self, port_list):
        self.port_distance_dict = dict.fromkeys(port_list)

    def run(self, iterations):  # todo add other exit condition (e.g. function)
        # update all ports (if possible)
        for port in self.port_distance_dict.keys():
            self._update(port)

        for _ in range(iterations):
            # apply update for message with greatest residual
            next_port = max(self.port_distance_dict, key=lambda key: self.port_distance_dict[key])  # optimize?
            next_port.apply_cached_out_message()
            self._update(next_port)

            # update dependent caches
            for port in next_port.other_port.parent_node.get_ports():
                # only work on given ports
                if port in self.port_distance_dict:
                    self._update(port)
                else:
                    # pass
                    assert False  # todo only for testing

    def _update(self, port):
        try:
            port.update(cached=True)
        except RuntimeError:  # todo broad
            pass

        if port.cached_out_msg is None:  # ignore ports, that can't be calculated
            distance = 0.0
        elif port.out_msg is None:  # prioritize uninitialized
            distance = float('Inf')
        else:  # get normal distance
            distance = port.cached_out_msg.distance(port.out_msg)

        self.port_distance_dict[port] = distance
