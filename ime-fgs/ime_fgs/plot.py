# For research and educational use only, do not distribute without permission.
#
# Created by:
# Eike Petersen, Felix Vollmer with contribution from institute members and students
#
# University of Lübeck
# Institute for Electrical Engineering in Medicine
#

from ime_fgs.base import Node
import networkx as nx
import matplotlib.pyplot as plt
from typing import Set, Tuple


def draw_graph(node_of_graph: Node) -> None:
    """
    Draws graph of a factorgraph. Assumes that all nodes in the graph implement get_ports()
    @param node_of_graph: a single node of the graph
    """
    nodes, edges = get_all_nodes_and_edges(node_of_graph)

    # create graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # create map to label nodes with their name
    node_labels = {node: node.name for node in nodes}

    # plot graph
    plt.figure()
    nx.draw(G, with_labels=True, labels=node_labels)
    plt.show()


def get_all_nodes_and_edges(node_of_graph: Node) -> Tuple[Set[Node], Set[Set[Node]]]:
    """
    Find all nodes and edges of a graph. Assumes that all nodes in the graph implement get_ports()
    @param node_of_graph: a single node of the graph
    @return: set of nodes and edges
    """
    nodes = set()
    edges = set()
    to_visit = {node_of_graph}
    # walk through all connected nodes
    while to_visit:
        node = to_visit.pop()
        ports = node.get_ports()

        if ports is None:
            # skip if we can't get any ports
            continue

        connected_nodes = {port.other_port.parent_node for port in ports if port.other_port}
        edges = edges.union({frozenset((node, connected_node)) for connected_node in connected_nodes})

        nodes.add(node)
        to_visit = to_visit.union((connected_nodes - nodes))
    return nodes, edges
