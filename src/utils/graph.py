import math
import random
from collections import deque
from itertools import combinations
from typing import Tuple

import networkx as nx
import numpy as np


def remove_isolated_nodes(graph: nx.Graph) -> nx.Graph:
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph


def keep_only_largest_component(graph: nx.Graph) -> nx.Graph:
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    return graph.subgraph(components[0])


def get_neighborhood(graph: nx.Graph, node: int):
    return list(graph.edges(node))


def sample_from_neighborhood(graph: nx.Graph, node: int, rate: float, std=None):
    neighborhood = get_neighborhood(graph, node)
    num_neighbors = len(neighborhood)
    rate = min(max(rate, 0.0), 1.0)
    if rate == 0.0:
        return []
    if rate == 1.0:
        return neighborhood
    mean = math.ceil(num_neighbors * rate)
    # use binomial distribution as default
    std = int(np.sqrt(num_neighbors * rate * (1 - rate))
              ) if std is None else std
    amount = int(np.random.normal(mean, std))
    amount = min(max(amount, 0), num_neighbors)
    return random.sample(neighborhood, amount)


def num_nodes_reached_in_k_hops(g: nx.Graph, x: int, k: int = 2):
    """
    Computes the number of nodes reached from `x` in a `k`-hop network in `g`.

    Parameters:
    - g (networkx.Graph): the graph
    - x: the node to start the search from
    - k: the maximum number of hops

    Returns:
    - num_reached (int): the number of nodes reached from `x` in a `k`-hop network in `g`
    """
    # Initialize a set to keep track of nodes visited during the search
    visited = set()
    # Initialize a queue for the breadth-first search
    queue = deque()
    # Add the starting node to the queue and mark it as visited
    queue.append(x)
    visited.add(x)

    # Perform the breadth-first search up to `k` hops
    for _ in range(k):
        # Get the number of nodes in the queue before processing the next level
        num_nodes_current_level = len(queue)
        # Process all nodes in the current level
        for _ in range(num_nodes_current_level):
            # Get the next node from the queue
            current = queue.popleft()
            # Check all neighbors of the current node
            for neighbor in g.neighbors(current):
                # If the neighbor has not been visited yet, add it to the queue and mark it as visited
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
    return len(visited)


def remove_edge_but_keep_if_would_disconnect_graph(
    graph: nx.Graph, edge: tuple
) -> bool:
    """
    Removes an edge from a graph, but only if removing the edge would not disconnect the graph.

    Parameters:
    - graph (networkx.Graph): the graph
    - edge (tuple): the edge to remove

    Returns:
    - graph (networkx.Graph): the graph with the edge removed, if removing the edge would not disconnect the graph
    """
    # Remove the edge
    graph.remove_edge(*edge)
    # Check if the graph is still connected
    if not nx.is_connected(graph):
        # If the graph is not connected, add the edge back
        graph.add_edge(*edge)
        return False
    return True


def remove_node_but_keep_if_would_disconnect_graph(graph: nx.Graph, node: int) -> bool:
    """
    Removes a node from a graph, but only if removing the node would not disconnect the graph.

    Parameters:
    - graph (networkx.Graph): the graph
    - node (int): the node to remove

    Returns:
    - graph (networkx.Graph): the graph with the node removed, if removing the node would not disconnect the graph
    """
    edges = list(graph.edges(node))
    # Remove the node
    graph.remove_node(node)
    # Check if the graph is still connected
    if not nx.is_connected(graph):
        # If the graph is not connected, add the node back
        graph.add_node(node)
        for edge in edges:
            graph.add_edge(*edge)
        return False
    return True


def sample_node_based_on_centrality_quantile(
    graph: nx.Graph, quantile: float, compute_centrality=nx.degree_centrality
) -> int:
    """
    Picks a random node in a graph, but pick a node that has degree centrality in the x-th quantile.
    :param graph: the input graph
    :param quantile: the quantile value (between 0 and 1)
    :return: a random node with degree centrality in the x-th quantile
    """
    centralities = compute_centrality(graph)
    threshold = np.quantile(list(centralities.values()), quantile)
    node = None
    items = list(centralities.items())
    random.shuffle(items)
    for n, v in items:
        if v >= threshold:
            node = n
    return node


def sample_subnetwork(
    graph: nx.Graph, amount: float, root=None, complete=True
) -> nx.Graph:
    """Sample a subnetwork from a given graph with at most x% of the nodes.

    Args:
        graph: The input graph to sample from.
        amount: The maximum percentage of nodes to include in the subnetwork (between 0 and 1).

    Returns:
        A subnetwork sampled from the input graph, containing at most x% of the original nodes.

    Raises:
        ValueError: If the input value of x is outside the range (0, 1).

    """

    # Validate input value of x
    if amount <= 0 or amount >= 1:
        raise ValueError("Value of x must be between 0 and 1, exclusive.")
    # Determine the maximum number of nodes allowed in the subnetwork
    max_nodes = int(len(graph.nodes()) * amount)

    # If the desired x% results in a subnetwork with fewer than 2 nodes, return the original graph
    if max_nodes < 2:
        return graph

    # Sample a random node from the graph
    node = root if root is not None else random.choice(list(graph.nodes()))

    # Sample nodes up to the maximum number of nodes
    subgraph_nodes = set([node])
    subgraph_edges = set([])
    while len(subgraph_nodes) < max_nodes:
        # Sample a random neighbor of the current subgraph nodes
        node = random.choice(list(subgraph_nodes))
        neighbor_edges = get_neighborhood(graph, node)
        if len(neighbor_edges) == 0:
            break
        edge = random.choice(list(neighbor_edges))
        # Add the neighbor to the subgraph nodes
        if edge[1] not in subgraph_nodes:
            subgraph_nodes.add(edge[1])
            subgraph_edges.add(edge)
    if complete:
        # Create a subgraph using the sampled nodes and their edges
        subgraph = nx.subgraph(graph, subgraph_nodes)
    else:
        subgraph = nx.Graph()
        subgraph.add_nodes_from(subgraph_nodes)
        subgraph.add_edges_from(subgraph_edges)

    return subgraph


def count_shortest_paths(G: nx.Graph, source: int):
    shortest_path_lengths = nx.shortest_path_length(G, source=source)
    path_length_counts = {}

    for _, path_length in shortest_path_lengths.items():
        if path_length not in path_length_counts:
            path_length_counts[path_length] = 1
        else:
            path_length_counts[path_length] += 1

    return path_length_counts


def relabel_nodes_consecutive(graph: nx.Graph) -> Tuple[nx.Graph, dict]:
    mapping = {
        old_label: new_label for new_label, old_label in enumerate(graph.nodes())
    }
    return nx.relabel_nodes(graph, mapping), mapping


def find_missing_edges(G):
    nodes = G.nodes()
    all_possible_edges = set(combinations(nodes, 2))
    existing_edges = set(G.edges())

    # Create a set of reverse edges
    reverse_edges = set((v, u) for u, v in existing_edges)
    existing_edges = existing_edges.union(reverse_edges)

    missing_edges = all_possible_edges - existing_edges

    return list(missing_edges)
