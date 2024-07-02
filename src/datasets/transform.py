import networkx as nx
from torch_geometric.data import Data
import torch

from src.generation.propagation.propagation import Propagation, PropagationMetrics
from src.utils.objects import pad_list


class PropagationGraphTransform:
    def __init__(self, name: str) -> None:
        self.name = name

    def transform(self, propagation: Propagation, max_t=None) -> Data:
        raise NotImplementedError(
            "This method must be implemented by a subclass")


class TemporalGraphAttributesTransform(PropagationGraphTransform):
    def __init__(self) -> None:
        super().__init__(name="TemporalGraphAttributesTransform")

    def transform(self, propagation: Propagation, max_t=None) -> Data:
        graph = propagation.behavior_graph.snapshots[-1]

        def get_node_attrs(node):
            return pad_list(
                [
                    1.0 if snapshot.has_node(node) else 0.0
                    for snapshot in propagation.snapshots
                ],
                max_t,
                0,
            )

        def get_edge_attrs(edge):
            return pad_list(
                [
                    1.0 if snapshot.has_edge(*edge) else 0.0
                    for snapshot in propagation.snapshots
                ],
                max_t,
                0,
            )

        return networkx_to_pyg_data(
            G=graph, node_attr_fn=get_node_attrs, edge_attr_fn=get_edge_attrs
        )


class PropagationMetricsGraphTransform(PropagationGraphTransform):
    def __init__(self) -> None:
        super().__init__(name=" PropagationMetricsGraphTransform")

    def transform(self, propagation: Propagation, max_t=None) -> Data:
        graph = propagation.behavior_graph.snapshots[-1]
        node_metrics = PropagationMetrics(
            propagation=propagation, local=True)._
        node_attrs = {}
        for _, values in node_metrics.items():
            for node_idx, value in enumerate(values):
                if node_idx not in node_attrs:
                    node_attrs[node_idx] = []
                node_attrs[node_idx].append(value)

        def get_node_attrs(node):
            return node_attrs[node]

        def get_edge_attrs(edge):
            return [1. if propagation.snapshots[-1].has_edge(*edge) else 0.]

        return networkx_to_pyg_data(
            G=graph, node_attr_fn=get_node_attrs, edge_attr_fn=get_edge_attrs
        )


class TemporalSnapshotListTransform(PropagationGraphTransform):
    def __init__(self) -> None:
        super().__init__(name="TemporalSnapshotListTransform")

    def transform(self, propagation: Propagation, max_t=None) -> Data:
        # placeholder data for combined entry
        data = networkx_to_pyg_data(nx.complete_graph(10))
        data.snapshots = []

        all_nodes = set()
        for snapshot in propagation.behavior_graph.snapshots:
            all_nodes.update(snapshot.nodes())

        def get_node_attrs(node, t, node_diff):
            return [
                1.0 if propagation.node_is_covered(node, t) else 0.0,
                1.0 if node in node_diff else 0.0,
                1.0 if propagation.node_is_uncovered_but_exposed(
                    node, t) else 0.0,
                1.0 if node in [edge[0] for edge in edge_diff] else 0.0,
            ]

        def get_edge_attrs(edge, t, edge_diff):
            return [
                1.0 if propagation.snapshots[t].has_edge(*edge) else 0.0,
            ]

        for t, snapshot in enumerate(propagation.behavior_graph.snapshots):
            snapshot_copy = nx.Graph()
            snapshot_copy.add_nodes_from(all_nodes)
            snapshot_copy.add_edges_from(snapshot.edges())
            node_diff, edge_diff = propagation.diff(t - 1, t)

            data.snapshots.append(
                networkx_to_pyg_data(
                    G=snapshot_copy,
                    node_attr_fn=lambda node: get_node_attrs(
                        node, t, node_diff),
                    edge_attr_fn=lambda edge: get_edge_attrs(
                        edge, t, edge_diff),
                )
            )

        return data


class TemporalSnapshotListOneHotTransform(PropagationGraphTransform):
    def __init__(self) -> None:
        super().__init__(name="TemporalSnapshotListOneHotTransform")

    def transform(self, propagation: Propagation, max_t=None) -> Data:
        # placeholder data for combined entry
        data = networkx_to_pyg_data(nx.complete_graph(10))
        data.snapshots = []

        all_nodes = set()
        for snapshot in propagation.behavior_graph.snapshots:
            all_nodes.update(snapshot.nodes())

        def get_node_attrs(node, t, node_diff):
            # Determine if the node is covered at time t
            covered = propagation.node_is_covered(node, t)

            # Initialize a list of zeros of length max_t
            attrs = [0] * max_t

            # If the node is covered at time t, set the t-th entry to 1
            if covered:
                attrs[t] = 1

            return attrs

        def get_edge_attrs(edge, t, edge_diff):
            covered = propagation.snapshots[t].has_edge(*edge)
            # Initialize a list of zeros of length max_t
            attrs = [0] * max_t

            # If the node is covered at time t, set the t-th entry to 1
            if covered:
                attrs[t] = 1

            return attrs

        for t, snapshot in enumerate(propagation.behavior_graph.snapshots):
            snapshot_copy = nx.Graph()
            snapshot_copy.add_nodes_from(all_nodes)
            snapshot_copy.add_edges_from(snapshot.edges())
            node_diff, edge_diff = propagation.diff(t - 1, t)

            data.snapshots.append(
                networkx_to_pyg_data(
                    G=snapshot_copy,
                    node_attr_fn=lambda node: get_node_attrs(
                        node, t, node_diff),
                    edge_attr_fn=lambda edge: get_edge_attrs(
                        edge, t, edge_diff),
                )
            )

        return data


def networkx_to_pyg_data(G, node_attr_fn=None, edge_attr_fn=None):
    """Custom implementation of the from_networkx function from pytorch_geometric.utils, as the original implementation does some more sanity checks, that we don't need, as
    we are sure that the graph is valid. By removing these checks, we can speed up the conversion process.

    G: A NetworkX graph
    node_attr_fn: A function that accepts a NetworkX node and its data dictionary and returns a list or tensor of node attributes
    edge_attr_fn: A function that accepts a NetworkX edge and its data dictionary and returns a list or tensor of edge attributes
    """
    if node_attr_fn is None:
        def node_attr_fn(node): return []
    if edge_attr_fn is None:
        def edge_attr_fn(edge): return []

    # Initialize empty lists for edge_index, node_features, and edge_features
    edge_index = []
    node_features = []
    edge_features = []

    # Iterate through the nodes and append their attributes to the node_features list
    for node in G.nodes():
        node_features.append(node_attr_fn(node))

    # Iterate through the edges and append their attributes to the edge_index and edge_features list
    for edge in G.edges():
        # as our graphs are undirected, we need to add both directions in the edge_index
        reversed_edge = (edge[1], edge[0])
        edge_index.append(list(edge))
        edge_index.append(list(reversed_edge))
        edge_features.append(edge_attr_fn(edge))
        edge_features.append(edge_attr_fn(reversed_edge))

    # Convert lists to PyTorch tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    # Create a Data object
    data = Data(
        num_nodes=len(G.nodes()),
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features,
    )

    return data
