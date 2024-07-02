import copy

import networkx as nx
import numpy as np

from src.perturbations.perturbation import BehaviorPerturbation
from src.utils.graph import remove_edge_but_keep_if_would_disconnect_graph, find_missing_edges


class EdgeRewiringPerturbation(BehaviorPerturbation):
    def __init__(self, rewiring_amount=0.1, increase=True, **kwargs):
        self.rewiring_amount = rewiring_amount
        self.increase = increase
        super().__init__(**kwargs)

    def compute_metric(self, graph: nx.Graph) -> dict:
        raise NotImplementedError(
            "Perturbation.compute_metric() is not implemented.")

    def get_metric_mean(self, new_metric) -> float:
        return (
            np.mean(list(new_metric.values()))
            if type(new_metric) is dict
            else new_metric
        )

    def delete_edges(self, propagation_graph: nx.Graph, behavior_graph: nx.Graph, base_value: float, amount: int):
        actual_deletions = 0
        # for each edge check how much does the avg change if we would remove it
        changes = []
        edges = (
            list(
                filter(
                    lambda edge: not propagation_graph.has_edge(*edge),
                    list(behavior_graph.edges()),
                )
            )
            if self.posterior
            else list(behavior_graph.edges())
        )
        for edge in edges:
            behavior_graph.remove_edge(*edge)
            new_value = self.get_metric_mean(
                self.compute_metric(behavior_graph))
            changes.append((edge, new_value - base_value))
            behavior_graph.add_edge(*edge)
        # either remove edges that cause highest decrease or highest increase
        sorted_list = sorted(
            changes, key=lambda x: x[1], reverse=self.increase)
        while actual_deletions < amount:
            if len(sorted_list) == 0:
                break
            edge = sorted_list.pop(0)[0]
            removed_edge = remove_edge_but_keep_if_would_disconnect_graph(
                behavior_graph, edge
            )
            if removed_edge:
                actual_deletions += 1

    def add_edges(self, propagation_graph: nx.Graph, behavior_graph: nx.Graph, base_value: float, amount: int):
        actual_additions = 0
        # for each edge check how much does the avg change if we would add it
        changes = []
        for edge in find_missing_edges(behavior_graph):
            behavior_graph.add_edge(*edge)
            new_value = self.get_metric_mean(
                self.compute_metric(behavior_graph))
            changes.append((edge, new_value - base_value))
            behavior_graph.remove_edge(*edge)
        # either add edges that cause highest decrease or highest increase
        sorted_list = sorted(
            changes, key=lambda x: x[1], reverse=self.increase)
        while actual_additions < amount:
            if len(sorted_list) == 0:
                break
            edge = sorted_list.pop(0)[0]
            behavior_graph.add_edge(*edge)
            actual_additions += 1

    def perturb(self):
        behavior_graph_copy = copy.deepcopy(
            self.propagation.behavior_graph.snapshots[0]
        )
        propagation_graph_copy = copy.deepcopy(self.propagation.snapshots[-1])
        num_rewirings = int(len(behavior_graph_copy.edges())
                            * self.rewiring_amount)
        base_value = self.get_metric_mean(
            self.compute_metric(behavior_graph_copy))

        self.delete_edges(propagation_graph_copy,
                          behavior_graph_copy, base_value, num_rewirings)
        self.add_edges(propagation_graph_copy,
                       behavior_graph_copy, base_value, num_rewirings)

        return [behavior_graph_copy for _ in self.propagation.behavior_graph.snapshots]


class AssortativityEdgeRewiring(EdgeRewiringPerturbation):
    def __init__(self, **kwargs):
        super().__init__(
            name=f"assortativity_{'increase' if kwargs['increase'] else 'decrease'}",
            **kwargs,
        )

    def compute_metric(self, graph: nx.Graph) -> dict:
        return nx.degree_assortativity_coefficient(graph)


class ClusteringEdgeRewiring(EdgeRewiringPerturbation):
    def __init__(self, **kwargs):
        super().__init__(
            name=f"clustering_{'increase' if kwargs['increase'] else 'decrease'}",
            **kwargs,
        )

    def compute_metric(self, graph: nx.Graph) -> dict:
        clustering_coeffs = nx.clustering(graph)
        return sum(clustering_coeffs.values()) / len(clustering_coeffs)
