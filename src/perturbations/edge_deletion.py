import copy
import random
from typing import Union

import networkx as nx
import numpy as np

from src.perturbations.perturbation import BehaviorPerturbation
from src.utils.graph import remove_edge_but_keep_if_would_disconnect_graph


class RandomEdgeDeletion(BehaviorPerturbation):
    def __init__(
        self,
        deletion_amount: Union[int, float] = 0.1,
        method="global",
        **kwargs,
    ):
        self.deletion_amount = deletion_amount
        self.method = method
        super().__init__(name=f"edge_deletion_{method}", **kwargs)

    def delete_global(self, propagation_graph: nx.Graph, behavior_graph: nx.Graph):
        num_deletions = int(len(behavior_graph.edges()) * self.deletion_amount)
        actual_deletions = 0
        while actual_deletions < num_deletions:
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
            edge = random.choice(edges)
            removed_edge = remove_edge_but_keep_if_would_disconnect_graph(
                behavior_graph, edge
            )
            if removed_edge:
                actual_deletions += 1
        return [
            behavior_graph
            for _ in range(len(self.propagation.behavior_graph.snapshots))
        ]

    def delete_linear(self, propagation_graph: nx.Graph, behavior_graph: nx.Graph):
        amount = self.deletion_amount
        if type(amount) is float:
            raise ValueError(
                "Local linear deletion only works with integer deletion amounts."
            )
        for node in behavior_graph.nodes():
            edges = (
                list(
                    filter(
                        lambda edge: not propagation_graph.has_edge(*edge),
                        list(behavior_graph.edges(node)),
                    )
                )
                if self.posterior
                else list(behavior_graph.edges(node))
            )
            edges = random.sample(edges, min(amount, len(edges)))
            for edge in edges:
                remove_edge_but_keep_if_would_disconnect_graph(
                    behavior_graph, edge)
        return [
            behavior_graph
            for _ in range(len(self.propagation.behavior_graph.snapshots))
        ]

    def delete_proportionate(
        self, propagation_graph: nx.Graph, behavior_graph: nx.Graph
    ):
        for node in behavior_graph.nodes():
            edges = (
                list(
                    filter(
                        lambda edge: not propagation_graph.has_edge(*edge),
                        list(behavior_graph.edges(node)),
                    )
                )
                if self.posterior
                else list(behavior_graph.edges(node))
            )
            amount = int(self.deletion_amount * len(edges))
            edges = random.sample(edges, amount)
            for edge in edges:
                remove_edge_but_keep_if_would_disconnect_graph(
                    behavior_graph, edge)
        return [
            behavior_graph
            for _ in range(len(self.propagation.behavior_graph.snapshots))
        ]

    def perturb(self):
        behavior_graph_copy = copy.deepcopy(
            self.propagation.behavior_graph.snapshots[0]
        )
        propagation_graph_copy = copy.deepcopy(self.propagation.snapshots[-1])
        if self.method == "global":
            return self.delete_global(propagation_graph_copy, behavior_graph_copy)
        elif self.method == "linear":
            return self.delete_linear(propagation_graph_copy, behavior_graph_copy)
        elif self.method == "proportionate":
            return self.delete_proportionate(
                propagation_graph_copy, behavior_graph_copy
            )

        raise ValueError(f"Unknown method {self.method}.")
