import copy
import random
from typing import Union, List

import networkx as nx

from src.perturbations.perturbation import BehaviorPerturbation
from src.utils.graph import (
    remove_node_but_keep_if_would_disconnect_graph,
    relabel_nodes_consecutive,
)


class RandomNodeDeletion(BehaviorPerturbation):
    def __init__(
        self,
        deletion_amount: Union[int, float] = 0.1,
        method="proportionate",
        **kwargs,
    ):
        self.deletion_amount = deletion_amount
        self.method = method
        super().__init__(name=f"node_deletion_{method}", **kwargs)

    def delete_by_amount(
        self,
        propagation_snapshots: List[nx.Graph],
        behavior_graph: nx.Graph,
        amount: int,
    ):
        actual_deletions = 0
        nodes_attempted = set()
        while actual_deletions < amount:
            nodes = (
                list(
                    filter(
                        lambda node: not propagation_snapshots[-1].has_node(node),
                        list(behavior_graph.nodes()),
                    )
                )
                if self.posterior
                else list(
                    filter(
                        lambda node: node != self.propagation.anchor,
                        list(behavior_graph.nodes()),
                    )
                )
            )
            nodes_remaining = set(nodes) - nodes_attempted
            if not nodes_remaining:
                break  # All nodes have been attempted for deletion
            node = random.choice(list(nodes_remaining))
            nodes_attempted.add(node)
            if remove_node_but_keep_if_would_disconnect_graph(behavior_graph, node):
                actual_deletions += 1
                nodes_attempted.clear()  # Reset the nodes_attempted set

        behavior_graph, mapping = relabel_nodes_consecutive(behavior_graph)
        propagation_snapshots = [
            nx.relabel_nodes(g, mapping) for g in propagation_snapshots
        ]
        anchor = mapping[self.propagation.anchor]

        return (
            [
                behavior_graph
                for _ in range(len(self.propagation.behavior_graph.snapshots))
            ],
            propagation_snapshots,
            anchor,
        )

    def delete_linear(
        self, propagation_graph: List[nx.Graph], behavior_graph: nx.Graph
    ):
        amount = self.deletion_amount
        if type(amount) is float:
            raise ValueError(
                "Local linear deletion only works with integer deletion amounts."
            )
        return self.delete_by_amount(propagation_graph, behavior_graph, amount)

    def delete_proportionate(
        self, propagation_graph: List[nx.Graph], behavior_graph: nx.Graph
    ):
        amount = int(self.deletion_amount * len(behavior_graph.nodes()))
        return self.delete_by_amount(propagation_graph, behavior_graph, amount)

    def perturb(self):
        behavior_graph_copy = copy.deepcopy(
            self.propagation.behavior_graph.snapshots[0]
        )
        propagation_snapshots = copy.deepcopy(self.propagation.snapshots)
        if self.method == "linear":
            return self.delete_linear(propagation_snapshots, behavior_graph_copy)
        elif self.method == "proportionate":
            return self.delete_proportionate(propagation_snapshots, behavior_graph_copy)

        raise ValueError(f"Unknown method {self.method}.")
