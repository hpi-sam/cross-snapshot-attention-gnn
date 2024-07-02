import copy
import random
from typing import Union

import networkx as nx

from src.perturbations.perturbation import BehaviorPerturbation


class RandomEdgeAddition(BehaviorPerturbation):
    def __init__(
        self,
        addition_amount: Union[int, float] = 0.1,
        method="proportionate",
        **kwargs,
    ):
        self.addition_amount = addition_amount
        self.method = method
        super().__init__(name=f"edge_addition_{method}", **kwargs)

    def add_linear(self, behavior_graph: nx.Graph):
        amount = self.addition_amount
        if type(amount) is float:
            raise ValueError(
                "Local linear addition only works with integer addition amounts."
            )
        for node in behavior_graph.nodes():
            # Find the nodes that are not connected to the current node
            non_neighbors = list(
                set(behavior_graph.nodes())
                - set(behavior_graph.neighbors(node))
                - {node}
            )

            # Randomly select nodes to connect to, without replacement
            new_neighbors = random.sample(
                non_neighbors, min(amount, len(non_neighbors))
            )

            # Add edges to the behavior_graph
            for new_node in new_neighbors:
                behavior_graph.add_edge(node, new_node)

        return [
            behavior_graph
            for _ in range(len(self.propagation.behavior_graph.snapshots))
        ]

    def add_proportionate(self, behavior_graph: nx.Graph):
        for node in behavior_graph.nodes():
            # Find the nodes that are not connected to the current node
            non_neighbors = list(
                set(behavior_graph.nodes())
                - set(behavior_graph.neighbors(node))
                - {node}
            )

            # Calculate the number of edges to add
            amount = int(self.addition_amount * len(non_neighbors))

            # Randomly select nodes to connect to, without replacement
            new_neighbors = random.sample(non_neighbors, amount)

            # Add edges to the behavior_graph
            for new_node in new_neighbors:
                behavior_graph.add_edge(node, new_node)

        return [
            behavior_graph
            for _ in range(len(self.propagation.behavior_graph.snapshots))
        ]

    def perturb(self):
        behavior_graph_copy = copy.deepcopy(
            self.propagation.behavior_graph.snapshots[0]
        )
        if self.method == "linear":
            return self.add_linear(behavior_graph_copy)
        elif self.method == "proportionate":
            return self.add_proportionate(behavior_graph_copy)

        raise ValueError(f"Unknown method {self.method}.")
