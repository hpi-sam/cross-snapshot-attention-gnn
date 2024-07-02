import copy
import random
from typing import Union

import networkx as nx

from src.perturbations.perturbation import BehaviorPerturbation


class RandomNodeAddition(BehaviorPerturbation):
    def __init__(
        self,
        addition_amount: Union[int, float] = 0.1,
        method="proportionate",
        **kwargs,
    ):
        self.addition_amount = addition_amount
        self.method = method
        super().__init__(name=f"node_addition_{method}", **kwargs)

    def add_by_amount(self, behavior_graph: nx.Graph, amount: int):
        # Find the highest node ID in the current graph
        highest_id = max(behavior_graph.nodes())

        for _ in range(amount):
            # Add a new node with the next highest ID
            new_node_id = highest_id + 1
            behavior_graph.add_node(new_node_id)

            # Select a random number of nodes to connect the new node to
            num_connections = random.randint(1, len(behavior_graph.nodes()) - 1)

            # Randomly select nodes to connect to, without replacement
            nodes_to_connect = random.sample(
                set(behavior_graph.nodes()) - {new_node_id}, num_connections
            )

            # Add edges to the behavior_graph
            for node in nodes_to_connect:
                behavior_graph.add_edge(new_node_id, node)

            highest_id = new_node_id

        return [
            behavior_graph
            for _ in range(len(self.propagation.behavior_graph.snapshots))
        ]

    def add_linear(self, behavior_graph: nx.Graph):
        amount = self.addition_amount
        if type(amount) is float:
            raise ValueError(
                "Local linear addition only works with integer addition amounts."
            )
        return self.add_by_amount(behavior_graph, amount)

    def add_proportionate(self, behavior_graph: nx.Graph):
        amount = int(self.addition_amount * len(behavior_graph.nodes()))
        return self.add_by_amount(behavior_graph, amount)

    def perturb(self):
        behavior_graph_copy = copy.deepcopy(
            self.propagation.behavior_graph.snapshots[0]
        )
        if self.method == "linear":
            return self.add_linear(behavior_graph_copy)
        elif self.method == "proportionate":
            return self.add_proportionate(behavior_graph_copy)

        raise ValueError(f"Unknown method {self.method}.")
