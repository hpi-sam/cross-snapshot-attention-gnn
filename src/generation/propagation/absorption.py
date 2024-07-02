from typing import Callable, Tuple, Union

from src.generation.behavior.random import BehaviorGraph
from src.utils.graph import sample_from_neighborhood


class AbsorptionConfig:
    def __init__(
        self,
        rate: Union[float, Callable[[int, int], float]] = 1.0,
        latency: Union[int, Callable[[int, int], int]] = 0,
    ) -> None:
        self.rate = (
            rate if callable(rate) else lambda node, t: rate
        )  # type: Callable[[int, int], float]
        self.latency = (
            latency if callable(latency) else lambda node, t: latency
        )  # type: Callable[[int, int], int]


class Absorption:
    """Helper to sample the absorbed edges.

    During the absorption process the edges are sampled that a node absorbs at time t.
    For a node, these edges represent the connections to other nodes that the node is susceptible to.
    """

    def __init__(self, behavior_graph: BehaviorGraph, config: AbsorptionConfig) -> None:
        self.config = config
        self.behavior_graph = behavior_graph
        self.absorptions: dict[int, dict[Tuple[int, int], int]] = {}

    def sample(self, propagations: dict[int, Tuple[int, Tuple[int, int]]], t: int):
        """Samples edges that are absorbed at time t.

        Takes all uncovered nodes and samples edges from their neighborhood in the behavior graph
        at time t. Each of these sampled edges is considered to be an absorption at time t, while the
        actual cover time is t + absorption_latency(t).

        :param propagations: dict of propagations that include nodes at are covered or will be covered in the future
        :param t: time to sample the absorptions for
        :return: dict of edges that are absorbed at time t and their actual cover time
        """
        behavior_graph = self.behavior_graph.snapshots[t]
        uncovered_nodes = [
            node for node in behavior_graph.nodes() if node not in propagations
        ]
        self.absorptions[t] = {}
        for node in uncovered_nodes:
            node_absorption_rate = self.config.rate(node, t)
            node_absorption_latency = self.config.latency(node, t)
            absorbed_edges = [
                # reverse edges as sampled for target node
                (edge[1], edge[0])
                for edge in sample_from_neighborhood(
                    behavior_graph, node, node_absorption_rate
                )
            ]
            actual_t = t + node_absorption_latency
            for edge in absorbed_edges:
                self.absorptions[t][edge] = actual_t

        return self.absorptions[t]
