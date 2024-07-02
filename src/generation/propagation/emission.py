import math
from typing import Callable, List, Tuple, Union, Any

from src.generation.behavior.random import BehaviorGraph
from src.utils.graph import sample_from_neighborhood


class EmissionConfig:
    def __init__(
        self,
        rate: Union[float, Callable[[int, int, int, Any], float]] = 1.0,
        latency: Union[int, Callable[[int, int], int]] = 0,
        duration: Union[int, Callable[[int, int], int]] = math.inf,
    ) -> None:
        self.rate = (
            rate if callable(
                rate) else lambda node, t, node_emission_start, *args: rate
        )  # type: Callable[[int, int, int, Any], float]
        self.latency = (
            latency if callable(latency) else lambda node, t: latency
        )  # type: Callable[[int, int], int]
        self.duration = (
            duration if callable(duration) else lambda node, t: duration
        )  # type: Callable[[int, int], int]


class Emission:
    """Helper to sample the emitted edges.

    During the emission process the edges are sampled that a covered node emits to at time t.
    For a node, these edges represent the connections to other nodes that the node spreads the event to.
    """

    def __init__(
        self, behavior_graph: BehaviorGraph, config: EmissionConfig, anchor: int
    ) -> None:
        self.config = config
        self.behavior_graph = behavior_graph
        # holds the emission start time for each node
        self.emitters = {anchor: self.config.latency(anchor, 0)}
        self.emissions: dict[int, List[Tuple[int, int]]] = {}

    def can_emit(self, node: int, t: int):
        """Returns whether the node can emit at time t. A node can emit if the emission start reached and the emission duration is not yet reached."""
        if node not in self.emitters.keys():
            return False
        emission_start = self.emitters[node]
        emission_duration = self.config.duration(node, emission_start)
        return t >= emission_start and t <= (emission_start + emission_duration)

    def get_emitters(self, t: int):
        """Returns all nodes that can emit at time t."""
        return list(
            filter(lambda node: self.can_emit(
                node, t), list(self.emitters.keys()))
        )

    def set_emitters(self, propagations: dict[int, Tuple[int, Tuple[int, int]]]):
        """Sets the emitters for the given nodes using the cover time and the emission latency."""
        for node, (cover_time, _) in propagations.items():
            if node in self.emitters.keys():
                continue
            time_until_emission_start_after_cover = self.config.latency(
                node, cover_time
            )
            self.emitters[node] = (
                time_until_emission_start_after_cover + cover_time
            )

    def sample(
        self,
        propagations: dict[int, Tuple[int, Tuple[int, int]]],
        t: int,
        *args,
    ):
        """Samples edges that are emitted to at time t.

        Computes all nodes that can emit the event from the given propagations. Afterward, for each of these emitters,
        edges from their neighborhood in the behavior graph at time t are sampled. Each of these sampled edges is considered to
        be an emission at time t.

        :param propagations: dict of propagations that include nodes at are covered or will be covered in the future
        :param t: time to sample the emission for
        :return: list of edges that are emitted at time t
        """
        self.emissions[t] = []
        self.set_emitters(propagations)
        behavior_graph = self.behavior_graph.snapshots[t]
        emitters = self.get_emitters(t)
        for emitter in emitters:
            node_emission_start = self.emitters[emitter]
            node_emission_rate = self.config.rate(
                emitter, t, node_emission_start, *args
            )
            emitted_edges = sample_from_neighborhood(
                behavior_graph, emitter, node_emission_rate
            )
            for edge in emitted_edges:
                self.emissions[t].append(edge)
        return self.emissions[t]
