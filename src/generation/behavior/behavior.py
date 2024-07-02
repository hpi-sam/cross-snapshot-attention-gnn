import networkx as nx
from typing import List, Union
import math

from src.utils.metrics import Metrics


class BehaviorGraph:
    def __init__(self, max_t: int, sample=True) -> None:
        self.max_t = max_t
        self.snapshots: List[nx.Graph] = []
        if sample:
            self.sample()

    def sample_static(self) -> nx.Graph:
        raise NotImplementedError("Static sample function not implemented.")

    def sample_dynamic(self, static_graph: nx.Graph) -> None:
        return [static_graph for _ in range(self.max_t)]

    def diff(self, t1: int, t2: int):
        """Returns the difference between two behavior graph snapshots."""
        if t1 < 0:
            return list(self.snapshots[0].nodes()), list(self.snapshots[0].edges())
        if t2 <= t1:
            raise RuntimeError("t2 has to be smaller than t1")
        if len(self.snapshots) <= t2:
            raise RuntimeError("no snapshot for t2 found")
        snapshot_t2 = self.snapshots[t2]
        snapshot_t1 = self.snapshots[t1]
        nodes_diff = list(set(snapshot_t2.nodes()) - snapshot_t1.nodes())
        edges_diff = list(set(snapshot_t2.edges()) - snapshot_t1.edges())
        return nodes_diff, edges_diff

    def sample(self) -> List[nx.Graph]:
        static_graph = None
        while static_graph is None or not nx.is_connected(static_graph):
            # pylint: disable=assignment-from-no-return
            static_graph = self.sample_static()
        self.snapshots = self.sample_dynamic(static_graph)
        return self.snapshots


class BehaviorGraphMetrics(Metrics):
    def __init__(self, behavior_graph: Union[BehaviorGraph, None] = None) -> None:
        super().__init__()
        self.compute(behavior_graph)

    def compute(self, behavior_graph: Union[BehaviorGraph, None] = None) -> dict:
        graph = behavior_graph.snapshots[-1] if behavior_graph is not None else None
        self._["node_count"] = len(graph.nodes) if graph is not None else 0
        self._["edge_count"] = len(graph.edges) if graph is not None else 0
        self._["density"] = nx.density(graph) if graph is not None else 0

        self._["average_clustering_coefficient"] = (
            nx.average_clustering(graph) if graph is not None else 0
        )
        assortativity = (
            nx.degree_assortativity_coefficient(graph) if graph is not None else 0
        )
        self._["degree_assortativity_coefficient"] = (
            assortativity if not math.isnan(assortativity) else 0
        )

        self._["radius"] = nx.radius(graph) if graph is not None else 0
        self._["diameter"] = nx.diameter(graph) if graph is not None else 0

        self._["degree_centrality"] = self.compute_from_list(
            nx.degree_centrality(graph) if graph is not None else {}
        )
        self._["closeness_centrality"] = self.compute_from_list(
            nx.closeness_centrality(graph) if graph is not None else {}
        )
        self._["betweenness_centrality"] = self.compute_from_list(
            nx.betweenness_centrality(graph) if graph is not None else {}
        )
        return self._
