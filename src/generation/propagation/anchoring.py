from typing import Union, Callable
import random

import networkx as nx

from src.utils.metrics import Metrics
from src.utils.graph import sample_node_based_on_centrality_quantile


class Anchoring:
    """Helper to sample the anchor (=starting node) of a propagation."""

    def __init__(self) -> None:
        pass

    def sample(self, graph: nx.Graph) -> int:
        """Samples a random anchor node from the graph."""
        return random.choice(list(graph.nodes()))


class FixedAnchoring(Anchoring):
    def __init__(self, node_id: int = 0) -> None:
        super().__init__()
        self.anchor = node_id

    def sample(self, graph: nx.Graph) -> int:
        return self.anchor


class CentralityAnchoring(Anchoring):
    def __init__(
        self, quantile: Union[float, Callable, None] = None, exact: Union[str, None] = None
    ) -> None:
        super().__init__()
        self.exact = exact
        self.quantile = quantile

    def sample(self, graph: nx.Graph) -> int:
        if self.exact is not None:
            return self.sample_exact(graph)
        if self.quantile is not None:
            return self.sample_quantile(graph)
        raise RuntimeError("No sampling strategy selected.")

    def get_centralities(self, graph: nx.Graph) -> dict:
        raise RuntimeError("Function not implemented.")

    def sample_quantile(self, graph: nx.Graph) -> int:
        quantile = self.quantile() if callable(self.quantile) else self.quantile
        return sample_node_based_on_centrality_quantile(
            graph, quantile, self.get_centralities
        )

    def sample_exact(self, graph: nx.Graph) -> int:
        centralities = self.get_centralities(graph)
        anchor = None
        anchor_value = None
        if self.exact == "lowest":
            for n, v in centralities.items():
                if anchor is None or v < anchor_value:
                    anchor = n
                    anchor_value = v
        else:
            for n, v in centralities.items():
                if anchor is None or v > anchor_value:
                    anchor = n
                    anchor_value = v
        return anchor


class DegreeCentralityAnchoring(CentralityAnchoring):
    def get_centralities(self, graph: nx.Graph) -> dict:
        return nx.degree_centrality(graph)


class ClosenessCentralityAnchoring(CentralityAnchoring):
    def get_centralities(self, graph: nx.Graph) -> dict:
        return nx.closeness_centrality(graph)


class BetweennessCentralityAnchoring(CentralityAnchoring):
    def get_centralities(self, graph: nx.Graph) -> dict:
        return nx.betweenness_centrality(graph)


class CentralityAnchoringMetrics(Metrics):
    def __init__(
        self,
        prefix: str,
        graph: Union[nx.Graph, None] = None,
        anchor: Union[int, None] = None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.compute(graph, anchor)

    def get_centralities(self, graph: nx.Graph) -> dict:
        raise RuntimeError("Function not implemented.")

    def compute(
        self, graph: Union[nx.Graph, None] = None, anchor: Union[int, None] = None
    ) -> dict:
        if graph is not None:
            centralities = self.get_centralities(graph)
            self._[f"{self.prefix}_centrality_anchoring"] = centralities[anchor]
        else:
            self._[f"{self.prefix}_centrality_anchoring"] = 0

        return self._


class DegreeCentralityAnchoringMetrics(CentralityAnchoringMetrics):
    def __init__(
        self, graph: Union[nx.Graph, None] = None, anchor: Union[int, None] = None
    ) -> None:
        super().__init__("degree", graph, anchor)

    def get_centralities(self, graph: nx.Graph) -> dict:
        return nx.degree_centrality(graph)


class ClosenessCentralityAnchoringMetrics(CentralityAnchoringMetrics):
    def __init__(
        self, graph: Union[nx.Graph, None] = None, anchor: Union[int, None] = None
    ) -> None:
        super().__init__("closeness", graph, anchor)

    def get_centralities(self, graph: nx.Graph) -> dict:
        return nx.closeness_centrality(graph)


class BetweennessCentralityAnchoringMetrics(CentralityAnchoringMetrics):
    def __init__(
        self, graph: Union[nx.Graph, None] = None, anchor: Union[int, None] = None
    ) -> None:
        super().__init__("betweenness", graph, anchor)

    def get_centralities(self, graph: nx.Graph) -> dict:
        return nx.betweenness_centrality(graph)
