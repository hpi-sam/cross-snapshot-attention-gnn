import networkx as nx
from typing import List, Union, Tuple
import copy

from src.generation.propagation.propagation import Propagation


class Perturbation:
    def __init__(self, propagation: Propagation, name: str):
        self.propagation = copy.deepcopy(propagation)
        self.name = name

    def perturb(self) -> Union[List[nx.Graph], Tuple[List[nx.Graph], List[nx.Graph]]]:
        raise NotImplementedError("Perturbation.perturb() is not implemented.")

    def is_valid_perturbation(self, graph: nx.Graph):
        return nx.is_connected(graph)


class BehaviorPerturbation(Perturbation):
    def __init__(self, propagation: Propagation, posterior=False, **kwargs):
        super().__init__(propagation, **kwargs)
        self.posterior = posterior
        perturbed_behavior_graph_snapshots = None
        perturbed_propagation_graph_snapshots = None
        perturbed_anchor = None
        while (
            perturbed_behavior_graph_snapshots is None
            # pylint: disable=unsubscriptable-object
            or not self.is_valid_perturbation(perturbed_behavior_graph_snapshots[0])
        ):
            result = self.perturb()
            # if tuple returned, first is behavior snapshots, second is propagation snapshots
            if isinstance(result, tuple):
                (
                    perturbed_behavior_graph_snapshots,
                    perturbed_propagation_graph_snapshots,
                    perturbed_anchor,
                ) = result
            else:
                perturbed_behavior_graph_snapshots = result

        self.propagation.behavior_graph.snapshots = perturbed_behavior_graph_snapshots
        if perturbed_propagation_graph_snapshots is not None:
            self.propagation.snapshots = perturbed_propagation_graph_snapshots
        if perturbed_anchor is not None:
            self.propagation.anchor = perturbed_anchor
        if not posterior:
            self.propagation.sample(keep_anchor=True)
