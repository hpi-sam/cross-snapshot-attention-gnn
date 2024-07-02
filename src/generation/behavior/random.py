from enum import Enum
import networkx as nx
import numpy as np
import random
from typing import List
import copy
from typing import Union
import math

from src.generation.behavior.behavior import BehaviorGraph
from src.generation.behavior.expanders import (
    random_expander,
    angluin_expander,
    margulis_expander,
)


class StaticFixedBehaviorGraph(BehaviorGraph):
    def __init__(self, graph: nx.Graph, max_t: int) -> None:
        self.graph = graph
        super().__init__(max_t)

    def sample_static(self) -> nx.Graph:
        return self.graph


# fully connected graphs


class StaticFCBehaviorGraphConfig:
    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes


class StaticFCBehaviorGraph(BehaviorGraph):
    def __init__(self, config: StaticFCBehaviorGraphConfig, max_t: int) -> None:
        self.config = config
        super().__init__(max_t)

    def sample_static(self) -> nx.Graph:
        return nx.complete_graph(self.config.num_nodes)


# erdos renyi graphs


class StaticERBehaviorGraphConfig:
    def __init__(self, num_nodes: int, edge_probability: float) -> None:
        self.num_nodes = num_nodes
        self.edge_probability = edge_probability


class StaticERBehaviorGraph(BehaviorGraph):
    def __init__(self, config: StaticERBehaviorGraphConfig, max_t: int) -> None:
        self.config = config
        super().__init__(max_t)

    def sample_static(self) -> nx.Graph:
        return nx.erdos_renyi_graph(self.config.num_nodes, self.config.edge_probability)


# watts strogatz graphs


class StaticWSBehaviorGraphConfig:
    def __init__(
        self, num_nodes: int, join_amount: int, rewiring_probability: float
    ) -> None:
        self.num_nodes = num_nodes
        self.join_amount = join_amount
        self.rewiring_probability = rewiring_probability


class StaticWSBehaviorGraph(BehaviorGraph):
    def __init__(self, config: StaticWSBehaviorGraphConfig, max_t: int) -> None:
        self.config = config
        super().__init__(max_t)

    def sample_static(self) -> nx.Graph:
        return nx.watts_strogatz_graph(
            self.config.num_nodes,
            self.config.join_amount,
            self.config.rewiring_probability,
        )


# barabasi albert graphs


class StaticBABehaviorGraphConfig:
    def __init__(self, num_nodes: int, join_amount: int) -> None:
        self.num_nodes = num_nodes
        self.join_amount = join_amount


class StaticBABehaviorGraph(BehaviorGraph):
    def __init__(self, config: StaticBABehaviorGraphConfig, max_t: int) -> None:
        self.config = config
        super().__init__(max_t)

    def sample_static(self) -> nx.Graph:
        if self.config.join_amount >= self.config.num_nodes:
            raise RuntimeError("Can't join to more than n nodes")
        return nx.barabasi_albert_graph(
            self.config.num_nodes,
            self.config.join_amount,
        )


# expanders


class ExpanderMethod(Enum):
    RANDOM = 1
    ANGLUIN = 2
    MARGULIS = 3


class StaticEXPBehaviorGraphConfig:
    def __init__(
        self,
        num_nodes: int,
        join_amount: int = 3,
        method: ExpanderMethod = ExpanderMethod.RANDOM,
    ) -> None:
        self.num_nodes = num_nodes
        self.join_amount = join_amount
        self.method = method


class StaticEXPBehaviorGraph(BehaviorGraph):
    def __init__(self, config: StaticEXPBehaviorGraphConfig, max_t: int) -> None:
        self.config = config
        super().__init__(max_t)

    def gen_random(self):
        return random_expander.generate_expander(
            self.config.join_amount, self.config.num_nodes
        )

    def gen_angluin(self):
        n = int(math.sqrt(self.config.num_nodes / 2))
        indices_of_pairs = np.arange(n * n)
        indices = np.random.permutation(indices_of_pairs).reshape((n, n))
        return angluin_expander.generate_expander(n * n, indices, n)

    def gen_margulis(self):
        n = int(math.sqrt(self.config.num_nodes / 2))
        indices_of_pairs = np.arange(n * n)
        indices = np.random.permutation(indices_of_pairs).reshape((n, n))
        result = margulis_expander.generate_expander(n * n, indices, n)
        return result

    def get_gen_method(self):
        if self.config.method == ExpanderMethod.RANDOM:
            return self.gen_random
        elif self.config.method == ExpanderMethod.ANGLUIN:
            return self.gen_angluin
        elif self.config.method == ExpanderMethod.MARGULIS:
            return self.gen_margulis
        else:
            raise RuntimeError("Unknown expander method")

    def sample_static(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(self.config.num_nodes))
        adj_list = self.get_gen_method()()
        edges = [
            (i, x)
            for i, connected_to in enumerate(adj_list)
            for x in connected_to
            if x != -1
        ]
        g.add_edges_from(edges)
        return g
