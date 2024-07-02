import random
from typing import List, Union
import time

import numpy as np
from tqdm import tqdm

from src.datasets.dataset import Dataset, Label, Sample
from src.generation.behavior.behavior import BehaviorGraph
from src.generation.behavior.random import (
    StaticBABehaviorGraph,
    StaticBABehaviorGraphConfig,
    StaticERBehaviorGraph,
    StaticERBehaviorGraphConfig,
    StaticEXPBehaviorGraph,
    StaticEXPBehaviorGraphConfig,
    StaticFCBehaviorGraph,
    StaticFCBehaviorGraphConfig,
    StaticWSBehaviorGraph,
    StaticWSBehaviorGraphConfig,
)
from src.generation.propagation.propagation import Propagation
from src.utils.numbers import get_random_seed


class RandomDataset(Dataset):
    """Random dataset generator.

    A dataset that generates random behavior graphs from a given distribution.
    It accepts a list of possible labels that hold the propagation configuration for that specific label.
    For each label the given amount of propagation samples is generated, where each sample is produced with a randomly chosen behavior graph.
    Available behavior graphs are:
        - StaticERBehaviorGraph
        - StaticWSBehaviorGraph
        - StaticBABehaviorGraph
        - StaticEXPBehaviorGraph
        - StaticFCBehaviorGraph

    Args:
        max_t (Union[int, List[int]]): Maximum time step of the generated behavior graphs. If a list is given, a random value is sampled from the range (Default: 10).
        num_nodes (Union[int, List[int]]): Number of nodes of the generated behavior graphs. If a list is given, a random value is sampled from the range (Default: 10).
        num_samples_per_label (int): Number of samples per label (Default: 100).
        train_behavior_distribution (dict): Distribution of the behavior graphs in the training set. The keys are the behavior graph types and the values are the probabilities (Default: {"er": 0.25, "ws": 0.25, "ba": 0.25, "exp": 0.25}).
        test_behavior_distribution (dict): Distribution of the behavior graphs in the test set. If none given, uses same as train_behavior_distribution.
        train_labels (List[Label]): List of labels that should be generated.
        test_labels (List[Label]): List of labels that should be generated.
        train_distinct_behavior_graphs (int): Number of distinct behavior graphs in the training set (Default: None).
        test_distinct_behavior_graphs (int): Number of distinct behavior graphs in the test set (Default: None).
        log_progress (bool): Whether to log the progress of the generation (Default: False).
        collision_threshold (float): Threshold for the collision detection. If the similarity between two propagations is greater than this threshold, they are considered to be colliding and excluded from the dataset sampling (Default: 1.0).
    """

    def __init__(
        self,
        max_t: Union[int, List[int]] = 10,
        num_nodes: Union[int, List[int]] = 10,
        num_samples_per_label: int = 100,
        behavior_edge_probability=None,
        train_behavior_distribution: dict = None,
        test_behavior_distribution: dict = None,
        train_labels=None,
        test_labels=None,
        train_distinct_behavior_graphs=None,
        test_distinct_behavior_graphs=None,
        train_collision_threshold=None,
        test_collision_threshold=1.0,
        log_progress=False,
        name_suffix=None,
        **kwargs,
    ) -> None:
        self.max_t = max_t
        self.num_nodes = num_nodes
        self.behavior_edge_probability = behavior_edge_probability
        self.log_progress = log_progress
        self.num_train_samples_per_label = round(num_samples_per_label * 0.8)
        self.num_test_samples_per_label = round(num_samples_per_label * 0.2)
        self.train_behavior_distribution = (
            train_behavior_distribution
            if train_behavior_distribution is not None
            else {"er": 0.25, "ws": 0.25, "ba": 0.25, "exp": 0.25}
        )

        self.test_behavior_distribution = (
            test_behavior_distribution
            if test_behavior_distribution is not None
            else self.train_behavior_distribution
        )
        self.train_labels = (
            train_labels if train_labels is not None else kwargs.get("labels")
        )
        self.test_labels = (
            test_labels if test_labels is not None else kwargs.get("labels")
        )
        self.train_distinct_behavior_graphs = train_distinct_behavior_graphs
        self.test_distinct_behavior_graphs = test_distinct_behavior_graphs
        self.train_collision_threshold = train_collision_threshold
        self.test_collision_threshold = test_collision_threshold

        self.train_behavior_seeds = [get_random_seed() for _ in range(
            self.train_distinct_behavior_graphs)] if self.train_distinct_behavior_graphs is not None else None
        self.test_behavior_seeds = [get_random_seed() for _ in range(
            self.test_distinct_behavior_graphs)] if self.test_distinct_behavior_graphs is not None else None

        # Update dataset name given the parameters
        name = kwargs.pop("name", "RandomDataset")
        abbreviation = kwargs.pop("abbreviation", "RD")
        if isinstance(num_nodes, int):
            name = f"{name}&num_nodes_{num_nodes}"
            abbreviation = f"{abbreviation}&num_nodes_{num_nodes}"
        if (
            train_behavior_distribution is not None
            or test_behavior_distribution is not None
        ):
            name = f"{name}&dist_{train_behavior_distribution}_{test_behavior_distribution}"
            abbreviation = f"{abbreviation}&dist_{train_behavior_distribution}_{test_behavior_distribution}"
        if (
            train_distinct_behavior_graphs is not None
            or test_distinct_behavior_graphs is not None
        ):
            name = f"{name}&num_graphs_{train_distinct_behavior_graphs}_{test_distinct_behavior_graphs}"
            abbreviation = f"{abbreviation}&num_graphs_{train_distinct_behavior_graphs}_{test_distinct_behavior_graphs}"
        if behavior_edge_probability is not None:
            name = f"{name}&edge_prob_{behavior_edge_probability}"
            abbreviation = f"{abbreviation}&edge_prob_{behavior_edge_probability}"
        if train_collision_threshold is not None or test_collision_threshold != 1.0:
            name = f"{name}&collision{train_collision_threshold}_{test_collision_threshold}"
            abbreviation = f"{abbreviation}&collision{train_collision_threshold}_{test_collision_threshold}"
        if name_suffix is not None:
            name = f"{name}&{name_suffix}"
            abbreviation = f"{abbreviation}&{name_suffix}"
        super().__init__(name=name, abbreviation=abbreviation, **kwargs)

    def sample_node_count(self) -> int:
        if isinstance(self.num_nodes, list):
            return random.choice(range(self.num_nodes[0], self.num_nodes[1]))
        return self.num_nodes

    def sample_max_t(self) -> int:
        if isinstance(self.max_t, list):
            return random.choice(range(self.max_t[0], self.max_t[1]))
        return self.max_t

    def sample_edge_probability(self, min_p=0.1, max_p=1.0) -> Union[int, float]:
        if self.behavior_edge_probability is not None:
            return self.behavior_edge_probability
        return min(random.random() + min_p, max_p)

    def sample_fc_graph(self) -> BehaviorGraph:
        node_count = self.sample_node_count()
        max_t = self.sample_max_t()
        return StaticFCBehaviorGraph(
            StaticFCBehaviorGraphConfig(node_count),
            max_t,
        )

    def sample_er_graph(self, min_edge_prob=0.1) -> BehaviorGraph:
        node_count = self.sample_node_count()
        max_t = self.sample_max_t()
        edge_probability = self.sample_edge_probability(
            min_p=min_edge_prob)
        return StaticERBehaviorGraph(
            StaticERBehaviorGraphConfig(node_count, edge_probability),
            max_t,
        )

    def sample_ws_graph(self, min_join_amount=2, min_edge_prob=0.1, min_rewiring_prob=0.1) -> BehaviorGraph:
        node_count = self.sample_node_count()
        max_t = self.sample_max_t()
        edge_probability = self.sample_edge_probability(
            min_p=min_edge_prob)
        join_amount = max(
            int(edge_probability * (node_count - 1)), min_join_amount)
        rewiring_probability = min(random.random() + min_rewiring_prob, 0.9)
        return StaticWSBehaviorGraph(
            StaticWSBehaviorGraphConfig(
                node_count, join_amount, rewiring_probability),
            max_t,
        )

    def sample_ba_graph(self, min_join_amount=2, min_edge_prob=0.1) -> BehaviorGraph:
        node_count = self.sample_node_count()
        max_t = self.sample_max_t()
        edge_probability = self.sample_edge_probability(
            min_p=min_edge_prob)
        join_amount = max(
            int(edge_probability * (node_count - 1)), min_join_amount)
        return StaticBABehaviorGraph(
            StaticBABehaviorGraphConfig(node_count, join_amount),
            max_t,
        )

    def sample_exp_graph(self, min_join_amount=3, min_edge_prob=0.1) -> BehaviorGraph:
        node_count = self.sample_node_count()
        max_t = self.sample_max_t()
        edge_probability = self.sample_edge_probability(
            min_p=min_edge_prob)
        join_amount = max(
            int(edge_probability * (node_count - 1)), min_join_amount)
        return StaticEXPBehaviorGraph(
            StaticEXPBehaviorGraphConfig(node_count, join_amount),
            max_t,
        )

    def transform_behavior_distribution(self, num_samples_per_label: int, train: bool):
        result = {}
        prev_sum = 0
        behavior_distribution = (
            self.train_behavior_distribution
            if train
            else self.test_behavior_distribution
        )
        for key, num in behavior_distribution.items():
            prev_sum = prev_sum + num * num_samples_per_label
            result[key] = int(prev_sum)
        return result

    def get_sample_fn_dict(self):
        return {
            "fc": self.sample_fc_graph,
            "er": self.sample_er_graph,
            "ws": self.sample_ws_graph,
            "ba": self.sample_ba_graph,
            "exp": self.sample_exp_graph,
        }

    def sample_behavior_graph(self, j: int, transformed_dist: dict, train=True) -> BehaviorGraph:
        seeds = self.train_behavior_seeds if train else self.test_behavior_seeds
        # limit the sampling to a given number of distinct graphs using seeds
        if seeds is not None:
            random.seed(random.choice(seeds))
        behavior_graph = None
        for key, value in transformed_dist.items():
            if j < value:
                behavior_graph = self.get_sample_fn_dict()[key]()
                break
        behavior_graph = behavior_graph if behavior_graph is not None else self.get_sample_fn_dict()[
            list(transformed_dist.keys())[-1]]()
        if seeds is not None:
            random.seed(None)
        return behavior_graph

    def sample_propagation(
        self, behavior_graph: BehaviorGraph, label: Label
    ) -> Propagation:
        return Propagation(behavior_graph, label.instantiate(behavior_graph))

    def is_collision_free(
        self, propagation: Propagation, samples: List[Sample], threshold: Union[float, None]
    ) -> bool:
        if threshold is None:
            return True
        for sample in samples:
            if propagation.equals(sample.propagation, threshold):
                return False
        return True

    def sample(self, train=True):
        labels = self.train_labels if train else self.test_labels
        if (len(labels) == 0):
            raise RuntimeError(
                "Cannot sample without labels")

        if not train and len(self.train) == 0:
            raise RuntimeError(
                "Cannot sample test set without train set")

        samples: List[Sample] = []

        for i, label in enumerate(labels):
            start_time = time.time()
            print_skip_collision_message = False
            max_time = 30

            num_samples_per_label = (
                self.num_train_samples_per_label
                if train
                else self.num_test_samples_per_label
            )
            transformed_dist = self.transform_behavior_distribution(
                num_samples_per_label=num_samples_per_label, train=train
            )
            for j in (
                tqdm(
                    range(num_samples_per_label),
                    f"Sample {label.name} ({'train' if train else 'test'}) for {self.name}",
                )
                if self.log_progress
                else range(num_samples_per_label)
            ):
                while True:
                    behavior_graph = self.sample_behavior_graph(
                        j=j, transformed_dist=transformed_dist, train=train)
                    propagation = self.sample_propagation(
                        behavior_graph=behavior_graph, label=label)
                    if self.is_collision_free(
                        propagation=propagation,
                        # for the test set, we also check for collisions with the train set
                        samples=samples if train else [*samples, *self.train],
                        threshold=self.train_collision_threshold if train else self.test_collision_threshold,
                    ):
                        break
                    elapsed_time = time.time() - start_time
                    if elapsed_time > max_time:
                        if not print_skip_collision_message:
                            print(
                                f"SAMPLING TIMEOUT: Skipping collision check for {label.name} ({'train' if train else 'test'})")
                        print_skip_collision_message = True
                        break
                sample = Sample(propagation=propagation, label=i)
                samples.append(sample)

        np.random.shuffle(samples)
        return samples
