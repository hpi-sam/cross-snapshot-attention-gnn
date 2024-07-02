from __future__ import annotations

import copy
import math
import uuid
from typing import List, Tuple, Union

import networkx as nx
import numpy as np

from src.generation.behavior.random import BehaviorGraph
from src.generation.propagation.absorption import Absorption, AbsorptionConfig
from src.generation.propagation.anchoring import Anchoring
from src.generation.propagation.emission import Emission, EmissionConfig
from src.generation.propagation.transmission import Transmission, TransmissionConfig
from src.utils.graph import (
    get_neighborhood,
    num_nodes_reached_in_k_hops,
    count_shortest_paths,
)
from src.utils.metrics import Metrics


class PropagationConfig:
    def __init__(
        self,
        anchoring: Anchoring = Anchoring(),
        emission: EmissionConfig = EmissionConfig(),
        absorption: AbsorptionConfig = AbsorptionConfig(),
        transmission: TransmissionConfig = TransmissionConfig(),
    ) -> None:
        self.anchoring = anchoring
        self.emission = emission
        self.absorption = absorption
        self.transmission = transmission


class Propagation:
    """Representation of the propagation in a behavior graph.

    Accepts a behavior graph and a propagation config and samples a propagation from the behavior graph using the given config.
    """

    def __init__(
        self,
        behavior_graph: BehaviorGraph,
        config: Union[PropagationConfig, None] = None,
    ) -> None:
        self.id = uuid.uuid1()
        self.anchor = None
        self.config = config
        self.behavior_graph = behavior_graph
        self.snapshots: List[nx.Graph] = []
        # store node id, time of cover, and edge that caused propagation to this node
        self.propagations: dict[int, Tuple[int, Tuple[int, int]]]
        if config is not None:
            self.sample()

    def diff(self, t1: int, t2: int):
        """Returns the difference between two propagation snapshots.

        NOTE: t1 has to be smaller than t2"""
        if t1 < 0:
            return list(self.snapshots[0].nodes()), list(self.snapshots[0].edges())
        if t2 <= t1:
            raise RuntimeError("t1 has to be smaller than t2")
        if len(self.snapshots) <= t2:
            raise RuntimeError("no snapshot for t2 found")
        snapshot_t2 = self.snapshots[t2]
        snapshot_t1 = self.snapshots[t1]
        nodes_diff = list(set(snapshot_t2.nodes()) - snapshot_t1.nodes())
        edges_diff = list(set(snapshot_t2.edges()) - snapshot_t1.edges())
        return nodes_diff, edges_diff

    def equals(self, other: Propagation, threshold=None) -> Union[bool, float]:
        """Compute similarity between two propagations by counting the number of isomorphic snapshots.

        If threshold is given, returns True if the similarity is greater than the threshold.
        """
        if len(self.snapshots) != len(other.snapshots):
            return False
        snapshot_matches = 0
        for i, snapshot in enumerate(self.snapshots):
            other_snapshot = other.snapshots[i]
            if len(snapshot.nodes()) != len(other_snapshot.nodes()):
                continue
            if len(snapshot.edges()) != len(other_snapshot.edges()):
                continue
            if nx.faster_could_be_isomorphic(snapshot, other_snapshot):
                snapshot_matches += 1
        if threshold is None:
            return snapshot_matches / len(self.snapshots)
        return snapshot_matches / len(self.snapshots) >= threshold

    def matches_most(self, others: List[Propagation]) -> Tuple[Propagation, float]:
        """Returns the propagation that is most similar from the given list of propagations."""
        max_threshold = 0
        max_equal_other = None
        for other in others:
            if self.id == other.id:
                continue
            similarity = self.equals(other=other)
            if similarity > max_threshold:
                max_threshold = similarity
                max_equal_other = other
        return max_equal_other, max_threshold

    def mask_snapshot(self, t: int, remove=False):
        """Masks the propagation snapshot at time t by using the previous snapshot as a mask.

        If remove given, completely removes the snapshot at time t.
        """
        if (
            t <= 0
            or t >= len(self.snapshots)
            or t >= len(self.behavior_graph.snapshots)
        ):
            raise RuntimeError("No snapshot for t in propagation found")
        if remove:
            self.snapshots.pop(t)
            self.behavior_graph.snapshots.pop(t)
        else:
            self.snapshots[t] = copy.deepcopy(self.snapshots[t - 1])
            self.behavior_graph.snapshots[t] = copy.deepcopy(
                self.behavior_graph.snapshots[t - 1]
            )

    def mask_node(self, node: int, t: int):
        """Masks a given node for a snapshot at time t by using the node state of the previous snapshot as a mask (historical imputation).

        If node state stayed the same, the masking does not lead to any information loss.
        """
        if (
            t <= 0
            or t >= len(self.snapshots)
            or t >= len(self.behavior_graph.snapshots)
        ):
            raise RuntimeError("No snapshot for t in propagation found")

        # remove node if it just got covered
        if node in self.snapshots[t].nodes() and node not in self.snapshots[t-1].nodes():
            self.snapshots[t].remove_node(node)
        # add node if it just got uncovered
        elif node not in self.snapshots[t].nodes() and node in self.snapshots[t-1].nodes():
            self.snapshots[t].add_node(node)

    def node_is_covered(self, node: int, t: int = -1) -> bool:
        return self.snapshots[t].has_node(node)

    def node_is_uncovered_but_exposed(self, node: int, t: int) -> bool:
        if self.node_is_covered(node, t):
            return False
        neighborhood = get_neighborhood(self.behavior_graph.snapshots[t], node)
        for neighbor in neighborhood:
            if self.node_is_covered(neighbor, t):
                return True
        return False

    def sample(self, keep_anchor=False):
        """Samples a propagation from the behavior graph.

        During sampling, for each time step t, the following steps are performed:
        1. Sample emissions from the covered and latent nodes in the behavior graph
        2. Sample absorptions from the non-covered and non-latent nodes in the behavior graph
        3. Sample transmission state for edges that are included in emission and absorption
        4. For each edge that is transmitted. add it to the propagation graph

        :param keep_anchor: if True, the anchor of the propagation is not sampled again
        :return: list of propagation snapshots
        """
        if (
            self.anchor is None
            or not keep_anchor
            or self.anchor not in list(self.behavior_graph.snapshots[0].nodes())
        ):
            self.anchor = self.config.anchoring.sample(
                self.behavior_graph.snapshots[0])
        self.propagations = {self.anchor: (0, ())}

        initial_snapshot = nx.Graph()
        initial_snapshot.add_node(self.anchor)
        self.snapshots = [initial_snapshot]

        absorption = Absorption(self.behavior_graph, self.config.absorption)
        emission = Emission(self.behavior_graph,
                            self.config.emission, self.anchor)
        transmission = Transmission(self.config.transmission)

        for t in range(len(self.behavior_graph.snapshots)-1):
            snapshot = copy.deepcopy(self.snapshots[t])
            emissions = emission.sample(self.propagations, t, self)
            absorptions = absorption.sample(self.propagations, t)
            for edge in emissions:
                if edge not in absorptions:
                    continue
                else:
                    source, target = edge
                    if target in self.propagations:
                        continue
                    cover_time = absorptions[edge]
                    transmitted = transmission.sample(
                        edge, emission.emitters[source], t
                    )
                    if transmitted:
                        self.propagations[target] = (
                            cover_time, edge)

            for _, (cover_time, edge) in self.propagations.items():
                if len(edge) > 0 and cover_time == t:
                    snapshot.add_edge(*edge)
            self.snapshots.append(snapshot)
        return self.snapshots


class PropagationConfigMetrics(Metrics):
    def __init__(self, config: Union[PropagationConfig, None] = None) -> None:
        super().__init__()
        self.compute(config)

    def compute(self, config: Union[PropagationConfig, None] = None) -> dict:
        self._["emission_rate"] = (
            config.emission.rate(0, 0, 0) if config is not None else 0
        )
        self._["emission_latency"] = (
            config.emission.latency(0, 0) if config is not None else 0
        )
        self._["emission_duration"] = (
            config.emission.duration(0, 0) if config is not None else 0
        )
        self._["absorption_rate"] = (
            config.absorption.rate(0, 0) if config is not None else 0
        )
        self._["absorption_latency"] = (
            config.absorption.latency(0, 0) if config is not None else 0
        )
        self._["transmission_rate"] = (
            config.transmission.rate((0, 0), 0, 0) if config is not None else 0
        )
        return self._


class PropagationMetrics(Metrics):
    """Metrics that measure characteristics of an observed propagation.

    - `coverage`: The fraction of nodes in the behavior graph that are covered by the propagation at each time step.
    - `serial_interval`: The time it takes for a covered node to pass on the state to the next non-covered node. Is a measure of infection latency.
        [1]: The Interval between Successive Cases of an Infectious Disease (https://academic.oup.com/aje/article/158/11/1039/162725)
        [2]: Transmission Dynamics and Control of Severe Acute Respiratory Syndrom (https://www.science.org/doi/abs/10.1126/science.1086616)
    - `incidence_prevalence_ratio`: The number of new infections per snapshot divided by the number of infected nodes in the prior snapshot.
        Measures the propagation speed per snapshot in relation to the number of already covered nodes.
        [3]: Epidemiological metrics and benchmarks for a transition in the HIV epidemic (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6201869/)
    - `efficiency_index`: For each behavior graph, we can find a propagation that is most efficient for this behavior graph (i.e. covers all nodes in the
        shortest possible time). For a node, the shortest possible time for becoming covered depends on the distance to the anchor. 
        Intuitively, the most efficient propagation is the one that covers the anchor first, then directly the nodes that are closest to the anchor, and so on.
        On the other hand, if no propagation happens (not more than the anchor is covered), over the course of the observation, this is the slowest propagation possible.
        The efficiency index is computed on node level by calculating the deviation (sum over all edges) to the fastest possible propagation for each node.
        Thus, each propagation can be put somewhere in this spectrum between fastest (0 deviations) and slowest (n deviations).
        An efficiency index of 1 refers to the fastest possible propagation, while the efficiency index of 0 refers to the slowest possible propagation.
        [3]: Relationship between information propagation efficiency and its influence factors in online social network (https://ieeexplore.ieee.org/document/8248560)
            - The metric definition here is very different, but as motivation interesting (Social Media context)
        [4]: Information propagation in the Bitcoin network (https://ieeexplore.ieee.org/abstract/document/6688704)
            - Also not the same, but interesting as motivation (inspiration)
    - `depth_breadth_ratio`: Depth measures over how many layers (hops) from the anchor the propagation is spread in the network. Breadth measures how many nodes are covered
        at each hop. The ratio indicates how many hops where necessary to cover all nodes in the network in relation to the number of nodes covered
        at each hop. For instance, when there is a path graph of length n, the ratio is n/1, and for a star graph the ratio is 1/n.
        This can be used to differentiate between deep and shallow propagations. A deep propagation might indicate that a state is adopted and redistributed by many nodes,
        while shallow propagations consists only of a few (but maybe strong) distributors (spreaders).
        [5]: The impact of network clustering and assortativity on epidemic behaviour (https://pubmed.ncbi.nlm.nih.gov/19948179/)
        [6]: The spread of true and fake news online (https://ide.mit.edu/wp-content/uploads/2018/12/2017-IDE-Research-Brief-False-News.pdf)
    - `reproduction_ratio`: The reproduction ratio is a measure of the number of secondary cases that are generated by a single primary case in relation to the number of secondary cases
        that would have been possible. It is therefore a relative measure of the reproduction number.
        [6]: The analysis of equilibrium in malaria (https://pubmed.ncbi.nlm.nih.gov/12995455/) - first use of the reproduction number term
    - `reproduction_variance`: The variance in the reproduction ratio over all relevant nodes in the propagation. This metric measures how uniform the reproduction ratio is.
        It could be used to detect if the propagation is caused by a single node ("superspreader") or if it is caused by most nodes equally.
        [7]: Superspreaders and high variance infectious diseases (https://iopscience.iop.org/article/10.1088/1742-5468/abed44)
    - `path_criticality`: Edges in a behavior graph can be more or less critical. The critical edges can be important for the overall coverage of the propagation
        (because such edges can bridge different parts of the network and help the propagation to reach new regions or communities that it might not have otherwise reached).
        We define the cirticalness of an edge by the amount of common neighbors between the incident nodes of the edge. Intuitively, the more common neighbors two nodes have,
        the less critical the edge is (because there are other edges that can be used to propagate the state). The path criticality is the average criticality of all edges in the propagation,
        it might be used to differentiate propagation based on how many of the critical paths they used.
        [8]: Identifying critical edges in complex networks (https://www.nature.com/articles/s41598-018-32631-8)

    If `include_anchor_metric` is set to `True`, the following metrics are also computed:
        - `anchor_reachability_index`: The number of nodes reachable in the two-hop network of the anchor in relation to the number of nodes at t=0 in the behavior graph.
            This metric can be used to measure how well the anchor is connected to the rest of the network.
            Might be interesting to see how different anchoring techniques influence the propagation, as a higher connectivity of the anchor leads to higher initial propagation speed. Also,
            a higher closeness could lead to higher coverage in the end (as all nodes are not so far away from the anchor).
    """

    def __init__(
        self, propagation: Union[Propagation, None] = None, include_anchor_metric=False, local=False
    ) -> None:
        super().__init__()
        self.include_anchor_metric = include_anchor_metric
        self.compute(propagation=propagation, local=local)

    def compute_coverage(self, propagation: Propagation):
        return [
            len(propagation_graph.nodes) / len(behavior_graph.nodes)
            for propagation_graph, behavior_graph in zip(
                propagation.snapshots, propagation.behavior_graph.snapshots
            )
        ]

    def compute_serial_interval(self, propagation: Propagation):
        propagations = []
        node_start_times = {propagation.anchor: 0}
        for t, _ in enumerate(propagation.snapshots):
            if t == 0:
                continue
            _, edge_diff = propagation.diff(t - 1, t)
            for edge in edge_diff:
                source_start = (
                    # if the source node emitted before shown as covered, we set the travel time to 0
                    node_start_times[edge[0]]
                    if edge[0] in node_start_times
                    else t
                )
                propagations += [t - source_start]
                node_start_times[edge[1]] = t
        return propagations if len(propagations) > 0 else [len(propagation.snapshots)]

    def compute_incidence_prevalence_ratio(self, propagation: Propagation):
        incidence_prevalence_ratios = []
        for t, _ in enumerate(propagation.snapshots):
            if t == 0:
                continue
            _, edge_diff = propagation.diff(t - 1, t)
            incidence = len(edge_diff)
            prevalence = len(propagation.snapshots[t - 1].nodes())
            if prevalence == len(propagation.behavior_graph.snapshots[t - 1].nodes()):
                break
            incidence_prevalence_ratios.append(incidence / prevalence)
        return incidence_prevalence_ratios

    def compute_earliest_cover_times(self, propagation: Propagation, keep_anchor=False):
        earliest_cover_times = {}
        for snapshot in propagation.behavior_graph.snapshots:
            for node in list(snapshot.nodes):
                if node == propagation.anchor:
                    if not keep_anchor:
                        continue
                    earliest_cover_times[node] = 0
                if node not in earliest_cover_times:
                    earliest_cover_times[node] = math.inf
                distance = nx.shortest_path_length(
                    snapshot, propagation.anchor, node)
                earliest_cover_times[node] = min(
                    distance, earliest_cover_times[node], len(
                        propagation.snapshots)
                )
        return earliest_cover_times

    def compute_actual_cover_times(self, propagation: Propagation):
        actual_cover_times = {}
        for t, snapshot in enumerate(propagation.snapshots):
            for node in list(snapshot.nodes):
                if node in actual_cover_times:
                    continue
                actual_cover_times[node] = t
        return actual_cover_times

    def compute_latest_cover_times(
        self, propagation: Propagation, earliest_cover_times: dict
    ):
        return {
            node: len(propagation.snapshots) - earliest_time
            for node, earliest_time in earliest_cover_times.items()
        }

    def compute_cover_time_deviations(self, propagation: Propagation, keep_anchor=False):
        earliest_cover_times = self.compute_earliest_cover_times(
            propagation, keep_anchor=keep_anchor)
        latest_cover_times = self.compute_latest_cover_times(
            propagation, earliest_cover_times
        )
        actual_cover_times = self.compute_actual_cover_times(propagation)
        deviations = []
        max_deviations = []
        for node, earliest_time in earliest_cover_times.items():
            actual_cover_time = (
                actual_cover_times[node]
                if node in actual_cover_times
                else len(propagation.snapshots)
            )
            deviations.append(actual_cover_time - earliest_time)
            max_deviations.append(latest_cover_times[node])
        return deviations, max_deviations

    def compute_efficiency_index(self, propagation: Propagation):
        deviations, max_deviations = self.compute_cover_time_deviations(
            propagation=propagation)
        return 1 - (sum(deviations) / sum(max_deviations))

    def compute_reproduction_ratio(self, propagation: Propagation, keep_sinks=False):
        node_propagation_ratios = {}
        propagation_snapshot = propagation.snapshots[-1]
        for node in list(propagation_snapshot.nodes):
            num_propagations = len(
                get_neighborhood(propagation_snapshot, node))
            possible_propagations = set()
            for behavior_snapshot in propagation.behavior_graph.snapshots:
                # NOTE: Not sure if we should subtract the already infected nodes here
                neighbors = get_neighborhood(behavior_snapshot, node)
                for neighbor in neighbors:
                    possible_propagations.add(neighbor)
            num_possible_propagations = len(possible_propagations)
            # for non-anchor nodes reduce by incoming edge
            if node != propagation.anchor:
                num_propagations -= 1
                num_possible_propagations -= 1
            # filter sinks as they are not interesting as they can't propagate per definition
            if num_possible_propagations > 0 or keep_sinks:
                node_propagation_ratios[node] = (
                    num_propagations / num_possible_propagations
                ) if num_possible_propagations > 0 else 0

        # if sinks should be kept, also add the uncovered nodes with ratio 0
        if keep_sinks:
            for node in propagation.behavior_graph.snapshots[-1].nodes:
                if node not in node_propagation_ratios:
                    node_propagation_ratios[node] = 0
        return (
            list(node_propagation_ratios.values())
            if len(node_propagation_ratios) > 0
            else [0]
        )

    def compute_depth(self, propagation: Propagation):
        propagation_path_lengths = nx.shortest_path_length(
            propagation.snapshots[-1], source=propagation.anchor
        )
        return max(propagation_path_lengths.values())

    def compute_breadth(self, propagation: Propagation):
        propagation_path_length_counts = count_shortest_paths(
            propagation.snapshots[-1], source=propagation.anchor
        )
        propagation_paths = list(propagation_path_length_counts.values())
        return np.mean(propagation_paths) if len(propagation_paths) > 0 else 0

    def compute_path_criticality(self, propagation: Propagation):
        edge_criticality = []
        for t, _ in enumerate(propagation.snapshots):
            if t == 0:
                continue
            _, edge_diff = propagation.diff(t - 1, t)
            behavior_graph = propagation.behavior_graph.snapshots[t - 1]
            jaccard_coeff = nx.jaccard_coefficient(behavior_graph, edge_diff)
            for _, _, coeff in jaccard_coeff:
                edge_criticality.append(1 - coeff)

        mean_criticality = [
            coeff
            for _, _, coeff in nx.jaccard_coefficient(
                behavior_graph, propagation.behavior_graph.snapshots[0].edges()
            )
        ]
        return edge_criticality if len(edge_criticality) > 0 else [0], np.mean(
            mean_criticality
        )

    def compute_local_serial_interval(self, propagation: Propagation):
        node_serial_interval = {}
        propagation_snapshot = propagation.snapshots[-1]
        for node in propagation.behavior_graph.snapshots[-1].nodes:
            if node not in propagation_snapshot.nodes:
                node_serial_interval[node] = 0
                continue
            for t1, snapshot in enumerate(propagation.snapshots):
                if node in snapshot.nodes:
                    next_t = t1
                    for t2, next_snapshot in enumerate(propagation.snapshots[t1:]):
                        len_neighborhood = len(
                            get_neighborhood(next_snapshot, node))
                        if propagation.anchor == node and len_neighborhood == 1 or len_neighborhood == 2:
                            next_t = t2
                            break
                    node_serial_interval[node] = next_t - t1
                    break
        return list(node_serial_interval.values())

    def compute_global_metrics(self, propagation: Union[Propagation, None] = None) -> dict:
        if propagation is None:
            self._["final_coverage"] = 0
            self._["avg_coverage"] = 0
            self._["depth_breadth_ratio"] = 0
            self._["serial_interval"] = 0
            self._["incidence_prevalence_ratio"] = 0
            self._["efficiency_index"] = 0
            self._["reproduction_ratio"] = 0
            self._["reproduction_variance"] = 0
            self._["path_criticality"] = 0

            if self.include_anchor_metric:
                self._["anchor_reachability_index"] = 0

        else:
            coverage = self.compute_coverage(propagation)
            serial_interval = self.compute_serial_interval(propagation)
            reproduction_ratio = self.compute_reproduction_ratio(propagation)
            path_criticality, mean_criticality = self.compute_path_criticality(
                propagation
            )
            incidence_prevalence_ratio = self.compute_incidence_prevalence_ratio(
                propagation
            )
            self._["final_coverage"] = coverage[-1]
            self._["avg_coverage"] = np.mean(coverage)
            self._["depth_breadth_ratio"] = self.compute_depth(
                propagation
            ) / self.compute_breadth(propagation)
            self._["serial_interval"] = np.mean(serial_interval)
            self._["incidence_prevalence_ratio"] = np.mean(
                incidence_prevalence_ratio)
            self._["efficiency_index"] = self.compute_efficiency_index(
                propagation)
            self._["reproduction_ratio"] = np.mean(reproduction_ratio)
            self._["reproduction_variance"] = np.var(reproduction_ratio)
            self._["path_criticality"] = np.mean(
                path_criticality) - mean_criticality

            if self.include_anchor_metric:
                self._["anchor_reachability_index"] = num_nodes_reached_in_k_hops(
                    propagation.behavior_graph.snapshots[0],
                    propagation.anchor,
                ) / len(propagation.behavior_graph.snapshots[0].nodes())

        return self._

    def compute_local_metrics(self, propagation: Union[Propagation, None] = None) -> dict:
        if propagation is None:
            self._["coverage"] = []
            self._["efficiency_index"] = []
            self._["reproduction_ratio"] = []
            self._["serial_interval"] = []

        else:
            self._["coverage"] = [1 if propagation.snapshots[-1].has_node(
                node) else 0 for node in propagation.behavior_graph.snapshots[-1].nodes()]
            deviations, max_deviations = self.compute_cover_time_deviations(
                propagation, keep_anchor=True)
            self._["efficiency_index"] = [1 - d/max_d if max_d > 0 else 1 for d,
                                          max_d in zip(deviations, max_deviations)]
            reproduction_ratio = self.compute_reproduction_ratio(
                propagation, keep_sinks=True)
            self._["reproduction_ratio"] = reproduction_ratio
            self._["serial_interval"] = self.compute_local_serial_interval(
                propagation)

        return self._

    def compute(self, propagation: Union[Propagation, None] = None, local=False) -> dict:
        if not local:
            return self.compute_global_metrics(propagation)
        return self.compute_local_metrics(propagation)
