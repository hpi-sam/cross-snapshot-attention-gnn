from enum import Enum
from typing import Tuple
import random

from src.datasets.dataset import Label
from src.datasets.random.random import RandomDataset
from src.generation.propagation.propagation import PropagationConfig
from src.generation.propagation.emission import EmissionConfig
from src.generation.propagation.transmission import TransmissionConfig
from src.generation.behavior.behavior import BehaviorGraph
from src.utils.numbers import clamped_sample_fn


def get_regular_transmission_rate_fn():
    return clamped_sample_fn(mean=0.3, std=0.1, bounds=[0.1, 0.5], round_int=False)


def sample_attacked_nodes_fn(behavior_graph: BehaviorGraph):
    attacked_nodes = {}

    def sample_transmission_rate_for_edge(edge: Tuple[int, int], t: int):
        graph = behavior_graph.snapshots[t - 1]
        num_attacked_nodes = clamped_sample_fn(
            mean=0.2, std=0.1, bounds=[0.1, 0.3], round_int=False
        )()
        if t not in attacked_nodes:
            attacked_nodes[t] = list(
                random.sample(
                    graph.nodes(), int(num_attacked_nodes * len(graph.nodes()))
                )
            )
        if edge[0] in attacked_nodes[t] or edge[1] in attacked_nodes[t]:
            return 0
        return get_regular_transmission_rate_fn()()

    return lambda edge, t, _: sample_transmission_rate_for_edge(edge, t)


class DDoSLabel(Enum):
    """Labels for the DDoS dataset."""

    REGULAR = Label(
        "regular",
        PropagationConfig(
            emission=EmissionConfig(latency=1),
            transmission=TransmissionConfig(
                rate=get_regular_transmission_rate_fn()),
        ),
    )
    ATTACKED = Label(
        "attacked",
        lambda behavior_graph: PropagationConfig(
            emission=EmissionConfig(latency=1),
            transmission=TransmissionConfig(
                rate=sample_attacked_nodes_fn(behavior_graph)
            ),
        ),
    )


class DDoSDataset(RandomDataset):
    """A dataset that includes propagations of computer node communication. The behavior graphs serve as instances of computer networks.

    In a DDoS (Distributed Denial of Service) attack, a malicious actor sends a large amount of traffic to a target system, such as a web server,
    in order to overload the system and make it unavailable to legitimate users. The attacker can use a variety of methods to send this traffic, such as sending a large number of requests to the server,
    or sending a large number of requests to a server that is connected to the target server. The attacker can also use a botnet,
    which is a network of computers that have been infected with malware and are controlled by the attacker. The attacker can use the botnet to send the traffic to the target system.

    We can combine this with our notion of propagations. We assume there is some critical information that should be propagated through the network (the event).
    Now, in one part of the propagations some nodes are attacked as part of a DDoS attack such that they neither receive nor transmit the event.
    The idea is that a model should detect that in some propagations a part of the network was affected by this.

    Literature used:
    - [1] - A taxonomy of DDoS attack and DDoS defense mechanisms (https://dl.acm.org/doi/abs/10.1145/997150.997156)
    """

    def __init__(self, **kwargs) -> None:
        labels = [label.value for label in DDoSLabel]
        name = kwargs.pop("name", "DDoS")
        abbreviation = kwargs.pop("abbreviation", "DDoS")
        num_nodes = kwargs.pop("num_nodes", [20, 50])
        super().__init__(
            **kwargs,
            name=name,
            abbreviation=abbreviation,
            labels=labels,
            num_nodes=num_nodes,
            num_samples_per_label=round(1000 / len(labels)),
            max_t=14,
        )
