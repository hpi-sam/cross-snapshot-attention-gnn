from enum import Enum
from typing import Tuple
import numpy as np

from src.datasets.dataset import Label
from src.datasets.random.random import RandomDataset
from src.generation.propagation.propagation import PropagationConfig
from src.generation.propagation.transmission import TransmissionConfig
from src.generation.behavior.behavior import BehaviorGraph
from src.utils.numbers import clamped_sample_fn
from src.utils.graph import sample_subnetwork, sample_node_based_on_centrality_quantile


def get_regular_transmission_rate_fn():
    # return lambda *_: 0.0
    # [8] - Network traffic and internet speed is best modeled using a log normal distribution
    return clamped_sample_fn(
        mean=-2.5, std=0.5, bounds=[None, 0.1], fn=np.random.lognormal, round_int=False
    )


def get_relay_transmission_rate_fn():
    # [2] - Relay nodes have a higher transmission rate than regular nodes and provide high speed links for the block propagation
    return clamped_sample_fn(mean=0.7, std=0.1, bounds=[0.6, 0.8], round_int=False)


def sample_relay_nodes_fn(behavior_graph: BehaviorGraph, decentralized=True):
    graph = behavior_graph.snapshots[0]
    num_relay_nodes = clamped_sample_fn(
        mean=0.3, std=0.1, bounds=[0.2, 0.4], round_int=False
    )()
    # We pick a root relay node that is near the center of the graph
    relay_root_centrality = clamped_sample_fn(
        mean=0.9, std=0.1, bounds=[0.8, 1.0], round_int=False
    )()
    root = sample_node_based_on_centrality_quantile(
        graph, relay_root_centrality)
    relay_network = sample_subnetwork(
        graph=graph, amount=num_relay_nodes, root=root, complete=decentralized
    )
    # The differences between centralized and decentralized are minor, thus, we add all edges between
    # the relay nodes to the graph to make the difference more visible
    # if decentralized:
    #     complete = nx.complete_graph(relay_network.nodes())
    #     relay_network = nx.compose(relay_network, complete)
    #     for i, snapshot in enumerate(behavior_graph.snapshots):
    #         behavior_graph.snapshots[i] = nx.compose(snapshot, complete)

    def sample_transmission_rate_for_edge(edge: Tuple[int, int]):
        if relay_network.has_edge(*edge):
            rate = get_relay_transmission_rate_fn()()
            return rate
        rate = get_regular_transmission_rate_fn()()
        return rate

    return lambda edge, *_: sample_transmission_rate_for_edge(edge)


class BitcoinBlockPropagationLabel(Enum):
    """Labels for the BitcoinBlockPropagation dataset."""

    REGULAR = Label(
        "regular",
        PropagationConfig(
            transmission=TransmissionConfig(
                rate=get_regular_transmission_rate_fn()),
        ),
    )
    RELAY = Label(
        "relay",
        lambda behavior_graph: PropagationConfig(
            # [2] - All relay nodes are connected to each other
            transmission=TransmissionConfig(
                rate=sample_relay_nodes_fn(behavior_graph, decentralized=True)
            ),
        ),
    )


class BitcoinBlockPropagationDataset(RandomDataset):
    """A dataset that includes Bitcoin block propagations. The behavior graphs serve as instances of the bitcoin network.

    Bitcoin is a decentralized digital currency that is built on a blockchain, which is a distributed ledger that records all the transactions made on the network.
    These transactions are broadcast to the Bitcoin network, where they are verified by so called miners and added to the blockchain.
    Block propagation delay in Bitcoin refers to the amount of time it takes for a new block to be propagated throughout the network. When a new block is created,
    it must be distributed to all the nodes on the network so that they can update their own replicas of the blockchain.
    If this process takes too long, it can lead to delays in transaction processing and can create opportunities for malicious actors to attempt double-spending attacks.
    The propagation delay is influenced by several factors, including the size of the block, the number of nodes on the network, and the quality of the connections between the nodes.
    Overall, reducing block propagation delay is critical for ensuring that the Bitcoin network can process transactions in a timely and reliable manner.
    To mitigate block propagation delay, Bitcoin nodes use a variety of techniques, including the use of relay nodes, which are high-bandwidth nodes that help to quickly distribute new blocks throughout the network.

    This dataset adopts characteristics of the different bandwidths of the nodes in the Bitcoin network. Specifically, relay nodes with higher bandwidths increase the transmission rate
    of propagations. The idea is to classify block propagations into different classes based on the bandwidth of the nodes.
    For instance, we want to detect whether a block was sent via the relay network or between normal nodes. Afterward, the propagation could further be analyzed, e.g., to
    identify optimal relay node distribution in the network. The propagations are an instance of the near-normal phenomenon, if only a subset of the nodes are different.
    Application: It could be used to find the most efficient way to distribute relay nodes in the Bitcoin network.

    Literature used:
    - [1] Bitcoin: A Peer-to-Peer Electronic Cash System (https://bitcoin.org/bitcoin.pdf)
    - [2] A Theoretical Model for Block Propagation Analysis in Bitcoin Network (https://theblockchaintest.com/uploads/resources/file-490533279520.pdf)
    - [3] Information propagation in the Bitcoin network (https://www.gsd.inesc-id.pt/~ler/docencia/rcs1314/papers/P2P2013_041.pdf)
    - [4] Bandwidth-Efficient Transaction Relay in Bitcoin (https://arxiv.org/pdf/1905.10518.pdf)
    - [5] Identifying Impacts of Protocol and Internet Development on the Bitcoin Network (https://ieeexplore.ieee.org/abstract/document/9219639)
    - [6] On Scaling Decentralized Blockchains (https://www.researchgate.net/publication/292782219_On_Scaling_Decentralized_Blockchains_A_Position_Paper)
    - [7] An Analysis of Bitcoin's Throughput Bottlenecks, Potential Solutions, and Future Prospects (https://github.com/fresheneesz/bitcoinThroughputAnalysis#bandwidth)
    - [8] On the Distribution of Traffic Volumes in the Internet and its Implications (https://arxiv.org/abs/1902.03853)
    """

    def __init__(self, **kwargs) -> None:
        labels = [label.value for label in BitcoinBlockPropagationLabel]
        # [2] suggests that ER graphs are a good approximation for the Bitcoin network structure
        # behavior_graph_dist = {"er": 1}

        name = kwargs.pop("name", "BTC-BlockPropagation")
        abbreviation = kwargs.pop("abbreviation", "BTC-BP")
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
