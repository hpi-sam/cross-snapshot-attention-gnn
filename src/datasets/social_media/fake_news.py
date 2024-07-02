from enum import Enum
from typing import Callable
import numpy as np
import networkx as nx

from src.datasets.dataset import Label
from src.datasets.random.random import RandomDataset
from src.generation.propagation.propagation import PropagationConfig
from src.generation.propagation.emission import EmissionConfig
from src.generation.propagation.anchoring import DegreeCentralityAnchoring
from src.utils.numbers import clamped_sample_fn


def official_true_news_pdf(layer: int):
    # [2] - Exponential decay in the reposting probability as it was found that true news have a large layer 1 size but all succeeding layers are small
    return np.exp(-((layer / 2) ** 2)) - 0.3


def non_official_true_news_pdf(layer: int):
    # [2] - Exponential decay in the reposting probability, but now the sizes of the succeeding layers are a bit larger
    return np.exp(-((layer / 5) ** 2)) - 0.6


def fake_news_pdf(layer: int):
    # [2] -  Small linear decrease in the reposting probability, but in general more uniform in the layers with a bit of noise
    return (
        (-0.03) * layer
        + 0.3
        + clamped_sample_fn(mean=0, std=0.1,
                            bounds=[-0.03, 0.03], round_int=False)()
    )


def get_emission_rate_fn(pdf: Callable[[int], float]):
    def sample_fn(
        node: int,
        t: int,
        _: int,
        *args,
    ):
        layer = nx.shortest_path_length(
            args[0].snapshots[t], source=args[0].anchor, target=node
        )
        # emission rate depends on the layer of the node (distance to anchor)
        return pdf(layer + 1)

    return sample_fn


class FakeNewsLabel(Enum):
    """Labels for the FakeNews dataset."""

    FAKE = Label(
        "fake",
        PropagationConfig(
            anchoring=DegreeCentralityAnchoring(0.2),
            emission=EmissionConfig(
                rate=get_emission_rate_fn(fake_news_pdf), duration=1
            ),
        ),
    )
    TRUE_NON_OFFICIAL = Label(
        "true_non_official",
        PropagationConfig(
            anchoring=DegreeCentralityAnchoring(0.2),
            emission=EmissionConfig(
                rate=get_emission_rate_fn(non_official_true_news_pdf),
                duration=1,
            ),
        ),
    )
    TRUE_OFFICIAL = Label(
        "true_official",
        PropagationConfig(
            anchoring=DegreeCentralityAnchoring(0.2),
            emission=EmissionConfig(
                rate=get_emission_rate_fn(official_true_news_pdf),
                duration=1,
            ),
        ),
    )


class FakeNewsDataset(RandomDataset):
    """A dataset that includes propagations of fake news and true news. The behavior graphs serve as social media networks.

    Social networks have tremendously accelerated the exchange of information around the world. However, the spread of fake news has become a serious problem in recent years.
    These fake news, which can be fabricated stories or statements yet without confirmation, circulate online pervasively through the conduit offered by on-line social networks.
    Without proper debunking and verification, the fast circulation of fake news can largely reshape public opinion and undermine modern society.
    It was found that fake news spread significantly different than true news in terms of speed as well as deepness of propagation (number of repostings).

    We use these characteristics for fake news and true news to generate a dataset that includes propagations of fake news and true news.
    For the behavior graphs, we use a distribution that matches the structure of social media networks.

    Literature used:
    - [1] The spread of true and fake news online (https://ide.mit.edu/wp-content/uploads/2018/12/2017-IDE-Research-Brief-False-News.pdf)
    - [2] Fake news propagates differently from real news even at early stages of spreading (https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-020-00224-z#Bib1)
    - [3] The LDBC Social Network Benchmark (https://arxiv.org/pdf/2001.02299.pdf)
    - [4] The Anatomy of the Facebook Social Graph (https://arxiv.org/abs/1111.4503)
    - [5] Preferential Attachment in Online Networks: Measurement and Explanations (https://arxiv.org/pdf/1303.6271.pdf)
    """

    def __init__(self, **kwargs) -> None:
        labels = [label.value for label in FakeNewsLabel]
        name = kwargs.pop("name", "FakeNews")
        abbreviation = kwargs.pop("abbreviation", "FAKE")
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
