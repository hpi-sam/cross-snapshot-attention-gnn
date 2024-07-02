from enum import Enum
import random

from src.datasets.dataset import Label
from src.datasets.random.random import RandomDataset
from src.generation.propagation.propagation import PropagationConfig
from src.generation.propagation.emission import EmissionConfig
from src.generation.propagation.absorption import AbsorptionConfig
from src.generation.propagation.transmission import TransmissionConfig
from src.utils.numbers import clamped_sample_fn

MAX_T = 14
base_transmission_rate = 0.1


def sample_rate():
    mean = random.choice(range(1, 10)) / 10
    return clamped_sample_fn(mean, 0.1, round_int=False)


def sample_latency():
    mean = random.choice(range(1, 10))
    return clamped_sample_fn(mean, 3)


def sample_duration():
    mean = random.choice(range(3, 10))
    return clamped_sample_fn(mean, 3)


class PreTextLabel(Enum):
    """Labels for the PreText dataset."""

    EMISSION_RATE = Label(
        "emission_rate",
        lambda x: PropagationConfig(
            emission=EmissionConfig(rate=sample_rate()),
            transmission=TransmissionConfig(rate=base_transmission_rate),
        ),
    )
    EMISSION_LATENCY = Label(
        "emission_latency",
        lambda x: PropagationConfig(
            emission=EmissionConfig(latency=sample_latency()),
            transmission=TransmissionConfig(rate=base_transmission_rate),
        ),
    )
    EMISSION_DURATION = Label(
        "emission_duration",
        lambda x: PropagationConfig(
            emission=EmissionConfig(duration=sample_duration()),
            transmission=TransmissionConfig(rate=base_transmission_rate),
        ),
    )
    ABSORPTION_RATE = Label(
        "absorption_rate",
        lambda x: PropagationConfig(
            absorption=AbsorptionConfig(rate=sample_rate()),
            transmission=TransmissionConfig(rate=base_transmission_rate),
        ),
    )
    ABSORPTION_LATENCY = Label(
        "absorption_latency",
        lambda x: PropagationConfig(
            absorption=AbsorptionConfig(latency=sample_latency()),
            transmission=TransmissionConfig(rate=base_transmission_rate),
        ),
    )


class PreTextDataset(RandomDataset):
    """A dataset that includes propagations for the pre-text tasks."""

    def __init__(self, **kwargs) -> None:
        labels = [label.value for label in PreTextLabel]
        name = kwargs.pop("name", "PreText")
        abbreviation = kwargs.pop("abbreviation", "Pre")
        num_nodes = kwargs.pop("num_nodes", [20, 50])
        super().__init__(
            **kwargs,
            name=name,
            abbreviation=abbreviation,
            labels=labels,
            num_nodes=num_nodes,
            num_samples_per_label=round(1000 / len(labels)),
            max_t=MAX_T,
        )
