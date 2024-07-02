from enum import Enum

from src.datasets.dataset import Label
from src.datasets.random.random import RandomDataset
from src.generation.propagation.absorption import AbsorptionConfig
from src.generation.propagation.emission import EmissionConfig
from src.generation.propagation.propagation import PropagationConfig
from src.generation.propagation.transmission import TransmissionConfig
from src.utils.numbers import clamped_sample_fn


# Adds some latency noise to the waves
DEF_EMISSION_LATENCY = clamped_sample_fn(mean=0.5, std=0.5, bounds=[-1, 1])
DEF_ABSORPTION_LATENCY = clamped_sample_fn(mean=1.0, std=0.5, bounds=[1, 2])


def get_transmission_rate_fn(shift: int, duration=7):
    regular_sample_fn = clamped_sample_fn(
        mean=0.1, std=0.1, bounds=[None, 0.1], round_int=False
    )
    wave_sample_fn = clamped_sample_fn(
        mean=0.2, std=0.1, bounds=[None, 0.2], round_int=False
    )

    def sample_rate_fn(_n, t, _):
        if t < shift or t >= shift + duration:
            return regular_sample_fn()
        return wave_sample_fn()

    return sample_rate_fn


class WaveLabel(Enum):
    """Labels for the Waves dataset."""

    WAVE_ONE = Label(
        "wave_one",
        lambda *_: PropagationConfig(
            transmission=TransmissionConfig(rate=get_transmission_rate_fn(0)),
            emission=EmissionConfig(
                latency=DEF_EMISSION_LATENCY,
            ),
            absorption=AbsorptionConfig(
                latency=DEF_ABSORPTION_LATENCY,
            ),
        ),
    )
    WAVE_TWO = Label(
        "wave_two",
        lambda *_: PropagationConfig(
            transmission=TransmissionConfig(rate=get_transmission_rate_fn(7)),
            emission=EmissionConfig(
                latency=DEF_EMISSION_LATENCY,
            ),
            absorption=AbsorptionConfig(
                latency=DEF_ABSORPTION_LATENCY,
            ),
        ),
    )


class WavesDataset(RandomDataset):
    """A dataset that includes the propagations sampled during different waves.

    Certainly, propagation characteristics are not static over time, but sometimes follow certain temporal patterns. Specifically,
    in the context of epidemics, infections happen in specific waves. This dataset is a collection of propagations sampled from different waves.
    The idea is that a model should detect the time shifting of the propagations that happen in different waves.

    Literature used:
    - [1] Influenza: historical aspects of epidemics and pandemics (https://pubmed.ncbi.nlm.nih.gov/15081510/)
    - [2] Periodic recurrent waves of Covid-19 epidemics and vaccination campaign (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9114150/)
    """

    def __init__(self, **kwargs) -> None:
        labels = [label.value for label in WaveLabel]
        name = kwargs.pop("name", "Waves")
        abbreviation = kwargs.pop("abbreviation", "WAV")
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
