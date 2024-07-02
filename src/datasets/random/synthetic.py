from src.datasets.dataset import Label
from src.datasets.random.random import RandomDataset
from src.generation.propagation.absorption import AbsorptionConfig
from src.generation.propagation.anchoring import (
    Anchoring, BetweennessCentralityAnchoring, ClosenessCentralityAnchoring,
    DegreeCentralityAnchoring)
from src.generation.propagation.emission import EmissionConfig
from src.generation.propagation.propagation import PropagationConfig
from src.generation.propagation.transmission import TransmissionConfig
from src.utils.numbers import clamped_sample_fn

MAX_T = 14
# normalized the values such that magnitudes are similar base mean
DEF_TRANSMISSION_RATE = 0.2
DEF_EMISSION_RATE = 0.5
DEF_ABSORPTION_RATE = 0.5
DEF_EMISSION_LATENCY = 0
DEF_ABSORPTION_LATENCY = 2
DEF_EMISSION_DURATION = 2
DEF_ANCHORING = 0.2


def get_rate(mean, factor=0):
    std = 0.1  # of 1
    mean = mean + factor * std
    return clamped_sample_fn(
        mean, std, bounds=[mean - std, mean + std], round_int=False
    )


def get_latency(mean, factor=0):
    std = 1  # of 10
    mean = mean + factor * std
    return clamped_sample_fn(
        mean, std, bounds=[max(0, mean - std), min(mean + std, MAX_T - 1)]
    )


def get_duration(mean, factor=0):
    std = 1  # of 10
    mean = mean + factor * std
    return clamped_sample_fn(
        mean, std, bounds=[max(1, mean - std), min(mean + std, MAX_T - 1)]
    )


def get_anchoring(mean, factor=0, method=None):
    if method == None:
        return Anchoring()
    std = 0.2  # of 1
    mean = mean + factor * std
    quantile = clamped_sample_fn(
        mean, std, bounds=[max(0, mean - std), min(mean + std, 1.0)], round_int=False
    )
    if method == "betweenness":
        return BetweennessCentralityAnchoring(quantile)
    elif method == "degree":
        return DegreeCentralityAnchoring(quantile)
    elif method == "closeness":
        return ClosenessCentralityAnchoring(quantile)


def get_synthetic_labels(
    rate_factor: float = 2.0,
    latency_factor: float = 2.0,
    duration_factor: float = 2.0,
    anchoring_factor: float = 0.0,
    anchoring_method=None,
    def_rate_factor: float = 0.0,
    def_latency_factor: float = 0.0,
    def_duration_factor: float = 0.0,
):
    """Adjusts the parameters of the Label enum using the given factors.

    - def_*_factor: the base factors to be used for all labels
    - *_factor: the factors to be used for the specific label that differentiated over this factor
    """
    rate_label = Label("rate", lambda x: PropagationConfig(
        anchoring=get_anchoring(
            DEF_ANCHORING, anchoring_factor, anchoring_method),
        emission=EmissionConfig(
            rate=get_rate(DEF_EMISSION_RATE),
            latency=get_latency(DEF_EMISSION_LATENCY),
            duration=get_duration(DEF_EMISSION_DURATION, def_duration_factor),
        ),
        absorption=AbsorptionConfig(
            rate=get_rate(DEF_ABSORPTION_RATE),
            latency=get_latency(DEF_ABSORPTION_LATENCY, def_latency_factor),
        ),
        transmission=TransmissionConfig(
            rate=get_rate(DEF_TRANSMISSION_RATE, def_rate_factor + rate_factor)
        ),
    ))
    latency_label = Label('latency', lambda x: PropagationConfig(
        anchoring=get_anchoring(
            DEF_ANCHORING, anchoring_factor, anchoring_method),
        emission=EmissionConfig(
            rate=get_rate(DEF_EMISSION_RATE),
            latency=get_latency(DEF_EMISSION_LATENCY),
            duration=get_duration(
                DEF_EMISSION_DURATION, def_duration_factor),
        ),
        absorption=AbsorptionConfig(
            rate=get_rate(DEF_ABSORPTION_RATE),
            latency=get_latency(
                DEF_ABSORPTION_LATENCY, def_latency_factor + latency_factor
            ),
        ),
        transmission=TransmissionConfig(
            rate=get_rate(DEF_TRANSMISSION_RATE, def_rate_factor)
        ),
    ))
    duration_label = Label('duration', lambda x: PropagationConfig(
        anchoring=get_anchoring(
            DEF_ANCHORING, anchoring_factor, anchoring_method),
        emission=EmissionConfig(
            rate=get_rate(DEF_EMISSION_RATE),
            latency=get_latency(
                DEF_EMISSION_LATENCY, def_duration_factor),
            duration=get_duration(
                DEF_EMISSION_DURATION, def_duration_factor + duration_factor
            ),
        ),
        absorption=AbsorptionConfig(
            rate=get_rate(DEF_ABSORPTION_RATE),
            latency=get_latency(DEF_ABSORPTION_LATENCY),
        ),
        transmission=TransmissionConfig(
            rate=get_rate(DEF_TRANSMISSION_RATE, def_rate_factor)
        ),
    ))

    return [rate_label, latency_label, duration_label]


class SyntheticDataset(RandomDataset):
    """A dataset that includes completely synthetic propagations based on the configurations."""

    def __init__(
        self,
        rate_factor: float = 2.0,
        latency_factor: float = 2.0,
        duration_factor: float = 2.0,
        anchoring_factor: float = 0.0,
        anchoring_method=None,
        def_rate_factor: float = 0.0,
        def_latency_factor: float = 0.0,
        def_duration_factor: float = 0.0,
        num_samples=1000,
        **kwargs
    ) -> None:
        # Trains on the default labels and tests on the modified labels
        train_labels = get_synthetic_labels()
        test_labels = get_synthetic_labels(
            rate_factor=rate_factor,
            latency_factor=latency_factor,
            duration_factor=duration_factor,
            anchoring_factor=anchoring_factor,
            anchoring_method=anchoring_method,
            def_rate_factor=def_rate_factor,
            def_latency_factor=def_latency_factor,
            def_duration_factor=def_duration_factor,
        )

        name = kwargs.pop("name", "Synthetic")
        abbreviation = kwargs.pop("abbreviation", "SYN")
        num_nodes = kwargs.pop("num_nodes", [20, 50])
        super().__init__(
            **kwargs,
            name=name,
            abbreviation=abbreviation,
            labels=train_labels,
            train_labels=train_labels,
            test_labels=test_labels,
            num_nodes=num_nodes,
            num_samples_per_label=round(num_samples / len(train_labels)),
            max_t=MAX_T,
        )
