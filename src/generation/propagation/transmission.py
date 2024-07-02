import random
from typing import Callable, Tuple, Union


class TransmissionConfig:
    def __init__(
        self, rate: Union[float, Callable[[Tuple[int, int], int, int], float]] = 1.0
    ) -> None:
        self.rate = (
            rate if callable(
                rate) else lambda edge, t, source_emission_start: rate
        )  # type: Callable[[Tuple[int, int], int, int], int]


class Transmission:
    """Helper to sample the transmission state for a given edge.

    During the transmission process, we decide if for an edge that received an emission and absorbed the incoming event signal,
    the event signal leads to event cover or not.
    """

    def __init__(self, config: TransmissionConfig) -> None:
        self.config = config

    def sample(self, edge: Tuple[int, int], source_emission_start: int, t: int):
        """Samples transmission state for an edge at time t.

        Uses the transmission rate for the edge at time t to decide if the edge is transmitted at time t.

        :param edge: edge to sample the transmission state for
        :param t: time to sample the transmission state for
        :return: bool indicating if the edge is transmitted at time t
        """
        edge_transmission_rate = self.config.rate(
            edge, t, source_emission_start)
        p = random.random()
        return p <= edge_transmission_rate
