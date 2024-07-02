from enum import Enum
from typing import Callable, Tuple

from src.datasets.dataset import Label
from src.datasets.random.random import RandomDataset
from src.generation.propagation.absorption import AbsorptionConfig
from src.generation.propagation.emission import EmissionConfig
from src.generation.propagation.propagation import PropagationConfig
from src.generation.propagation.transmission import TransmissionConfig
from src.utils.numbers import add_percent_increase, clamped_sample_fn


# [3] - There does not seem to be any difference in emission latency (latent period) between the different strains.
DEF_EMISSION_LATENCY = clamped_sample_fn(
    mean=-1.4, ci=[0.95, -1.8, -1], bounds=[-2, 0])
# [4] - There does not seem to be any differences in emission duration (period of communicability or viral load kinetics) between the different strains.
# "That suggests in a clinical study that the increased infectiousness of Omicron might likely be due to factors related to viral and host cell interactions, rather than viral load or duration of infectivity, which has been suggested in immune escape studies"
# [5] - "SARS-CoV-2 viral kinetics are partly determined by immunity and variant but dominated by individual-level variation"
DEF_EMISSION_DURATION = clamped_sample_fn(mean=12.9, std=6.9, bounds=[7, 21])
# NOTE: We pick an arbitrary transmission rate here, important is the relative difference between the strains.
DEF_TRANSMISSION_RATE = 0.1


def get_transmission_rate_sample_fn(
    base_rate: float,
) -> Callable[[int, int, int], float]:
    base_rates = {}
    slopes = {}

    def sample_base_rate(edge: Tuple[int, int]):
        node = edge[0]
        if node not in base_rates:
            # vary the base rate per edge a bit
            noise = 0.1
            base_rates[node] = clamped_sample_fn(
                mean=base_rate,
                std=noise * base_rate,
                bounds=[
                    max(0.1, base_rate - base_rate * noise * 2),
                    min(1, base_rate + base_rate * noise * 2),
                ],
                round_int=False,
            )()
        return base_rates[node]

    def sample_slope(edge: Tuple[int, int]):
        node = edge[0]
        if node not in slopes:
            # vary the slope per node a bit
            slopes[node] = clamped_sample_fn(
                mean=0.01, std=0.005, bounds=[0.005, 0.05], round_int=False
            )()
        return slopes[node]

    def time_passed_since_emission_start(t, source_emission_start):
        return max(t - source_emission_start, 0)

    return (
        lambda edge, t, source_emission_start: sample_base_rate(edge)
        + time_passed_since_emission_start(t, source_emission_start)
        * sample_slope(edge)
        * -1
    )


class Covid19Label(Enum):
    """Labels for the Covid19 dataset."""

    ANCESTRAL = Label(
        "ancestral",
        lambda *_: PropagationConfig(
            transmission=TransmissionConfig(
                rate=get_transmission_rate_sample_fn(DEF_TRANSMISSION_RATE)
            ),
            emission=EmissionConfig(
                latency=DEF_EMISSION_LATENCY,
                duration=DEF_EMISSION_DURATION,
            ),
            # [2]
            absorption=AbsorptionConfig(
                latency=clamped_sample_fn(
                    mean=6.38, ci=[0.95, 5.79, 6.97], bounds=[2, 14]
                )
            ),
        ),
    )
    ALPHA = Label(
        "alpha",
        lambda *_: PropagationConfig(
            # [6]
            transmission=TransmissionConfig(
                rate=get_transmission_rate_sample_fn(
                    add_percent_increase(DEF_TRANSMISSION_RATE, 70)
                )
            ),
            emission=EmissionConfig(
                latency=DEF_EMISSION_LATENCY,
                duration=DEF_EMISSION_DURATION,
            ),
            # [1]
            absorption=AbsorptionConfig(
                latency=clamped_sample_fn(
                    mean=5.0, ci=[0.95, 4.94, 5.06], bounds=[2, 14]
                ),
            ),
        ),
    )
    DELTA = Label(
        "delta",
        lambda *_: PropagationConfig(
            # [7]
            transmission=TransmissionConfig(
                rate=get_transmission_rate_sample_fn(
                    add_percent_increase(DEF_TRANSMISSION_RATE, 225)
                )
            ),
            emission=EmissionConfig(
                latency=DEF_EMISSION_LATENCY,
                duration=DEF_EMISSION_DURATION,
            ),
            # [1]
            absorption=AbsorptionConfig(
                latency=clamped_sample_fn(
                    mean=4.41, ci=[0.95, 3.76, 5.05], bounds=[2, 14]
                ),
            ),
        ),
    )
    OMICRON = Label(
        "omicron",
        lambda *_: PropagationConfig(
            # [9]
            transmission=TransmissionConfig(
                rate=get_transmission_rate_sample_fn(
                    add_percent_increase(
                        DEF_TRANSMISSION_RATE, 562.5)  # 225 * 2.5
                )
            ),
            emission=EmissionConfig(
                latency=DEF_EMISSION_LATENCY,
                duration=DEF_EMISSION_DURATION,
            ),
            # [1]
            absorption=AbsorptionConfig(
                latency=clamped_sample_fn(
                    mean=3.42, ci=[0.95, 2.88, 3.96], bounds=[2, 14]
                ),
            ),
        ),
    )


class Covid19Dataset(RandomDataset):
    """A dataset that includes the propagations of COVID-19 variants. The behavior graphs serve as contact tracing networks.

    This dataset adopts the characteristics of the most relevant (concerning) COVID-19 variants / strains and their propagation characteristics.
    Specifically, mutations of the virus lead to differences between transmissibility and/or incubation time.
    By modeling this here, we get a dataset that is close to real world characteristics in propagations.

    Literature used:
    - [1] Incubation Period of COVID-19 Caused by Unique SARS-CoV-2 Strains (https://jamanetwork.com/journals/jamanetworkopen/article-abstract/2795489)
    - [2] The incubation period of COVID-19: A meta-analysis (https://www.sciencedirect.com/science/article/pii/S1201971221000813)
    - [3] Estimating the Latent Period of Coronavirus Disease 2019 (COVID-19) (https://academic.oup.com/cid/article/74/9/1678/6359063?login=false)
    - [4] Duration of COVID-19 PCR positivity for Omicron vs earlier variants (https://www.sciencedirect.com/science/article/pii/S2667038022000242)
    - [5] Quantifying the impact of immune history and variant on SARS-CoV-2 viral kinetics and infection rebound: a retrospective cohort study (https://www.medrxiv.org/content/10.1101/2022.01.13.22269257v3)
    - [6] Assessing transmissibility of SARS-CoV-2 lineage B.1.1.7 in England (https://www.nature.com/articles/s41586-021-03470-x)
    - [7] SPI-M-O: Consensus Statement on COVID-19 (https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/993321/S1267_SPI-M-O_Consensus_Statement.pdf)
    - [8] Clinical Characteristics, Transmissibility, Pathogenicity, Susceptible Populations, and Re-infectivity of Prominent COVID-19 Variants (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8947836/)
    - [9] The effective reproductive number of the Omicron variant of SARS-CoV-2 is several times relative to Delta (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8992231/)
    """

    def __init__(self, **kwargs) -> None:
        labels = [label.value for label in Covid19Label]
        name = kwargs.pop("name", "COVID-19")
        abbreviation = kwargs.pop("abbreviation", "COV-19")
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
