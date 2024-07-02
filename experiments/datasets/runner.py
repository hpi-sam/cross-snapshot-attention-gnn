from src.datasets.crypto.bitcoin_block_propagation import BitcoinBlockPropagationDataset
from src.datasets.cybersecurity.ddos import DDoSDataset
from src.datasets.epidemics.covid19 import Covid19Dataset
from src.datasets.epidemics.waves import WavesDataset
from src.datasets.social_media.fake_news import FakeNewsDataset
from src.datasets.random.synthetic import SyntheticDataset
from src.datasets.random.pretext import PreTextDataset
from src.datasets.analyzer import DatasetAnalyzer


def run_experiments(recreate=True):
    path = "experiments/datasets"
    datasets = [
        SyntheticDataset(log_progress=True, recreate=recreate),
        Covid19Dataset(log_progress=True, recreate=recreate),
        FakeNewsDataset(log_progress=True, recreate=recreate),
        BitcoinBlockPropagationDataset(log_progress=True, recreate=recreate),
        DDoSDataset(log_progress=True, recreate=recreate),
        WavesDataset(log_progress=True, recreate=recreate),
        PreTextDataset(log_progress=True, recreate=recreate),
    ]

    for dataset in datasets:
        DatasetAnalyzer(dataset=dataset, path=path)


if __name__ == "__main__":
    run_experiments()
