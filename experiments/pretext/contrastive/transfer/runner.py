from enum import Enum

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.utils import (compute_means_with_significance,
                               default_runner, deserialize_enum_values,
                               parallelize_experiment_runs, read_results,
                               save_results, serialize_enum_values)
from src.datasets.crypto.bitcoin_block_propagation import BitcoinBlockPropagationDataset
from src.datasets.cybersecurity.ddos import DDoSDataset
from src.datasets.epidemics.covid19 import Covid19Dataset
from src.datasets.epidemics.waves import WavesDataset
from src.datasets.random.synthetic import SyntheticDataset
from src.datasets.social_media.fake_news import FakeNewsDataset
from src.datasets.transform import TemporalSnapshotListTransform
from src.models.cross_snapshot_attention import CrossSnapshotAttentionNet
from src.pretext.contrastive.trainer import load_from_contrastive_pretext_model
from src.training.trainer import GNNTrainer
from src.utils.drawing import save_fig


class PreTextModel(Enum):
    UNTRAINED = {
        "run": lambda dataset: (
            model := CrossSnapshotAttentionNet(
                node_feat_dim=1,
                edge_feat_dim=1,
                attention_layer_dims=[64],
                attention_layer_hidden_dims=[64],
                output_dim=len(dataset.labels),
            ),
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListTransform,
                force_feats=1
            ).train(epochs=30),
        )[-1]
    }
    PRE_BASE = {
        "run": lambda dataset: (
            model := load_from_contrastive_pretext_model(
                model_name="encoder",
                encoder_only=True,
                node_feat_dim=1,
                edge_feat_dim=1,
                attention_layer_dims=[64],
                attention_layer_hidden_dims=[64],
                output_dim=len(dataset.labels),
                train_epochs=20,
                train_feats=1,
            ),
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListTransform,
                force_feats=1
            ).train(epochs=30),
        )[-1]
    }
    PRE_INDOM = {
        "run": lambda dataset: (
            model := load_from_contrastive_pretext_model(
                model_name="encoder",
                encoder_only=True,
                pretext_dataset=dataset,
                node_feat_dim=1,
                edge_feat_dim=1,
                attention_layer_dims=[64],
                attention_layer_hidden_dims=[64],
                output_dim=len(dataset.labels),
                train_epochs=20,
                train_feats=1,
            ),
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListTransform,
                force_feats=1
            ).train(epochs=30),
        )[-1]
    }


def analyze_results(results: pd.DataFrame, path: str):
    results.loc[:, "value"] = results["value"].round(4)
    compute_means_with_significance(
        results=results, group_col="model", path=path)
    for dataset_name, dataset_df in results.groupby("dataset"):
        dataset_path = f"{path}/{dataset_name}"
        compute_means_with_significance(
            results=dataset_df, group_col="model", path=dataset_path)
        for type_name, type_df in dataset_df.groupby("type"):
            for metric, metric_df in type_df.groupby("metric"):
                type_path = f"{dataset_path}/{type_name}"
                sns.boxplot(
                    data=metric_df,
                    x="model",
                    y="value",
                )
                fig = plt.gcf()
                save_fig(fig, f"{type_path}/{metric}_boxplot")


def get_datasets(recreate=False):
    return [
        SyntheticDataset(log_progress=True, recreate=recreate),
        Covid19Dataset(log_progress=True, recreate=recreate),
        FakeNewsDataset(log_progress=True, recreate=recreate),
        BitcoinBlockPropagationDataset(log_progress=True, recreate=recreate),
        DDoSDataset(log_progress=True, recreate=recreate),
        WavesDataset(log_progress=True, recreate=recreate),
    ]


def get_models():
    return [
        PreTextModel.UNTRAINED,
        PreTextModel.PRE_BASE,
        PreTextModel.PRE_INDOM,
    ]


def get_dataset_classes():
    return [
        SyntheticDataset,
        Covid19Dataset,
        FakeNewsDataset,
        BitcoinBlockPropagationDataset,
        DDoSDataset,
        WavesDataset,
    ]


def run_experiment(exp_args):
    (
        dataset_class,
        models,
    ) = exp_args

    models = deserialize_enum_values(models, PreTextModel)
    dataset = dataset_class(log_progress=True)
    results = []
    for model in models:
        print(f"Running {model.name}...")
        result = model.value["run"](dataset)
        results.append(
            {
                "dataset": dataset.abbreviation,
                "model": model.name,
                "result": result,
            }
        )
    return results


def run_experiments(recreate=False, analyze=True, execute=True, parallelize=False, **kwargs):
    path = "experiments/pretext/contrastive/transfer"
    results = read_results(path)
    if execute:

        # train the pretext models to avoid concurrency issues
        for dataset in [None, *get_datasets(recreate=recreate)]:
            load_from_contrastive_pretext_model(
                model_name="encoder",
                encoder_only=True,
                pretext_dataset=dataset,
                node_feat_dim=1,
                edge_feat_dim=1,
                attention_layer_dims=[64],
                attention_layer_hidden_dims=[64],
                train_epochs=20,
                train_feats=1,
                force_retrain=True
            )

        dataset_classes = get_dataset_classes()
        models = get_models()
        models = serialize_enum_values(models)
        combinations = [
            (
                dataset_class,
                models,
            )
            for dataset_class in dataset_classes
        ]

        parallelize_experiment_runs(
            results=results,
            combinations=combinations,
            run_experiment=run_experiment,
            parallelize=parallelize,
            path=path)

        save_results(results, path)
    if analyze:
        analyze_results(results, path)


if __name__ == "__main__":
    default_runner(run_experiments)
