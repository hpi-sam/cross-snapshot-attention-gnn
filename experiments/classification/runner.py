from enum import Enum

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import src.models.metrics.metrics as metrics_baselines
from experiments.utils import (compute_means_with_significance, default_runner,
                               deserialize_enum_values,
                               parallelize_experiment_runs, read_results,
                               save_results, serialize_enum_values)
from src.datasets.crypto.bitcoin_block_propagation import \
    BitcoinBlockPropagationDataset
from src.datasets.cybersecurity.ddos import DDoSDataset
from src.datasets.epidemics.covid19 import Covid19Dataset
from src.datasets.epidemics.waves import WavesDataset
from src.datasets.random.synthetic import SyntheticDataset
from src.datasets.social_media.fake_news import FakeNewsDataset
from src.datasets.transform import (PropagationMetricsGraphTransform,
                                    TemporalGraphAttributesTransform,
                                    TemporalSnapshotListTransform,
                                    TemporalSnapshotListOneHotTransform)
from src.generation.propagation.propagation import PropagationMetrics
from src.models.base_gnn import BaseGNN
from src.models.cross_snapshot_attention import CrossSnapshotAttentionNet
from src.training.trainer import GNNTrainer
from src.utils.drawing import get_grid_size, save_fig
from DySAT.models.model import DySAT, get_model_args
from Roland.model import ROLAND


class ClassificationModel(Enum):
    METRICS_LOG = {
        "run": lambda dataset, path: metrics_baselines.execute(
            dataset=dataset,
            metrics_flavor="propagation",
            method="linear_regression",
        ),
    }
    METRICS_MLP = {
        "run": lambda dataset, path: metrics_baselines.execute(
            dataset=dataset,
            metrics_flavor="propagation",
            method="mlp",
        ),
    }
    METRICS_XGB = {
        "run": lambda dataset, path: metrics_baselines.execute(
            dataset=dataset,
            metrics_flavor="propagation",
            method="xgboost",
            path=f"{path}/{dataset.abbreviation}",
        ),
    }
    METRICS_GAT = {
        "run": lambda dataset, path: (
            model := BaseGNN(
                layer_sizes=[len(PropagationMetrics(
                    local=True).get_all_keys()), 64, 64, 64, len(dataset.labels)],
                conv_type="GAT",
            ),
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=PropagationMetricsGraphTransform,
            ).train(epochs=30),
        )[-1]
    }
    ATTR_GCN = {
        "run": lambda dataset, path: (
            model := BaseGNN(
                layer_sizes=[dataset.max_t, 64, 64, 64, len(dataset.labels)],
                conv_type="GCN",
            ),
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalGraphAttributesTransform,
            ).train(epochs=30),
        )[-1]
    }
    ATTR_GAT = {
        "run": lambda dataset, path: (
            model := BaseGNN(
                layer_sizes=[dataset.max_t, 64, 64, 64, len(dataset.labels)],
                edge_feat_dim=dataset.max_t,
                conv_type="GAT",
            ),
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalGraphAttributesTransform,
            ).train(epochs=30),
        )[-1]
    }
    ATTR_GAT_NO_EDGE = {
        "run": lambda dataset, path: (
            model := BaseGNN(
                layer_sizes=[dataset.max_t, 64, 64, 64, len(dataset.labels)],
                edge_feat_dim=None,
                conv_type="GAT",
            ),
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalGraphAttributesTransform,
            ).train(epochs=30),
        )[-1]
    }
    ATTR_GAT_NO_EDGE_SINGLE = {
        "run": lambda dataset, path: (
            model := BaseGNN(
                layer_sizes=[dataset.max_t, 64, len(dataset.labels)],
                edge_feat_dim=None,
                conv_type="GAT",
            ),
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalGraphAttributesTransform,
            ).train(epochs=30),
        )[-1]
    }
    CSA_BASE = {
        "run": lambda dataset, path: (
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
    CSA_EXT = {
        "run": lambda dataset, path: (
            model := CrossSnapshotAttentionNet(
                node_feat_dim=4,
                edge_feat_dim=1,
                attention_layer_dims=[64],
                attention_layer_hidden_dims=[64],
                output_dim=len(dataset.labels),
            ),
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListTransform,
            ).train(epochs=30),
        )[-1]
    }
    CSA = {
        "run": lambda dataset, path: (
            model := CrossSnapshotAttentionNet(
                node_feat_dim=14,
                edge_feat_dim=14,
                attention_layer_dims=[64],
                attention_layer_hidden_dims=[64],
                output_dim=len(dataset.labels),
            ),
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListOneHotTransform,
            ).train(epochs=30),
        )[-1]
    }
    DY_SAT = {
        "run": lambda dataset, path: (
            args := get_model_args(),
            model := DySAT(
                args=args,
                num_features=14,
                time_length=14,
                num_classes=len(dataset.labels),
            ),
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListOneHotTransform,
            ).train(epochs=30),
        )[-1]
    }
    DY_SAT_SINGLE = {
        "run": lambda dataset, path: (
            args := get_model_args(heads='2'),
            model := DySAT(
                args=args,
                num_features=14,
                time_length=14,
                num_classes=len(dataset.labels),
            ),
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListOneHotTransform,
            ).train(epochs=30),
        )[-1]
    }
    ROLAND = {
        "run": lambda dataset, path: (
            model := ROLAND(
                input_dim=14,
                num_nodes=dataset.get_max_num_nodes() * 64,
                num_classes=len(dataset.labels),
            ),
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListOneHotTransform,
            ).train(epochs=30),
        )[-1]
    }
    ROLAND_SINGLE = {
        "run": lambda dataset, path: (
            model := ROLAND(
                input_dim=14,
                num_nodes=dataset.get_max_num_nodes() * 64,
                num_classes=len(dataset.labels),
                second_layer=False
            ),
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListOneHotTransform,
            ).train(epochs=30),
        )[-1]
    }


def analyze_results(results: pd.DataFrame, path: str):
    complete_test_results = results[results["type"] == "test"]
    compute_means_with_significance(
        results=results, group_col="model", path=path)
    # plot combined heatmap for all datasets and metrics using mean values of test results
    metric_count = len(complete_test_results.groupby(
        "metric")["metric"].nunique())
    nrows, ncols = get_grid_size(metric_count)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(8 * nrows, 8 * ncols))
    ax = ax.flat if nrows * ncols > 1 else [ax]
    for j, (metric, metric_result) in enumerate(
        complete_test_results.groupby("metric")
    ):
        mean_result = metric_result.groupby(
            list(metric_result.columns.difference(["value"])), as_index=False
        )
        mean_result = mean_result["value"].mean()
        sns.heatmap(
            mean_result.pivot("model", "dataset", "value"),
            ax=ax[j],
            cmap="YlGn",
            vmin=0,
            vmax=1,
            annot=True,
        )
        ax[j].set_title(f"{metric}")
    save_fig(fig, f"{path}/heatmap_test")

    # reduced png
    reduced = complete_test_results[complete_test_results["metric"] == "accuracy"]
    reduced = reduced[(reduced["model"] == "CSA_EXT") | (reduced["model"] == "ATTR_GAT") | (
        reduced["model"] == "METRICS_GAT") | (reduced["model"] == "METRICS_XGB")]
    reduced = reduced.groupby(
        list(reduced.columns.difference(["value"])), as_index=False
    )
    reduced = reduced["value"].mean()
    sns.heatmap(
        reduced.pivot("model", "dataset", "value"),
        cmap="YlGn",
        vmin=0,
        vmax=1,
        annot=True,
    )
    save_fig(plt, f"{path}/heatmap_test_reduced")

    # plot boxplots containing all types for all datasets and metrics
    for dataset, dataset_df in results.groupby("dataset"):
        dataset_path = f"{path}/{dataset}"
        compute_means_with_significance(
            results=dataset_df, group_col="model", path=dataset_path)
        for metric, metric_df in dataset_df.groupby("metric"):
            type_count = len(metric_df.groupby("type")["type"].nunique())
            nrows, ncols = get_grid_size(type_count)
            fig, ax = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(8 * nrows, 8 * ncols)
            )
            ax = ax.flat if nrows * ncols > 1 else [ax]
            for j, (type_name, type_df) in enumerate(metric_df.groupby("type")):
                sns.boxplot(data=type_df,
                            ax=ax[j], x="value", y="model")
                ax[j].set_title(f"{type_name}")
            save_fig(fig, f"{dataset_path}/boxplot_{metric}")


def get_dataset_classes():
    return [
        # SyntheticDataset,
        Covid19Dataset,
        FakeNewsDataset,
        BitcoinBlockPropagationDataset,
        DDoSDataset,
        WavesDataset,
    ]


def get_models():
    return [
        # ClassificationModel.METRICS_LOG,
        # ClassificationModel.METRICS_MLP,
        # ClassificationModel.METRICS_XGB,
        # ClassificationModel.METRICS_GAT,
        # ClassificationModel.ATTR_GCN,
        # ClassificationModel.ATTR_GAT,
        # ClassificationModel.ATTR_GAT_NO_EDGE,
        # ClassificationModel.ATTR_GAT_NO_EDGE_SINGLE,
        # ClassificationModel.CSA_BASE,
        # ClassificationModel.CSA_EXT,
        # ClassificationModel.CSA,
        # ClassificationModel.DY_SAT,
        ClassificationModel.DY_SAT_SINGLE,
        # ClassificationModel.ROLAND,
        # ClassificationModel.ROLAND_SINGLE,
    ]


def run_experiment(exp_args):
    (
        dataset_class,
        models,
        recreate,
        path
    ) = exp_args

    models = deserialize_enum_values(models, ClassificationModel)
    dataset = dataset_class(log_progress=True, recreate=recreate)
    results = []
    for model in models:
        print(f"Running {model.name}...")
        result = model.value["run"](dataset, path)
        results.append(
            {
                "dataset": dataset.abbreviation,
                "model": model.name,
                "result": result,
            }
        )
    return results


def run_experiments(recreate=False, analyze=True, execute=True, parallelize=False, **kwargs):
    path = "experiments/classification"
    results = read_results(path)
    if execute:
        dataset_classes = get_dataset_classes()
        models = get_models()
        models = serialize_enum_values(models)
        combinations = [
            (
                dataset_class,
                models,
                recreate,
                path
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
        results = results[results['model'].isin(
            ["CSA", "DY_SAT", "ROLAND", "ROLAND_SINGLE"])]
        analyze_results(results, path)


if __name__ == "__main__":
    default_runner(run_experiments)
