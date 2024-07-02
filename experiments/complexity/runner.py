from enum import Enum
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import src.models.metrics.metrics as metrics_baselines
from experiments.utils import (compute_means_with_significance,
                               default_runner, default_result_columns,
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
                                    TemporalSnapshotListTransform)
from src.generation.propagation.propagation import PropagationMetrics
from src.models.base_gnn import BaseGNN
from src.models.cross_snapshot_attention import CrossSnapshotAttentionNet
from src.training.trainer import GNNTrainer
from src.utils.drawing import get_grid_size, save_fig


class ClassificationModel(Enum):
    METRICS_LOG = {
        "run": lambda dataset: metrics_baselines.execute(
            dataset=dataset,
            metrics_flavor="propagation",
            method="linear_regression",
            profile=True,
        ),
    }
    METRICS_MLP = {
        "run": lambda dataset: metrics_baselines.execute(
            dataset=dataset,
            metrics_flavor="propagation",
            method="mlp",
            profile=True,
        ),
    }
    METRICS_XGB = {
        "run": lambda dataset: metrics_baselines.execute(
            dataset=dataset,
            metrics_flavor="propagation",
            method="xgboost",
            profile=True,
        ),
    }
    METRICS_GAT = {
        "run": lambda dataset: (
            model := BaseGNN(
                layer_sizes=[len(PropagationMetrics(
                    local=True).get_all_keys()), 64, 64, 64, len(dataset.labels)],
                conv_type="GAT",
            ),
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=PropagationMetricsGraphTransform,
                profile=True,
                profile_keywords=[["forward_only"]]
            ).train(epochs=30),
        )[-1]
    }
    ATTR_GAT = {
        "run": lambda dataset: (
            model := BaseGNN(
                layer_sizes=[dataset.max_t, 64, 64, 64, len(dataset.labels)],
                edge_feat_dim=dataset.max_t,
                conv_type="GAT",
            ),
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalGraphAttributesTransform,
                profile=True,
                profile_keywords=[["forward_only"]]
            ).train(epochs=30),
        )[-1]
    }
    CSA_BASE = {
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
                force_feats=1,
                profile=True,
                profile_keywords=[["forward_only"], ["classify"]]
            ).train(epochs=30),
        )[-1]
    }
    CSA_EXT = {
        "run": lambda dataset: (
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
                profile=True,
                profile_keywords=[["forward_only"], ["classify"]]
            ).train(epochs=30),
        )[-1]
    }


def analyze_results(results: pd.DataFrame, path: str):
    results.loc[:, "value"] = results["value"].round(4)
    results = results[(results["type"] == "test") |
                      (results["type"] == "runtime")]
    compute_means_with_significance(
        results=results, group_col="model", path=path)
    compute_means_with_significance(
        results=results, group_col="num_nodes", path=path)
    for dataset_name, dataset_df in results.groupby("dataset"):
        dataset_path = f"{path}/{dataset_name}"
        compute_means_with_significance(
            results=dataset_df, group_col="model", path=dataset_path)
        compute_means_with_significance(
            results=dataset_df, group_col="num_nodes", path=dataset_path)

        # plot linechart for each metric
        for metric, metric_df in dataset_df.groupby("metric"):
            type_count = len(metric_df.groupby("graph")["graph"].nunique())
            nrows, ncols = get_grid_size(type_count)
            # plot for each graph type
            fig, ax = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(8 * nrows, 8 * ncols)
            )
            ax = ax.flat if nrows * ncols > 1 else [ax]
            for j, (graph, graph_type_df) in enumerate(metric_df.groupby("graph")):
                graph_type_df["value"] = graph_type_df["value"] * 100
                sns.lineplot(data=graph_type_df, x="num_nodes",
                             y="value", hue="model", ax=ax[j], errorbar=None, linewidth=2.5)
                ax[j].set_axisbelow(True)
                ax[j].set_title(f"{graph}")
                ax[j].set_xlabel("#nodes")
                ax[j].set_ylabel(f"{metric} (%)" if metric !=
                                 "runtime" else "runtime (s)")
                ax[j].grid(True)
            save_fig(fig, f"{dataset_path}/linechart_{metric}", svg=True)

            # add another combined chart
            plt.rcParams.update({'font.size': 14})
            plt.rcParams['legend.fontsize'] = 12
            metric_df["value"] = metric_df["value"] * \
                100 if metric != "runtime" else metric_df["value"] / 1000
            ax = sns.lineplot(data=metric_df, x="num_nodes",
                              y="value", hue="model", errorbar=None, legend=metric == "accuracy", linewidth=2.5)
            legend = ax.legend_
            if legend is not None:
                legend.set_title('')
            ax.set_axisbelow(True)
            plt.xlabel("#nodes")
            plt.ylabel(f"{metric} (%)" if metric !=
                       "runtime" else "runtime (s)")
            plt.grid(True)
            save_fig(
                plt, f"{dataset_path}/linechart_{metric}_combined", svg=True)
            plt.rcParams.update(
                {'font.size': plt.rcParamsDefault['font.size']})


def get_models():
    return [
        ClassificationModel.METRICS_LOG,
        ClassificationModel.METRICS_MLP,
        ClassificationModel.METRICS_XGB,
        ClassificationModel.METRICS_GAT,
        ClassificationModel.ATTR_GAT,
        ClassificationModel.CSA_BASE,
        ClassificationModel.CSA_EXT,
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


def create_dataset(dataset_class, graph_type: str, num_nodes: int, recreate=False):
    dataset = dataset_class(
        log_progress=True,
        train_behavior_distribution={graph_type: 1} if graph_type is not None else None,
        test_behavior_distribution={graph_type: 1} if graph_type is not None else None,
        num_nodes=num_nodes,
        recreate=recreate,
    )
    return dataset


def compute_propagation_metric_runtime(dataset, local=False):
    s = time.time()
    for sample in [*dataset.train, *dataset.test]:
        PropagationMetrics(propagation=sample.propagation, local=local)
    e = time.time()
    return (e - s) * 1000


def run_experiment(exp_args):
    (
        dataset_class,
        num_nodes,
        graph_type,
        models,
        recreate,
    ) = exp_args

    models = deserialize_enum_values(models, ClassificationModel)
    dataset = create_dataset(
        dataset_class=dataset_class, graph_type=graph_type, num_nodes=num_nodes, recreate=recreate
    )

    # we need to include the metric computation time in the runtime
    global_metric_time = compute_propagation_metric_runtime(dataset)
    local_metric_time = compute_propagation_metric_runtime(dataset, local=True)

    results = []
    for model in models:
        print(f"Running {model.name}...")
        add_time = 0
        if model.name == "METRICS_GAT":
            add_time = local_metric_time
        elif model.name in ["METRICS_LOG", "METRICS_MLP", "METRICS_XGB"]:
            add_time = global_metric_time
        result = model.value["run"](dataset)
        result["runtime"].add_to_property("runtime", add_time)
        results.append(
            {
                "dataset": dataset.abbreviation.split("&")[0],
                "model": model.name,
                "result": result,
                "more_values": {
                    "graph": graph_type,
                    "num_nodes": num_nodes,
                },
            }
        )
    return results


def run_experiments(recreate=False, analyze=True, execute=True, parallelize=False, **kwargs):
    path = "experiments/complexity"
    results = read_results(
        path, [*default_result_columns(), "graph", "num_nodes"])
    if execute:
        models = get_models()
        num_nodes = [10, 20, 30, 40, 50]
        dataset_classes = get_dataset_classes()
        graph_types = ["er", "ws", "ba", "exp"]

        models = serialize_enum_values(models)
        combinations = [
            (
                dataset_class,
                nodes,
                graph_type,
                models,
                recreate,
            )
            for graph_type in graph_types
            for nodes in num_nodes
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
