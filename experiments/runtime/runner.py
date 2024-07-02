from enum import Enum
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import src.models.metrics.metrics as metrics_baselines
from experiments.utils import (compute_means_with_significance,
                               default_runner, deserialize_enum_values,
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
from src.utils.drawing import save_fig


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
    complete_test_results = results[results["type"] == "runtime"]
    compute_means_with_significance(
        results=results, group_col="model", path=path, filter_colum="runtime")
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['legend.fontsize'] = 12
    for metric, metric_df in complete_test_results.groupby("metric"):
        sns.boxplot(data=metric_df, x="model", y="value")
        save_fig(plt, f"{path}/boxplot_{metric}")
        mean_df = (
            metric_df.groupby(["model"])
            .agg({"value": "mean"})
            .reset_index()
        )
        mean_df["value"] = mean_df["value"] // 1000
        model_order = [model.name for model in get_models()]
        ax = sns.barplot(data=mean_df, x="model", y="value", order=model_order)
        ax.set_axisbelow(True)
        legend = ax.legend_
        if legend is not None:
            legend.set_title('')
        plt.xticks([])
        plt.xlabel("model")
        plt.ylabel("runtime (s)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=color)
                   for color in sns.color_palette()]
        plt.legend(handles, model_order)
        plt.grid(True)
        save_fig(plt, f"{path}/barplot_{metric}", svg=True)
    plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})


def get_dataset_classes():
    return [
        SyntheticDataset,
        Covid19Dataset,
        FakeNewsDataset,
        BitcoinBlockPropagationDataset,
        DDoSDataset,
        WavesDataset,
    ]


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


def compute_propagation_metric_runtime(dataset, local=False):
    s = time.time()
    for sample in [*dataset.train, *dataset.test]:
        PropagationMetrics(propagation=sample.propagation, local=local)
    e = time.time()
    return (e - s) * 1000


def run_experiment(exp_args):
    (
        dataset_class,
        models,
        recreate,
    ) = exp_args

    models = deserialize_enum_values(models, ClassificationModel)
    dataset = dataset_class(log_progress=True, recreate=recreate)

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
                "dataset": dataset.abbreviation,
                "model": model.name,
                "result": result,
            }
        )
    return results


def run_experiments(recreate=False, analyze=True, execute=True, parallelize=False, **kwargs):
    if parallelize:
        raise ValueError("Parallelizing will influence runtime.")
    path = "experiments/runtime"
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
            )
            for dataset_class in dataset_classes
        ]

        parallelize_experiment_runs(
            results=results,
            combinations=combinations,
            run_experiment=run_experiment,
            parallelize=parallelize,
            path=path, metric_filter=["runtime"])

        save_results(results, path)
    if analyze:
        analyze_results(results, path)


if __name__ == "__main__":
    default_runner(run_experiments)
