from enum import Enum
from pathlib import Path

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
from src.pretext.predictive.model import (
    NaivePredictivePreTextModel,
    PredictivePreTextModel,
)
from src.pretext.predictive.trainer import PredictivePreTextTrainer
from src.utils.drawing import get_grid_size, save_fig
from src.utils.path import remove_after_last_slash


class PreTextModel(Enum):
    PREDICT_NAIVE = {
        "run": lambda dataset: (
            model := NaivePredictivePreTextModel(),
            PredictivePreTextTrainer(
                model=model, dataset=dataset, do_backward=False
            ).train(epochs=1),
        )[-1]
    }
    PREDICT_PROP = {
        "run": lambda dataset: (
            model := PredictivePreTextModel(
                node_feat_dim=1,
                edge_feat_dim=1,
                attention_layer_dims=[64],
                attention_layer_hidden_dims=[64],
                version="prop",
                max_num_nodes=dataset.get_max_num_nodes(),
            ),
            PredictivePreTextTrainer(
                model=model,
                dataset=dataset,
                force_feats=1,
            ).train(epochs=20),
        )[-1]
    }
    PREDICT_NODE = {
        "run": lambda dataset: (
            model := PredictivePreTextModel(
                node_feat_dim=1,
                edge_feat_dim=1,
                attention_layer_dims=[64],
                attention_layer_hidden_dims=[64],
                version="node",
                max_num_nodes=dataset.get_max_num_nodes(),
            ),
            PredictivePreTextTrainer(
                model=model,
                dataset=dataset,
                force_feats=1,
            ).train(epochs=20),
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

    # plot boxplots containing all types for all datasets and metrics
    for dataset, dataset_df in results.groupby("dataset"):
        dataset_path = f"{path}/{dataset}"
        compute_means_with_significance(
            results=dataset_df, group_col="model", path=dataset_path)
        for metric, metric_df in dataset_df.groupby("metric"):
            Path(remove_after_last_slash(f"{dataset_path}/")).mkdir(
                parents=True, exist_ok=True)
            type_count = len(metric_df.groupby("type")["type"].nunique())
            fig, _ = plt.subplots(figsize=(12, 4 * type_count))
            sns.boxplot(data=metric_df, x="value", y="type", hue="model")
            fig.suptitle(f"{dataset} - {metric}")
            save_fig(fig, f"{dataset_path}/boxplot_{metric}")

    plt.rcParams.update({'font.size': 14})
    plt.rcParams['legend.fontsize'] = 12
    results["value"] = results["value"] * 100
    r = results[results["metric"] == "precision"]
    r = r.groupby(
        list(r.columns.difference(["value", "metric", "type"])), as_index=False
    )
    r = r["value"].mean()
    sns.heatmap(
        r.pivot("model", "dataset", "value"),
        cmap="YlGn",
        vmin=0,
        vmax=100,
        annot=True,
    )
    save_fig(plt, f"{path}/heatmap_test_precision", svg=True)

    r = results[results["metric"] == "recall"]
    r = r.groupby(
        list(r.columns.difference(["value", "metric", "type"])), as_index=False
    )
    r = r["value"].mean()
    sns.heatmap(
        r.pivot("model", "dataset", "value"),
        cmap="YlGn",
        vmin=0,
        vmax=100,
        annot=True,
    )
    save_fig(plt, f"{path}/heatmap_test_recall",  svg=True)


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
        PreTextModel.PREDICT_NAIVE,
        PreTextModel.PREDICT_PROP,
        PreTextModel.PREDICT_NODE,
    ]


def run_experiment(exp_args):
    (
        dataset_class,
        models,
        recreate,
    ) = exp_args

    models = deserialize_enum_values(models, PreTextModel)
    dataset = dataset_class(log_progress=True, recreate=recreate)
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
    path = "experiments/pretext/predictive/base"
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
            path=path)

        save_results(results, path)
    if analyze:
        analyze_results(results, path)


if __name__ == "__main__":
    default_runner(run_experiments)
