from enum import Enum

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.utils import (compute_means_with_significance,
                               default_runner, deserialize_enum_values,
                               parallelize_experiment_runs, read_results,
                               save_results, serialize_enum_values, default_result_columns)
from src.datasets.crypto.bitcoin_block_propagation import BitcoinBlockPropagationDataset
from src.datasets.cybersecurity.ddos import DDoSDataset
from src.datasets.dataset import Dataset
from src.datasets.epidemics.covid19 import Covid19Dataset
from src.datasets.epidemics.waves import WavesDataset
from src.datasets.random.synthetic import SyntheticDataset
from src.datasets.social_media.fake_news import FakeNewsDataset
from src.datasets.transform import TemporalSnapshotListTransform
from src.models.cross_snapshot_attention import CrossSnapshotAttentionNet
from src.models.prompting import PrePromptingModel
from src.pretext.predictive.trainer import load_from_predictive_pretext_model
from src.training.trainer import GNNTrainer
from src.utils.drawing import get_grid_size, save_fig


class ClassificationModel(Enum):
    BASE = {
        "run": lambda dataset, amount: (
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
    PROMPT = {
        "run": lambda dataset, amount: execute_prompting_model(dataset, amount, indomain=False)
    }
    PROMPT_INDOM = {
        "run": lambda dataset, amount: execute_prompting_model(dataset, amount, indomain=True)
    }


def analyze_results(results: pd.DataFrame, path: str):
    results = results[results["type"] == "test"]
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['legend.fontsize'] = 16
    compute_means_with_significance(
        results=results, group_col="model", path=path)
    for dataset_name, dataset_df in results.groupby("dataset"):
        dataset_path = f"{path}/{dataset_name}"
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
                sns.lineplot(data=type_df, x="amount",
                             y="value", hue="model", ax=ax[j])
                ax[j].set_title(f"{type_name}")
            save_fig(fig, f"{dataset_path}/linechart_{metric}")

            # add another combined chart
            metric_df["value"] = metric_df["value"] * 100
            ax = sns.lineplot(data=metric_df, x="amount",
                              y="value", hue="model", errorbar=None, legend=dataset_name in ["COV-19"], linewidth=2.5)
            ax.set_axisbelow(True)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.xlabel("#masked snapshots")
            plt.ylabel(f"{metric} (%)" if metric !=
                       "runtime" else "runtime (s)")
            plt.grid(True)

            legend = ax.legend_
            if legend is not None:
                legend.set_title('')
            save_fig(
                plt, f"{dataset_path}/linechart_{metric}_combined", svg=True)
    plt.rcParams.update(
        {'font.size': plt.rcParamsDefault['font.size']})


def get_datasets(recreate=False, mask: dict = None):
    return [
        SyntheticDataset(
            log_progress=True, recreate=recreate, **(mask if mask is not None else {})
        ),
        Covid19Dataset(
            log_progress=True, recreate=recreate, **(mask if mask is not None else {})
        ),
        FakeNewsDataset(
            log_progress=True, recreate=recreate, **(mask if mask is not None else {})
        ),
        BitcoinBlockPropagationDataset(
            log_progress=True, recreate=recreate, **(mask if mask is not None else {})
        ),
        DDoSDataset(
            log_progress=True, recreate=recreate, **(mask if mask is not None else {})
        ),
        WavesDataset(
            log_progress=True, recreate=recreate, **(mask if mask is not None else {})
        ),
    ]


def get_models():
    return [
        ClassificationModel.BASE,
        ClassificationModel.PROMPT,
        ClassificationModel.PROMPT_INDOM,
    ]


dataset_helper = {
    "Synthetic": SyntheticDataset,
    "COVID-19": Covid19Dataset,
    "FakeNews": FakeNewsDataset,
    "BTC-BlockPropagation": BitcoinBlockPropagationDataset,
    "DDoS": DDoSDataset,
    "Waves": WavesDataset,
}


def execute_prompting_model(dataset: Dataset, mask_amount=int, indomain=False):
    # load the model that prompts the masked snapshots
    prompting_model = load_from_predictive_pretext_model(
        model_name="prompting_model",
        model_version="node",
        pretext_dataset=dataset if indomain else None,
        pretext_dropout=0.0,
        encoder_only=False,
        node_feat_dim=1,
        edge_feat_dim=1,
        attention_layer_dims=[64],
        attention_layer_hidden_dims=[64],
        train_epochs=20,
        train_feats=1,
        freeze=False,
    )
    classifier_model = CrossSnapshotAttentionNet(
        node_feat_dim=1,
        edge_feat_dim=1,
        attention_layer_dims=[64],
        attention_layer_hidden_dims=[64],
        output_dim=len(dataset.labels),
    )
    model = PrePromptingModel(
        predictor=prompting_model,
        classifier=classifier_model,
        prediction_amount=mask_amount,
    )
    result = GNNTrainer(
        model=model,
        dataset=dataset,
        transform=TemporalSnapshotListTransform,
        force_feats=1
    ).train(epochs=30)
    return result


def run_experiment(exp_args):
    (
        dataset_name,
        models,
        amount,
        recreate,
    ) = exp_args

    models = deserialize_enum_values(models, ClassificationModel)
    dataset = dataset_helper[dataset_name](log_progress=True, recreate=recreate, **{
        "mask_method": "end", "mask_amount": amount})
    results = []
    for model in models:
        print(f"Running {model.name}...")
        result = model.value["run"](dataset, amount)
        results.append(
            {
                "dataset": dataset.abbreviation,
                "model": model.name,
                "result": result,
                "more_values": {
                    "amount": amount,
                },
            }
        )
    return results


def run_experiments(recreate=False, analyze=True, execute=True, parallelize=False, **kwargs):
    path = "experiments/pretext/prompting"
    results = read_results(path, [*default_result_columns(), "amount"])
    if execute:
        # train the pretext models to avoid concurrency issues
        for dataset in [None, *get_datasets(recreate=recreate)]:
            load_from_predictive_pretext_model(
                model_name="prompting_model",
                model_version="node",
                pretext_dataset=dataset,
                encoder_only=False,
                node_feat_dim=1,
                edge_feat_dim=1,
                attention_layer_dims=[64],
                attention_layer_hidden_dims=[64],
                train_epochs=20,
                train_feats=1,
                force_retrain=True,
                freeze=False,
            )

        max_t = 8
        models = get_models()
        models = serialize_enum_values(models)
        combinations = [
            (
                dataset_name,
                models,
                amount,
                recreate,
            )
            for amount in range(1, (max_t // 2) + 1)
            for dataset_name in dataset_helper.keys()
        ]

        parallelize_experiment_runs(
            results=results,
            combinations=combinations,
            run_experiment=run_experiment,
            parallelize=parallelize,
            path=path,
            timeout_in_min=90,
        )

        save_results(results, path)
    if analyze:
        results = results[results['model'].isin(
            ["BASE", "PROMPT"])]
        results['model'] = results['model'].replace("BASE", "CSA")
        results['model'] = results['model'].replace("PROMPT", "CSA_PROMPT")
        analyze_results(results, path)


if __name__ == "__main__":
    default_runner(run_experiments)
