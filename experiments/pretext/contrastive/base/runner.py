from enum import Enum
from itertools import combinations
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.utils import (compute_means_with_significance,
                               default_runner, default_result_columns,
                               deserialize_enum_values,
                               parallelize_experiment_runs, read_results,
                               save_results, serialize_enum_values)
from src.augmentations.augmentation import PropagationAugmentation
from src.datasets.crypto.bitcoin_block_propagation import \
    BitcoinBlockPropagationDataset
from src.datasets.cybersecurity.ddos import DDoSDataset
from src.datasets.epidemics.covid19 import Covid19Dataset
from src.datasets.epidemics.waves import WavesDataset
from src.datasets.random.synthetic import SyntheticDataset
from src.datasets.social_media.fake_news import FakeNewsDataset
from src.pretext.contrastive.model import ContrastivePreTextModel
from src.pretext.contrastive.trainer import ContrastivePreTextTrainer
from src.utils.drawing import save_fig
from src.utils.objects import safely_parse_list


class PreTextModel(Enum):
    CONTRASTIVE = {
        "run": lambda dataset, augmentations: (
            model := ContrastivePreTextModel(
                node_feat_dim=1,
                edge_feat_dim=1,
                attention_layer_dims=[64],
                attention_layer_hidden_dims=[64],
            ),
            ContrastivePreTextTrainer(
                model=model, dataset=dataset, augmentations=augmentations, force_feats=1
            ).train(epochs=20),
        )[-1]
    }


def analyze_losses(results: pd.DataFrame, path: str):
    for dataset_name, dataset_df in results.groupby("dataset"):
        dataset_path = f"{path}/{dataset_name}"
        dataset_df["value"] = dataset_df["value"].apply(
            safely_parse_list)
        dataset_df["value"] = dataset_df.value.apply(
            lambda x: list(enumerate(x))).to_frame()
        # explode the column and split the tuple into two separate columns
        dataset_df = dataset_df.explode('value')
        dataset_df[['index', 'value']] = pd.DataFrame(
            dataset_df['value'].tolist(), index=dataset_df.index)
        dataset_df.reset_index(drop=True, inplace=True)

        sns.lineplot(data=dataset_df, x="index",
                     y="value", hue="augmentation")
        fig = plt.gcf()
        save_fig(fig, f"{dataset_path}/losses_linechart")

    plt.rcParams.update({'font.size': 14})
    plt.rcParams['legend.fontsize'] = 12
    metric_df = results
    metric_df["value"] = metric_df["value"].apply(
        safely_parse_list)
    metric_df["value"] = metric_df.value.apply(
        lambda x: list(enumerate(x))).to_frame()
    # explode the column and split the tuple into two separate columns
    metric_df = metric_df.explode('value')
    metric_df[['index', 'value']] = pd.DataFrame(
        metric_df['value'].tolist(), index=metric_df.index)
    metric_df.reset_index(drop=True, inplace=True)
    ax = sns.lineplot(data=metric_df, x="index",
                      y="value", hue="augmentation", errorbar=None, legend=True, linewidth=2.5)
    ax.set_axisbelow(True)
    plt.xlabel("training steps")
    plt.ylabel("training loss")
    plt.grid(True)

    legend = ax.legend_
    if legend is not None:
        legend.set_title('')
    save_fig(
        plt, f"{path}/linechart_loss_combined", svg=True)


def analyze_accuracy(results: pd.DataFrame, path: str):
    results["value"] = results["value"].astype(float)
    compute_means_with_significance(
        results=results, group_col="augmentation", path=path)
    for dataset_name, dataset_df in results.groupby("dataset"):
        dataset_path = f"{path}/{dataset_name}"
        compute_means_with_significance(
            results=dataset_df, group_col="augmentation", path=dataset_path)
        for metric, metric_df in dataset_df.groupby("metric"):
            metric_df['value'] = metric_df['value'].astype(float)
            sns.boxplot(
                data=metric_df,
                x="value",
                y="augmentation",
            )
            fig = plt.gcf()
            save_fig(fig, f"{dataset_path}/{metric}_boxplot")


def analyze_results(results: pd.DataFrame, path: str):
    results = results[results["type"] == "test"]
    results["augmentation"] = (
        results["augmentation1"] + " + " + results["augmentation2"]
    )
    analyze_accuracy(results[results["metric"] == "accuracy"], path)
    analyze_losses(results[results["metric"] == "train_losses"], path)


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
        PreTextModel.CONTRASTIVE,
    ]


def get_augmentation_name(augmentation: Union[None, PropagationAugmentation]):
    if augmentation is None:
        return "None"
    return augmentation.value


def run_experiment(exp_args):
    (
        dataset_class,
        augmentation_comb,
        models,
        recreate
    ) = exp_args

    models = deserialize_enum_values(models, PreTextModel)
    dataset = dataset_class(log_progress=True, recreate=recreate)
    results = []
    for model in models:
        print(
            f"Running {model.name} with {get_augmentation_name(augmentation_comb[0])} x {get_augmentation_name(augmentation_comb[1])} ...")
        result = model.value["run"](dataset, augmentation_comb)
        results.append(
            {
                "dataset": dataset.abbreviation,
                "model": model.name,
                "result": result,
                "more_values": {
                    "augmentation1": get_augmentation_name(
                        augmentation_comb[0]
                    ),
                    "augmentation2": get_augmentation_name(
                        augmentation_comb[1]
                    ),
                }
            }
        )
    return results


def run_experiments(recreate=False, analyze=True, execute=True, parallelize=False, **kwargs):
    path = "experiments/pretext/contrastive/base"
    results = read_results(
        path, [*default_result_columns(), "augmentation1", "augmentation2"]
    )
    if execute:
        dataset_classes = get_dataset_classes()
        models = get_models()
        models = serialize_enum_values(models)
        augmentations = [None, *list(PropagationAugmentation)]

        exp_combinations = [
            (
                dataset_class,
                list(augmentation_comb),
                models,
                recreate,
            )
            for augmentation_comb in combinations(augmentations, 2)
            for dataset_class in dataset_classes
        ]

        parallelize_experiment_runs(
            results=results,
            combinations=exp_combinations,
            run_experiment=run_experiment,
            parallelize=parallelize,
            path=path)
        save_results(results, path)
    if analyze:
        analyze_results(results, path)


if __name__ == "__main__":
    default_runner(run_experiments)
