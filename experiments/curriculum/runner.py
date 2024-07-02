from enum import Enum

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.utils import (compute_means_with_significance,
                               default_result_columns, default_runner,
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
from src.datasets.transform import TemporalSnapshotListTransform
from src.models.cross_snapshot_attention import CrossSnapshotAttentionNet
from src.training.curriculum import (complexity_curriculum,
                                     complexity_diversity_curriculum,
                                     complexity_density_curriculum,
                                     diversity_curriculum,
                                     diversity_density_curriculum,
                                     density_curriculum,
                                     train_with_curriculum)
from src.training.trainer import GNNTrainer
from src.utils.drawing import get_grid_size, save_fig


class ClassificationModel(Enum):
    CSA_EXT = {
        "init": lambda dataset: CrossSnapshotAttentionNet(
            node_feat_dim=4,
            edge_feat_dim=1,
            attention_layer_dims=[64],
            attention_layer_hidden_dims=[64],
            output_dim=len(dataset.labels),
        ),
        "train": lambda model, dataset, epochs: (
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListTransform,
            ).train(epochs=epochs),
        )[-1],
        "test": lambda model, dataset: (
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListTransform,
                training_share=0,
            ).train(epochs=1),
        )[-1],
    }


def analyze_results(results: pd.DataFrame, path: str):
    compute_means_with_significance(
        results=results, group_col="model", path=path)
    compute_means_with_significance(
        results=results, group_col="curriculum", path=path)
    # plot boxplots containing all types for all datasets and metrics
    for dataset, dataset_df in results.groupby("dataset"):
        dataset_path = f"{path}/{dataset}"
        compute_means_with_significance(
            results=dataset_df, group_col="model", path=dataset_path)
        compute_means_with_significance(
            results=dataset_df, group_col="curriculum", path=dataset_path)
        for metric, metric_df in dataset_df.groupby("metric"):
            type_count = len(metric_df.groupby("type")["type"].nunique())
            nrows, ncols = get_grid_size(type_count)
            fig, ax = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(8 * nrows, 8 * ncols)
            )
            ax = ax.flat if nrows * ncols > 1 else [ax]
            for j, (type_name, type_df) in enumerate(metric_df.groupby("type")):
                sns.boxplot(data=type_df,
                            ax=ax[j], x="curriculum", y="value", hue="model")
                ax[j].set_xticklabels(ax[j].get_xticklabels(), rotation=90)
                ax[j].set_title(f"{type_name}")
            save_fig(fig, f"{dataset_path}/boxplot_{metric}")


def get_models():
    return [
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


def run_experiment(exp_args):
    (
        dataset_class,
        curriculum_data,
        incremental,
        models,
        recreate,
    ) = exp_args

    models = deserialize_enum_values(models, ClassificationModel)
    test_dataset = dataset_class(log_progress=True)

    if curriculum_data is None:
        # it's enough to run the baseline just once
        if not incremental:
            return []
        curriculum = None
        duration_per_lesson = 30
    else:
        curriculum = curriculum_data[0](
            lesson_duration=curriculum_data[1], num_epochs=30, incremental=incremental)
        duration_per_lesson = curriculum_data[1]

    results = []
    for model in models:
        print(f"Running {model.name}...")
        trained_model = model.value["init"](test_dataset)
        if curriculum is None:
            # just train as normal
            model.value["train"](trained_model, test_dataset, 30)
        else:
            # use curriculum for training
            train_with_curriculum(train_fn=lambda dataset, epochs: model.value["train"](
                trained_model, dataset, epochs), curriculum=curriculum, base_dataset=test_dataset,
                incremental=incremental, recreate=recreate)

        result = model.value["test"](trained_model, test_dataset)
        results.append(
            {
                "dataset": test_dataset.abbreviation,
                "model": model.name,
                "result": result,
                "more_values": {
                    "curriculum": f"{curriculum.name}+{duration_per_lesson}+{'INC' if incremental else 'DEC'}" if curriculum is not None else 'None',
                },
            }
        )
    return results


def run_experiments(recreate=False, analyze=True, execute=True, parallelize=False, **kwargs):
    path = "experiments/curriculum"
    results = read_results(
        path, columns=[*default_result_columns(), "curriculum"]
    )
    if execute:
        dataset_classes = get_dataset_classes()
        models = get_models()
        curricula = [
            None,
            (complexity_curriculum, 5),
            (diversity_curriculum, 5),
            (density_curriculum, 5),
            (complexity_diversity_curriculum, 1),
            (complexity_density_curriculum, 1),
            (diversity_density_curriculum, 1)
        ]

        models = serialize_enum_values(models)
        combinations = [
            (
                dataset_class,
                curriculum,
                incremental,
                models,
                recreate,
            )
            for curriculum in curricula
            for incremental in [True, False]
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
