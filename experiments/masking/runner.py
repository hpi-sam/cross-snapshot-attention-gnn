from enum import Enum
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import xgboost as xgb

import src.models.metrics.metrics as metrics_baselines
from experiments.utils import (
    append_to_results, compute_bay_regression_per_group,
    compute_means_with_significance,
    compute_t_test_significance_from_bay_regression_results,
    default_result_columns, default_runner, deserialize_enum_values,
    parallelize_experiment_runs, plot_t_test_significance_rate, read_results,
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
from src.pretext.contrastive.trainer import load_from_contrastive_pretext_model
from src.pretext.predictive.trainer import load_from_predictive_pretext_model
from src.training.curriculum import (complexity_density_curriculum,
                                     train_with_curriculum)
from src.training.trainer import GNNTrainer
from src.utils.drawing import get_grid_size, save_fig
from DySAT.models.model import DySAT, get_model_args
from Roland.model import ROLAND


class ClassificationModel(Enum):
    METRICS_LOG = {
        "train": lambda dataset: metrics_baselines.execute(
            dataset=dataset,
            metrics_flavor="propagation",
            method="linear_regression",
            return_model=True,
        ),
        "test": lambda model, dataset: metrics_baselines.test(
            dataset=dataset,
            metrics_flavor="propagation",
            method="linear_regression",
            model=model,
        ),
        "save": lambda model, path: joblib.dump(model, f"{path}/metrics_log.joblib"),
        "load": lambda path: joblib.load(f"{path}/metrics_log.joblib"),
    }
    METRICS_MLP = {
        "train": lambda dataset: metrics_baselines.execute(
            dataset=dataset,
            metrics_flavor="propagation",
            method="mlp",
            return_model=True,
        ),
        "test": lambda model, dataset: metrics_baselines.test(
            dataset=dataset, metrics_flavor="propagation", method="mlp", model=model
        ),
        "save": lambda model, path: torch.save(
            model, f"{path}/metrics_mlp.pt"),
        "load": lambda path: torch.load(f"{path}/metrics_mlp.pt"),

    }
    METRICS_XGB = {
        "train": lambda dataset: metrics_baselines.execute(
            dataset=dataset,
            metrics_flavor="propagation",
            method="xgboost",
            return_model=True,
        ),
        "test": lambda model, dataset: metrics_baselines.test(
            dataset=dataset, metrics_flavor="propagation", method="xgboost", model=model
        ),
        "save": lambda model, path: model.save_model(f"{path}/metrics_xgb.model"),
        "load": lambda path: (model := xgb.XGBClassifier(), model.load_model(f"{path}/metrics_xgb.model"))[-2],
    }
    METRICS_GAT = {
        "train": lambda dataset: (
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
        ),
        "test": lambda model, dataset: (
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=PropagationMetricsGraphTransform,
                training_share=0,
            ).train(epochs=1),
        )[-1],
        "save": lambda model, path: torch.save(
            model, f"{path}/metrics_gat.pt"),
        "load": lambda path: torch.load(f"{path}/metrics_gat.pt"),
    }
    ATTR_GAT = {
        "train": lambda dataset: (
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
        ),
        "test": lambda model, dataset: (
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalGraphAttributesTransform,
                training_share=0,
            ).train(epochs=1),
        )[-1],
        "save": lambda model, path: torch.save(
            model, f"{path}/attr_gat.pt"),
        "load": lambda path: torch.load(f"{path}/attr_gat.pt"),
    }
    CSA_BASE = {
        "train": lambda dataset: (
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
            ).train(epochs=30),
        ),
        "test": lambda model, dataset: (
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListTransform,
                training_share=0,
                force_feats=1,
            ).train(epochs=1),
        )[-1],
        "save": lambda model, path: torch.save(
            model, f"{path}/csa_base.pt"),
        "load": lambda path: torch.load(f"{path}/csa_base.pt"),
    }
    CSA_EXT = {
        "train": lambda dataset: (
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
        ),
        "test": lambda model, dataset: (
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListTransform,
                training_share=0,
            ).train(epochs=1),
        )[-1],
        "save": lambda model, path: torch.save(
            model, f"{path}/csa_ext.pt"),
        "load": lambda path: torch.load(f"{path}/csa_ext.pt"),
    }
    CSA_CON = {
        "train": lambda dataset: (
            model := load_from_contrastive_pretext_model(
                model_name="encoder",
                encoder_only=True,
                force_load=True,
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
        ),
        "test": lambda model, dataset: (
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListTransform,
                training_share=0,
                force_feats=1,
            ).train(epochs=1),
        )[-1],
        "save": lambda model, path: torch.save(
            model, f"{path}/csa_con.pt"),
        "load": lambda path: torch.load(f"{path}/csa_con.pt"),
    }
    CSA_PRED = {
        "train": lambda dataset: (
            model := load_from_predictive_pretext_model(
                model_name="encoder",
                model_version="node",
                encoder_only=True,
                force_load=True,
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
        ),
        "test": lambda model, dataset: (
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListTransform,
                training_share=0,
                force_feats=1,
            ).train(epochs=1),
        )[-1],
        "save": lambda model, path: torch.save(
            model, f"{path}/csa_pred.pt"),
        "load": lambda path: torch.load(f"{path}/csa_pred.pt"),
    }
    CSA_CURR = {
        "train": lambda dataset: (
            model := CrossSnapshotAttentionNet(
                node_feat_dim=1,
                edge_feat_dim=1,
                attention_layer_dims=[64],
                attention_layer_hidden_dims=[64],
                output_dim=len(dataset.labels),
            ),
            train_with_curriculum(train_fn=lambda curr_dataset, epochs: GNNTrainer(
                model=model,
                dataset=curr_dataset,
                transform=TemporalSnapshotListTransform,
                force_feats=1
            ).train(epochs=epochs), curriculum=complexity_density_curriculum(incremental=True), base_dataset=dataset,
                incremental=True),
        ),
        "test": lambda model, dataset: (
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListTransform,
                training_share=0,
                force_feats=1,
            ).train(epochs=1),
        )[-1],
        "save": lambda model, path: torch.save(
            model, f"{path}/csa_curr.pt"),
        "load": lambda path: torch.load(f"{path}/csa_curr.pt"),
    }
    CSA = {
        "train": lambda dataset: (
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
        ),
        "test": lambda model, dataset: (
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListOneHotTransform,
                training_share=0,
            ).train(epochs=1),
        )[-1],
        "save": lambda model, path: torch.save(
            model, f"{path}/csa.pt"),
        "load": lambda path: torch.load(f"{path}/csa.pt"),
    }
    DY_SAT = {
        "train": lambda dataset: (
            model := DySAT(
                args=get_model_args(),
                num_features=14,
                time_length=14,
                num_classes=len(dataset.labels),
            ),
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListOneHotTransform,
            ).train(epochs=30),
        ),
        "test": lambda model, dataset: (
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListOneHotTransform,
                training_share=0,
            ).train(epochs=1),
        )[-1],
        "save": lambda model, path: torch.save(
            model, f"{path}/dy_sat.pt"),
        "load": lambda path: torch.load(f"{path}/dy_sat.pt"),
    }
    ROLAND = {
        "train": lambda dataset: (
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
        ),
        "test": lambda model, dataset: (
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListOneHotTransform,
                training_share=0,
            ).train(epochs=1),
        )[-1],
        "save": lambda model, path: torch.save(
            model, f"{path}/roland.pt"),
        "load": lambda path: torch.load(f"{path}/roland.pt"),
    }


def filter_pretext_models(results: pd.DataFrame, keep=False):
    pretext_models = ["CSA_CON", "CSA_PRED", "CSA_CURR"]
    # pretext_models = ["CSA_CURR"]
    # pretext_models = ["CSA_CON", "CSA_PRED"]

    def remove_them(x):
        return x not in pretext_models

    def remove_all_but_them(x):
        return x in ["CSA_BASE", *pretext_models]

    return results.loc[results["model"].apply(remove_them if not keep else remove_all_but_them)]


def analyze_result(results: pd.DataFrame, path: str):
    plt.rcParams.update({'font.size': 20})
    plt.rcParams['legend.fontsize'] = 18
    compute_means_with_significance(
        results=results, group_col="model", path=path)

    for dataset_name, dataset_df in results.groupby("dataset"):
        dataset_path = f"{path}/{dataset_name}"
        compute_means_with_significance(
            results=dataset_df, group_col="model", path=dataset_path)
        for method, method_df in dataset_df.groupby("method"):
            method_path = f"{dataset_path}/{method}"
            compute_means_with_significance(
                results=method_df, group_col="model", path=method_path)
            # plot linechart for each metric
            for metric, metric_df in method_df.groupby("metric"):
                type_count = len(metric_df.groupby("type")["type"].nunique())
                nrows, ncols = get_grid_size(type_count)
                fig, ax = plt.subplots(
                    nrows=nrows, ncols=ncols, figsize=(8 * nrows, 8 * ncols)
                )
                ax = ax.flat if nrows * ncols > 1 else [ax]
                for j, (type_name, type_df) in enumerate(metric_df.groupby("type")):
                    type_df["value"] = type_df["value"] * \
                        100 if metric != "runtime" else type_df["value"] / 1000
                    sns.lineplot(
                        data=type_df, x="amount", y="value", hue="model", ax=ax[j], errorbar=None, linewidth=2.5,
                        # method == "start" or method == "gaps" or method == "nodes"
                        legend=dataset_name == "COV-19"
                    )
                    legend = ax[j].legend_
                    if legend is not None:
                        legend.set_title('')
                    ax[j].set_axisbelow(True)
                    ax[j].set_title("")
                    ax[j].set_xlabel(
                        "portion of masked nodes" if method == "nodes" else "#masked snapshots")
                    ax[j].set_ylabel(f"{metric} (%)")
                    ax[j].grid(True)
                save_fig(fig, f"{method_path}/linechart_{metric}", svg=True)
    plt.rcParams.update(
        {'font.size': plt.rcParamsDefault['font.size']})


def analyze_results(results: pd.DataFrame, path: str):
    # we don't care about buckets rn
    results = results.loc[results["type"] == "test"]

    analyze_result(filter_pretext_models(results=results), path)
    # analyze_result(filter_pretext_models(
    #    results=results.loc[results["type"] == "test"], keep=True), f"{path}/pretext")


def compute_stat(results: pd.DataFrame, path: str):
    model_regression_results = compute_bay_regression_per_group(
        results=results, group_key="method", entity_key="model", predictor_column="amount", path=path)
    model_significance_results = compute_t_test_significance_from_bay_regression_results(
        results=model_regression_results, group_key="method", entity_key="model", path=path)
    plot_t_test_significance_rate(
        results=model_significance_results, group_key="method", entity_key="model", path=path)


def compute_stats(results: pd.DataFrame, path: str):
    compute_stat(filter_pretext_models(results=results), path)
    compute_stat(filter_pretext_models(
        results=results, keep=True), f"{path}/pretext")


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


def get_dataset_names():
    return [dataset.name for dataset in get_datasets()]


dataset_helper = {
    "Synthetic": SyntheticDataset,
    "COVID-19": Covid19Dataset,
    "FakeNews": FakeNewsDataset,
    "BTC-BlockPropagation": BitcoinBlockPropagationDataset,
    "DDoS": DDoSDataset,
    "Waves": WavesDataset,
}


def get_models():
    return [
        # ClassificationModel.METRICS_LOG,
        # ClassificationModel.METRICS_MLP,
        # ClassificationModel.METRICS_XGB,
        # ClassificationModel.METRICS_GAT,
        # ClassificationModel.ATTR_GAT,
        # ClassificationModel.CSA_BASE,
        # ClassificationModel.CSA_EXT,
        # ClassificationModel.CSA_CON,
        # ClassificationModel.CSA_PRED,
        # ClassificationModel.CSA_CURR,
        ClassificationModel.CSA,
        ClassificationModel.DY_SAT,
        ClassificationModel.ROLAND,
    ]


def run_experiment(exp_args):
    (
        dataset_name,
        models,
        mask_method,
        amount,
        recreate,
        trained_model_path,
    ) = exp_args

    models = deserialize_enum_values(models, ClassificationModel)
    dataset = dataset_helper[dataset_name](
        log_progress=True, recreate=recreate, mask_method=mask_method, mask_amount=amount)

    results = []
    for model in models:
        print(f"Running {model.name}...")
        dataset_model_path = f"{trained_model_path}/{dataset_name}"
        trained_model = model.value["load"](dataset_model_path)
        result = model.value["test"](trained_model, dataset)
        results.append(
            {
                "dataset": dataset.abbreviation,
                "model": model.name,
                "result": result,
                "more_values": {
                    "method": mask_method,
                    "amount": amount,
                },
            }
        )
    return results


def run_experiments(recreate=False, analyze=True, execute=True, parallelize=False, stats=False):
    path = "experiments/masking"
    results = read_results(
        path, [*default_result_columns(), "method", "amount"])
    if execute:
        masks = [
            # {"mask_method": "spots"},
            # {"mask_method": "gaps"},
            {"mask_method": "start"},
            {"mask_method": "end"},
            # {"mask_method": "nodes"},
        ]
        models = get_models()
        serialized_models = serialize_enum_values(models)
        max_t = 14

        # train the models on the base dataset
        trained_model_path = f"{path}/models"
        base_datasets = get_datasets(recreate=recreate)
        for base_dataset in base_datasets:
            for model in models:
                print(f"Training {model.name}...")
                dataset_path = f"{trained_model_path}/{base_dataset.name}"
                Path(dataset_path).mkdir(parents=True, exist_ok=True)
                trained_model, train_result = model.value["train"](
                    base_dataset)
                model.value["save"](trained_model, dataset_path)
                for mask in masks:
                    append_to_results(results=results, model=model.name, dataset=base_dataset.abbreviation, result=train_result, more_values={
                        "method": mask["mask_method"],
                        "amount": 0,
                    })
        save_results(results, path)

        combinations = [
            (
                dataset_name,
                serialized_models,
                mask["mask_method"],
                amount,
                recreate,
                trained_model_path
            )
            for mask in masks
            for amount in (range(1, (max_t // 2) + 1) if mask["mask_method"] != "nodes" else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
            for dataset_name in dataset_helper.keys()
        ]

        parallelize_experiment_runs(
            results=results,
            combinations=combinations,
            run_experiment=run_experiment,
            parallelize=parallelize,
            path=path)

        save_results(results, path)
    if analyze:
        results['model'] = results['model'].replace("ATTR_GAT", "GAT")
        results = results[results['model'].isin(
            ["CSA", "GAT", "DY_SAT", "ROLAND"])]
        analyze_results(results, path)
    if stats:
        compute_stats(results, path)


if __name__ == "__main__":
    default_runner(run_experiments)
