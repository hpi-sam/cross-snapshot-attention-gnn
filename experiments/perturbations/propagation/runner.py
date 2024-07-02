import ast
from enum import Enum
from itertools import combinations, product
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import xgboost as xgb
import numpy as np

import src.models.metrics.metrics as metrics_baselines
from experiments.utils import (
    append_to_results, compute_bay_regression_per_group,
    compute_means_with_significance,
    compute_t_test_significance_from_bay_regression_results,
    default_result_columns, default_runner, deserialize_enum_values,
    parallelize_experiment_runs, plot_t_test_significance_rate, read_results,
    save_results, serialize_enum_values)
from src.datasets.random.synthetic import SyntheticDataset
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
from src.utils.drawing import get_grid_size, save_fig, surfaceplot
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


def analyze_1d_results(results: pd.DataFrame, path: str):
    plt.rcdefaults()
    for dataset_name, dataset_df in results.groupby("dataset"):
        dataset_path = f"{path}/{dataset_name}"
        compute_means_with_significance(
            results=dataset_df, group_col="model", path=dataset_path, file_prefix="1d")
        compute_means_with_significance(
            results=dataset_df, group_col="graph", path=dataset_path,  file_prefix="1d")
        for perturbation_name, perturbation_df in dataset_df.groupby("perturbation"):
            perturbation_path = f"{dataset_path}/{perturbation_name}"
            compute_means_with_significance(
                results=perturbation_df, group_col="model", path=perturbation_path,  file_prefix="1d")
            compute_means_with_significance(
                results=perturbation_df, group_col="graph", path=perturbation_path,  file_prefix="1d")
            for type_name, type_df in perturbation_df.groupby("type"):
                for metric, metric_df in type_df.groupby("metric"):
                    type_path = f"{perturbation_path}/{type_name}"
                    # plot boxplots for each graph type
                    graph_type_count = len(
                        metric_df.groupby("graph")["graph"].nunique()
                    )
                    nrows, ncols = get_grid_size(graph_type_count)
                    fig, ax = plt.subplots(
                        nrows=nrows,
                        ncols=ncols,
                        figsize=(12 * ncols, 8 * nrows),
                    )
                    ax = ax.flatten() if nrows * ncols > 1 else [ax]
                    for j, (graph_type, graph_type_df) in enumerate(
                        metric_df.groupby("graph")
                    ):
                        graph_type_df["factor"] = graph_type_df["factor"].apply(
                            lambda x: ast.literal_eval(
                                x) if isinstance(x, str) else x
                        )
                        graph_type_df["factor"] = graph_type_df["factor"].apply(
                            lambda x: x[0] if isinstance(x, list) else x
                        )
                        sns.boxplot(
                            ax=ax[j],
                            data=graph_type_df,
                            x="factor",
                            y="value",
                            hue="model",
                        )
                        ax[j].set_title(
                            f"{perturbation_name} {metric} (graph_type: {graph_type})"
                        )

                    save_fig(fig, f"{type_path}/{metric}_boxplot")

                    # add another combined chart
                    plt.rcParams.update({'font.size': 14})
                    plt.rcParams['legend.fontsize'] = 12
                    metric_df["value"] = metric_df["value"] * 100
                    metric_df["factor"] = metric_df["factor"].apply(
                        lambda x: ast.literal_eval(
                            x) if isinstance(x, str) else x
                    )
                    metric_df["factor"] = metric_df["factor"].apply(
                        lambda x: x[0] if isinstance(x, list) else x
                    )
                    int_df = metric_df[metric_df['factor'].apply(
                        lambda x: x == np.floor(x))]
                    ax = sns.boxplot(data=int_df, x="factor",
                                     y="value", hue="model")
                    legend = ax.legend_
                    if legend is not None:
                        if perturbation_name == "latency":
                            legend.remove()
                        legend.set_title('')
                    ax.set_axisbelow(True)
                    plt.xlabel("#std. dev. from trained class definitions")
                    plt.ylabel(f"{metric} (%)")
                    plt.grid(True)
                    save_fig(
                        plt, f"{perturbation_path}/boxplot_{metric}_combined", svg=True)

                    graph = "mixed"
                    graph_type_df = metric_df[metric_df["graph"] == graph]
                    # add plot for specific graph type
                    plt.rcParams.update({'font.size': 14})
                    plt.rcParams['legend.fontsize'] = 12
                    int_df = graph_type_df[graph_type_df['factor'].apply(
                        lambda x: x == np.floor(x))]
                    ax = sns.boxplot(data=int_df, x="factor",
                                     y="value", hue="model")
                    legend = ax.legend_
                    if legend is not None:
                        legend.set_title('')
                    ax.set_axisbelow(True)
                    plt.xlabel(
                        "#std. dev. from trained class definitions")
                    plt.ylabel(f"{metric} (%)")
                    plt.grid(True)
                    save_fig(
                        plt, f"{perturbation_path}/boxplot_{metric}_{graph}", svg=True)


def analyze_2d_results(results: pd.DataFrame, path: str):
    for dataset_name, dataset_df in results.groupby("dataset"):
        dataset_path = f"{path}/{dataset_name}"
        compute_means_with_significance(
            results=dataset_df, group_col="model", path=dataset_path, file_prefix="2d")
        compute_means_with_significance(
            results=dataset_df, group_col="graph", path=dataset_path,  file_prefix="2d")
        for perturbation_name, perturbation_df in dataset_df.groupby("perturbation"):
            perturbation_path = f"{dataset_path}/{perturbation_name}"
            compute_means_with_significance(
                results=perturbation_df, group_col="model", path=perturbation_path,  file_prefix="2d")
            compute_means_with_significance(
                results=perturbation_df, group_col="graph", path=perturbation_path,  file_prefix="2d")
            for type_name, type_df in perturbation_df.groupby("type"):
                for metric, metric_df in type_df.groupby("metric"):
                    type_path = f"{perturbation_path}/{type_name}"
                    # plot scatter plots for each graph type
                    graph_type_count = len(
                        metric_df.groupby("graph")["graph"].nunique()
                    )
                    nrows, ncols = get_grid_size(graph_type_count)
                    fig, ax = plt.subplots(
                        nrows=nrows,
                        ncols=ncols,
                        figsize=(12 * ncols, 8 * nrows),
                        subplot_kw={'projection': '3d'}
                    )
                    ax = ax.flatten() if nrows * ncols > 1 else [ax]
                    for j, (graph_type, graph_type_df) in enumerate(
                        metric_df.groupby("graph")
                    ):
                        # Convert the strings in the "factor" column to actual lists (arrays)
                        graph_type_df["factor"] = graph_type_df["factor"].apply(
                            lambda x: ast.literal_eval(
                                x) if isinstance(x, str) else x
                        )
                        graph_type_df["x_values"] = graph_type_df["factor"].apply(
                            lambda x: x[0]
                        )
                        graph_type_df["y_values"] = graph_type_df["factor"].apply(
                            lambda x: x[1]
                        )
                        # Calculate the mean of the "value" column for each unique combination
                        mean_df = (
                            graph_type_df.groupby(["x_values", "y_values"])
                            .agg({"value": "mean"})
                            .reset_index()
                        )

                        surfaceplot(data=mean_df, x="x_values",
                                    y="y_values", z="value", ax=ax[j])
                        ax[j].set_title(
                            f"{perturbation_name} {metric} (graph_type: {graph_type})"
                        )
                        ax[j].set_xlabel(perturbation_name.split("+")[0])
                        ax[j].set_ylabel(perturbation_name.split("+")[1])
                        ax[j].set_zlabel("value")

                    save_fig(fig, f"{type_path}/{metric}_scatter")


def analyze_3d_results(results: pd.DataFrame, path: str):
    for dataset_name, dataset_df in results.groupby("dataset"):
        dataset_path = f"{path}/{dataset_name}"
        compute_means_with_significance(
            results=dataset_df, group_col="model", path=dataset_path, file_prefix="3d")
        compute_means_with_significance(
            results=dataset_df, group_col="graph", path=dataset_path,  file_prefix="3d")
        for perturbation_name, perturbation_df in dataset_df.groupby("perturbation"):
            perturbation_path = f"{dataset_path}/{perturbation_name}"
            compute_means_with_significance(
                results=perturbation_df, group_col="model", path=perturbation_path,  file_prefix="3d")
            compute_means_with_significance(
                results=perturbation_df, group_col="graph", path=perturbation_path,  file_prefix="3d")
            for type_name, type_df in perturbation_df.groupby("type"):
                for metric, metric_df in type_df.groupby("metric"):
                    type_path = f"{perturbation_path}/{type_name}"

                    # plot scatter plots for each graph type
                    graph_type_count = len(
                        metric_df.groupby("graph")["graph"].nunique()
                    )
                    nrows, ncols = get_grid_size(graph_type_count)
                    fig, ax = plt.subplots(
                        nrows=nrows,
                        ncols=ncols,
                        figsize=(12 * ncols, 8 * nrows),
                        subplot_kw={'projection': '3d'}
                    )
                    ax = ax.flatten() if nrows * ncols > 1 else [ax]
                    for j, (graph_type, graph_type_df) in enumerate(
                        metric_df.groupby("graph")
                    ):
                        colormap = plt.cm.get_cmap("viridis")

                        # Create a 3D subplot for the current graph_type
                        # Convert the strings in the "factor" column to actual lists (arrays)
                        graph_type_df["factor"] = graph_type_df["factor"].apply(
                            lambda x: ast.literal_eval(
                                x) if isinstance(x, str) else x
                        )
                        graph_type_df["x_values"] = graph_type_df["factor"].apply(
                            lambda x: x[0]
                        )
                        graph_type_df["y_values"] = graph_type_df["factor"].apply(
                            lambda x: x[1]
                        )
                        graph_type_df["z_values"] = graph_type_df["factor"].apply(
                            lambda x: x[2]
                        )
                        # Calculate the mean of the "value" column for each unique combination
                        mean_df = (
                            graph_type_df.groupby(
                                ["x_values", "y_values", "z_values"])
                            .agg({"value": "mean"})
                            .reset_index()
                        )
                        # create colors for the values to add the 4th dimension
                        colors = colormap(mean_df["value"])

                        surfaceplot(data=mean_df, x="x_values",
                                    y="y_values", z="z_values", custom_cmap=colors, ax=ax[j])
                        ax[j].set_title(
                            f"{perturbation_name} {metric} (graph_type: {graph_type})"
                        )
                        ax[j].set_xlabel(perturbation_name.split("+")[0])
                        ax[j].set_ylabel(perturbation_name.split("+")[1])
                        ax[j].set_zlabel(perturbation_name.split("+")[2])

                        # Add a colorbar to show the scale of the fourth dimension
                        sm = plt.cm.ScalarMappable(
                            cmap=colormap, norm=plt.Normalize(0, 1)
                        )
                        sm.set_array([])
                        fig.colorbar(sm, ax=ax[j], label="Value", pad=0.1)
                    save_fig(fig, f"{type_path}/{metric}_scatter")


def compute_base_stats(results: pd.DataFrame, path: str):
    graph_regression_results = compute_bay_regression_per_group(
        results=results, group_key="perturbation", entity_key="graph", predictor_column="factor", path=path)
    model_regression_results = compute_bay_regression_per_group(
        results=results, group_key="perturbation", entity_key="model", predictor_column="factor", path=path)

    graph_significance_results = compute_t_test_significance_from_bay_regression_results(
        results=graph_regression_results, group_key="perturbation", entity_key="graph", path=path)
    model_significance_results = compute_t_test_significance_from_bay_regression_results(
        results=model_regression_results, group_key="perturbation", entity_key="model", path=path)

    plot_t_test_significance_rate(
        results=graph_significance_results, group_key="perturbation", entity_key="graph", path=path)
    plot_t_test_significance_rate(
        results=model_significance_results, group_key="perturbation", entity_key="model", path=path)


def compute_pretext_stats(results: pd.DataFrame, path: str):
    model_regression_results = compute_bay_regression_per_group(
        results=results, group_key="perturbation", entity_key="model", predictor_column="factor", path=path)
    model_significance_results = compute_t_test_significance_from_bay_regression_results(
        results=model_regression_results, group_key="perturbation", entity_key="model", path=path)
    plot_t_test_significance_rate(
        results=model_significance_results, group_key="perturbation", entity_key="model", path=path)


def compute_stats(results: pd.DataFrame, path: str):
    compute_base_stats(filter_pretext_models(results=results), path)
    compute_pretext_stats(filter_pretext_models(
        results=results, keep=True), f"{path}/pretext")


def analyze_base_results(results: pd.DataFrame, path: str):
    results = results.loc[results["type"] == "test"]
    compute_means_with_significance(
        results=results, group_col="model", path=path)
    compute_means_with_significance(
        results=results, group_col="graph", path=path)
    analyze_1d_results(
        results[results["perturbation"].apply(
            lambda x: x.count("+") == 0)], path
    )
    analyze_2d_results(
        results[results["perturbation"].apply(
            lambda x: x.count("+") == 1)], path
    )
    analyze_3d_results(
        results[results["perturbation"].apply(
            lambda x: x.count("+") == 2)], path
    )


def analyze_pretext_results(results: pd.DataFrame, path: str):
    # test is enough for pretext analysis
    results = results.loc[results["type"] == "test"]
    compute_means_with_significance(
        results=results, group_col="model", path=path)
    analyze_1d_results(
        results[results["perturbation"].apply(
            lambda x: x.count("+") == 0)], path
    )


def analyze_results(results: pd.DataFrame, path: str):
    # train is not interesting in this experiment
    results = results.loc[results["type"] == "test"]

    analyze_base_results(filter_pretext_models(results=results), path)
    # analyze_pretext_results(filter_pretext_models(
    #     results=results, keep=True), f"{path}/pretext")


def create_dataset(graph_type: str, factors: dict, recreate=False):
    name = f"syn_{factors}"
    factors = {f"def_{k}": v for k, v in factors.items()}
    dataset = SyntheticDataset(
        log_progress=True,
        # train_behavior_distribution={graph_type: 1},
        # test_behavior_distribution={graph_type: 1},
        name=name,
        abbreviation=name,
        recreate=recreate,
        **factors,
    )
    return dataset


def get_train_dataset(graph_type: str, recreate=False):
    return SyntheticDataset(
        log_progress=True,
        # train_behavior_distribution={graph_type: 1},
        # test_behavior_distribution={graph_type: 1},
        recreate=recreate,
    )


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
        graph_type,
        dataset_name,
        perturbation,
        factors,
        recreate,
        models,
        trained_model_path
    ) = exp_args

    dataset = create_dataset(
        graph_type=graph_type,
        factors=factors,
        recreate=recreate,
    )
    models = deserialize_enum_values(models, ClassificationModel)

    results = []
    for model in models:
        print(f"Running {model.name}...")
        graph_path = f"{trained_model_path}/{graph_type}"
        trained_model = model.value["load"](graph_path)
        result = model.value["test"](trained_model, dataset)
        results.append(
            {
                "dataset": dataset_name,
                "model": model.name,
                "result": result,
                "more_values": {
                    "perturbation": perturbation,
                    "graph": graph_type,
                    "factor": list(factors.values()),
                },
            }
        )
    return results


def run_experiments(recreate=False, analyze=True, execute=True, stats=False, parallelize=False):
    path = "experiments/perturbations/propagation"
    results = read_results(
        path, columns=[*default_result_columns(), "perturbation",
                       "graph", "factor"]
    )
    datasets = ["SYN"]
    if execute:
        models = get_models()
        serialized_models = serialize_enum_values(models)
        # perturbations = ["rate", "latency", "duration"]
        perturbations = ["rate", "latency"]
        # factors_1d = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
        factors_1d = [0.5, 1, 1.5, 2, 2.5, 3]
        factors_2d = [1, 2, 3, 4]
        # graph_types = ["er", "ba", "ws", "exp"]
        graph_types = ["mixed"]

        # train the models on the base dataset
        trained_model_path = f"{path}/models"
        for graph_type in graph_types:
            train_dataset = get_train_dataset(
                graph_type=graph_type, recreate=recreate)
            for model in models:
                print(f"Training {model.name}...")
                graph_path = f"{trained_model_path}/{graph_type}"
                Path(graph_path).mkdir(parents=True, exist_ok=True)
                trained_model, train_result = model.value["train"](
                    train_dataset)
                model.value["save"](trained_model, graph_path)
                for perturbation in perturbations:
                    append_to_results(results=results, model=model.name, dataset=datasets[0], result=train_result, more_values={
                        "perturbation": perturbation.lower(),
                        "graph": graph_type,
                        "factor": 0,
                    })
        save_results(results, path)

        combinations_exp = [
            (
                graph_type,
                dataset_name,
                "+".join(list(perturbation_comb)),
                {
                    f"{perturbation}_factor": factor
                    for perturbation, factor in zip(
                        perturbation_comb, factor_comb
                    )
                },
                recreate,
                serialized_models,
                trained_model_path
            )
            for perturbation_section, perturbations_list in [
                ("1D perturbations", [(p,) for p in perturbations]),
                # ("2D perturbations", list(combinations(perturbations, 2))),
                # ("3D perturbations", list(combinations(perturbations, 3))),
            ]
            for perturbation_comb in perturbations_list
            for graph_type in graph_types
            for factor_comb in list(
                product(
                    *[
                        factors_1d
                        if len(perturbation_comb) == 1
                        else factors_2d
                        for _ in perturbation_comb
                    ]
                )
            )
            for dataset_name in datasets
        ]

        parallelize_experiment_runs(
            results=results,
            combinations=combinations_exp,
            run_experiment=run_experiment,
            parallelize=parallelize,
            path=path)
        save_results(results, path)
    if analyze:
        results = results[results['model'].isin(
            ["CSA", "GAT", "DY_SAT", "ROLAND", ])]
        analyze_results(results, path)
    if stats:
        compute_stats(results, path)


if __name__ == "__main__":
    default_runner(run_experiments)
