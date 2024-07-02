from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

import src.models.metrics.metrics as metrics_baselines
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
from src.datasets.transform import (PropagationMetricsGraphTransform,
                                    TemporalGraphAttributesTransform,
                                    TemporalSnapshotListTransform)
from src.generation.propagation.propagation import PropagationMetrics
from src.models.base_gnn import BaseGNN
from src.models.cross_snapshot_attention import CrossSnapshotAttentionNet
from src.pretext.contrastive.trainer import load_from_contrastive_pretext_model
from src.pretext.predictive.trainer import load_from_predictive_pretext_model
from src.training.curriculum import (complexity_density_curriculum,
                                     train_with_curriculum)
from src.training.trainer import GNNTrainer
from src.utils.correlation import bonferroni_correction
from src.utils.drawing import get_grid_size, save_fig


class ClassificationModel(Enum):
    METRICS_LOG = {
        "run": lambda dataset: metrics_baselines.execute(
            dataset=dataset,
            metrics_flavor="propagation",
            method="linear_regression",
        ),
    }
    METRICS_MLP = {
        "run": lambda dataset: metrics_baselines.execute(
            dataset=dataset,
            metrics_flavor="propagation",
            method="mlp",
        ),
    }
    METRICS_XGB = {
        "run": lambda dataset: metrics_baselines.execute(
            dataset=dataset,
            metrics_flavor="propagation",
            method="xgboost",
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
            ).train(epochs=30),
        )[-1]
    }
    CSA_CON = {
        "run": lambda dataset: (
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
        )[-1],
    }
    CSA_PRED = {
        "run": lambda dataset: (
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
        )[-1],
    }
    CSA_CURR = {
        "run": lambda dataset: (
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
        )[-1],
    }


def filter_pretext_models(results: pd.DataFrame, keep=False):
    pretext_models = ["CSA_CON", "CSA_PRED", "CSA_CURR"]

    def remove_them(x):
        return x not in pretext_models

    def remove_all_but_them(x):
        return x in ["CSA_BASE", *pretext_models]

    return results.loc[results["model"].apply(remove_them if not keep else remove_all_but_them)]


def analyze_result(results: pd.DataFrame, path: str):
    compute_means_with_significance(
        results=results, group_col="source", path=path)
    compute_means_with_significance(
        results=results, group_col="target", path=path)
    compute_means_with_significance(
        results=results, group_col="model", path=path)
    same_source_target = results[results["source"] == results["target"]]
    compute_means_with_significance(
        results=same_source_target, group_col="target", path=path, file_prefix="same_source_")
    for dataset, dataset_df in results.groupby("dataset"):
        dataset_path = f"{path}/{dataset}"
        compute_means_with_significance(
            results=dataset_df, group_col="source", path=dataset_path)
        compute_means_with_significance(
            results=dataset_df, group_col="target", path=dataset_path)
        compute_means_with_significance(
            results=dataset_df, group_col="model", path=dataset_path)
        same_source_target = dataset_df[dataset_df["source"]
                                        == dataset_df["target"]]
        compute_means_with_significance(
            results=same_source_target, group_col="target", path=dataset_path, file_prefix="same_source_")
        for type_name, type_df in dataset_df.groupby("type"):
            for metric, metric_df in type_df.groupby("metric"):
                metric_path = f"{dataset_path}/{type_name}/{metric}"

                # plot boxplots for each source
                source_count = len(metric_df.groupby(
                    "source")["source"].nunique())
                nrows, ncols = get_grid_size(source_count)
                fig, ax = plt.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    figsize=(12 * ncols, 8 * nrows),
                )
                ax = ax.flatten() if nrows * ncols > 1 else [ax]
                for j, (source, source_df) in enumerate(metric_df.groupby("source")):
                    sns.boxplot(
                        ax=ax[j], data=source_df, x="value", y="target", hue="model"
                    )
                    ax[j].set_title(f"source: {source}")
                save_fig(fig, f"{metric_path}_boxplot")


def analyze_results(results: pd.DataFrame, path: str):
    results.loc[:, "value"] = results["value"].round(4)
    analyze_result(filter_pretext_models(results=results), path)
    analyze_result(filter_pretext_models(
        results=results, keep=True), f"{path}/pretext")


def compute_model_stats(results: pd.DataFrame, path: str):
    # Perform Mann-Whitney U test to test if the performance gets significantly worse (or better) when transferring to a different target graph type
    stats = pd.DataFrame(
        [], columns=["model", "dataset", "change", "inc_suc", "dec_suc"])

    for model_name, model_df in results.groupby("model"):
        for source, source_df in model_df.groupby("source"):
            datasets = source_df['dataset'].unique()
            targets = source_df['target'].unique()

            for dataset in datasets:
                dataset_group = source_df[source_df['dataset'] == dataset]
                source_results = dataset_group[dataset_group['target']
                                               == source]['value']
                for target, source_target_df in dataset_group.groupby("target"):
                    if target == source:
                        continue
                    target_results = source_target_df['value']

                    if len(source_results) == 0 or len(target_results) == 0:
                        print(f"Missing data for {dataset} {source} {target}")
                        continue

                    _, increase_p_value = mannwhitneyu(
                        target_results, source_results, alternative='greater')
                    _, decrease_p_value = mannwhitneyu(
                        target_results, source_results, alternative='less')

                    change = target_results.mean() - source_results.mean()
                    corrected_alpha = bonferroni_correction(
                        0.05, len(targets) - 1)
                    inc_suc = increase_p_value < corrected_alpha
                    dec_suc = decrease_p_value < corrected_alpha
                    stats.loc[len(stats)] = {
                        "model": model_name,
                        "dataset": dataset,
                        "change": change,
                        "inc_suc": inc_suc,
                        "dec_suc": dec_suc,
                    }

    stats.to_csv(f"{path}/model_significance_stats.csv", index=False)

    # calculate total number of tests performed
    sign_model_stats = (stats.groupby(['model'])
                        .size()
                        .reset_index(name='total'))

    # calculate the rates for all models
    sign_model_stats['inc_rate'] = (stats[stats['inc_suc'] == True].groupby(['model'])
                                    .size()
                                    .reset_index(name='inc_count')
                                    .inc_count) / sign_model_stats['total']

    sign_model_stats['dec_rate'] = (stats[stats['dec_suc'] == True].groupby(['model'])
                                    .size()
                                    .reset_index(name='dec_count')
                                    .dec_count) / sign_model_stats['total']

    # aggregate the stats to a new type column for better plotting
    model_inc_stats = sign_model_stats[[
        'model', 'total', 'inc_rate']].copy()
    model_inc_stats.columns = ['model', 'total', 'rate']
    model_inc_stats['type'] = 'inc'

    model_dec_stats = sign_model_stats[[
        'model', 'total', 'dec_rate']].copy()
    model_dec_stats.columns = ['model', 'total', 'rate']
    model_dec_stats['type'] = 'dec'

    sign_stats = pd.concat(
        [model_inc_stats, model_dec_stats], ignore_index=True)
    sign_stats = sign_stats.replace(np.nan, 0)

    # calculate the change in performance per model respectively
    change_model_stats = stats[['model', 'change']].copy()

    # only include the significant changes
    filtered_sign = stats[(stats['inc_suc'] == True) |
                          (stats['dec_suc'] == True)]
    sign_change_model_stats = filtered_sign[['model', 'change']].copy()

    # test for significant differences between changes of models
    final_change_stats = pd.DataFrame(
        [], columns=["model1", "model2", "p_value", "is_significant"])
    models = sign_model_stats['model'].unique()
    for model1, model1_df in sign_change_model_stats.groupby('model'):
        for model2, model2_df in sign_change_model_stats.groupby('model'):
            if model1 == model2:
                continue

            model1_values = model1_df["change"]
            model2_values = model2_df["change"]

            _, p_value = mannwhitneyu(
                model1_values, model2_values)
            corrected_alpha = bonferroni_correction(
                0.05, len(models) - 1)
            is_significant = p_value < corrected_alpha
            final_change_stats.loc[len(final_change_stats)] = {
                "model1": model1,
                "model2": model2,
                "p_value": p_value,
                "is_significant": is_significant
            }
    final_change_stats.to_csv(
        f"{path}/model_performance_drop_significance.csv", index=False)

    # report the number of non-significant changes to avoid publication bias
    non_sign_model_stats = (stats.groupby(['model'])
                            .size()
                            .reset_index(name='total'))
    filtered_non_sign = stats[(stats['inc_suc'] == False)
                              & (stats['dec_suc'] == False)]

    non_sign_model_stats["rate"] = (filtered_non_sign.groupby(['model'])
                                    .size()
                                    .reset_index(name='total').total) / non_sign_model_stats["total"]

    non_sign_model_stats = non_sign_model_stats[['model', 'rate']].copy()

    # plot barchart for all results
    nrows, ncols = get_grid_size(4)
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10 * ncols, 10 * nrows),
    )
    ax = ax.flatten() if nrows * ncols > 1 else [ax]

    sns.barplot(ax=ax[0], data=sign_stats, x="model", y="rate", hue="type")
    ax[0].set_title("Significance Rate")
    sns.boxplot(ax=ax[1], data=change_model_stats, x="model", y="change")
    ax[1].set_title("Accuracy Change")
    sns.boxplot(ax=ax[2], data=sign_change_model_stats, x="model", y="change")
    ax[2].set_title("Significant Accuracy Change")
    sns.barplot(ax=ax[3], data=non_sign_model_stats, x="model", y="rate")
    ax[3].set_title("Non-Significant Rate")
    save_fig(fig, f"{path}/model_significance_barchart")

    plt.rcParams.update({'font.size': 14})
    plt.rcParams['legend.fontsize'] = 12
    # add separate figures for thesis
    model_order = [model.name for model in get_models() if model.name not in [
        "CSA_CON", "CSA_PRED", "CSA_CURR"]]
    sign_stats = sign_stats[sign_stats['type'] == 'dec']
    sign_stats["rate"] = sign_stats["rate"] * 100
    ax = sns.barplot(data=sign_stats, x="model", y="rate", order=model_order)
    ax.set_axisbelow(True)
    legend = ax.legend_
    if legend is not None:
        legend.set_title('')
    plt.xticks([])
    plt.xlabel("model")
    plt.ylabel("sign. rate of accuracy drop")
    plt.grid(True)
    save_fig(plt, f"{path}/model_significance_sign_rate", svg=True)

    change_model_stats["change"] = change_model_stats["change"] * 100
    ax = sns.boxplot(data=change_model_stats,
                     x="model", y="change", order=model_order)
    ax.set_axisbelow(True)
    legend = ax.legend_
    if legend is not None:
        legend.set_title('')
    handles = [plt.Rectangle((0, 0), 1, 1, color=color)
               for color in sns.color_palette()]
    plt.legend(handles, model_order)
    plt.grid(True)
    plt.xticks([])
    plt.xlabel("model")
    plt.ylabel("accuracy change (%)")

    save_fig(plt, f"{path}/model_significance_accuracy_drop", svg=True)
    plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})


def compute_graph_stats(results: pd.DataFrame, path: str):
    # Perform Mann-Whitney U test to test if the performance gets significantly worse (or better) when transferring to a different target graph type
    stats = pd.DataFrame(
        [], columns=["source", "target", "dataset", "change", "inc_suc", "dec_suc"])

    for source, source_df in results.groupby("source"):
        datasets = source_df['dataset'].unique()
        targets = source_df['target'].unique()

        for dataset in datasets:
            dataset_group = source_df[source_df['dataset'] == dataset]
            source_results = dataset_group[dataset_group['target']
                                           == source]['value']
            for target, source_target_df in dataset_group.groupby("target"):
                if target == source:
                    continue
                target_results = source_target_df['value']

                if len(source_results) == 0 or len(target_results) == 0:
                    print(f"Missing data for {dataset} {source} {target}")
                    continue

                _, increase_p_value = mannwhitneyu(
                    target_results, source_results, alternative='greater')
                _, decrease_p_value = mannwhitneyu(
                    target_results, source_results, alternative='less')

                change = target_results.mean() - source_results.mean()
                corrected_alpha = bonferroni_correction(
                    0.05, len(targets) - 1)
                inc_suc = increase_p_value < corrected_alpha
                dec_suc = decrease_p_value < corrected_alpha
                stats.loc[len(stats)] = {
                    "source": source,
                    "target": target,
                    "dataset": dataset,
                    "change": change,
                    "inc_suc": inc_suc,
                    "dec_suc": dec_suc,
                }

    stats.to_csv(f"{path}/graph_significance_stats.csv", index=False)

    # calculate total number of tests performed
    sign_source_stats = (stats.groupby(['source'])
                         .size()
                         .reset_index(name='total'))
    sign_target_stats = (stats.groupby(['target'])
                         .size()
                         .reset_index(name='total'))

    # calculate the rates for source and target
    sign_source_stats['inc_rate'] = (stats[stats['inc_suc'] == True].groupby(['source'])
                                     .size()
                                     .reset_index(name='inc_count')
                                     .inc_count) / sign_source_stats['total']

    sign_source_stats['dec_rate'] = (stats[stats['dec_suc'] == True].groupby(['source'])
                                     .size()
                                     .reset_index(name='dec_count')
                                     .dec_count) / sign_source_stats['total']
    sign_target_stats['inc_rate'] = (stats[stats['inc_suc'] == True].groupby(['target'])
                                     .size()
                                     .reset_index(name='inc_count')
                                     .inc_count) / sign_target_stats['total']

    sign_target_stats['dec_rate'] = (stats[stats['dec_suc'] == True].groupby(['target'])
                                     .size()
                                     .reset_index(name='dec_count')
                                     .dec_count) / sign_target_stats['total']

    # aggregate the stats to a new type column for better plotting
    source_inc_stats = sign_source_stats[[
        'source', 'total', 'inc_rate']].copy()
    source_inc_stats.columns = ['graph', 'total', 'rate']
    source_inc_stats['type'] = 'from_inc'

    source_dec_stats = sign_source_stats[[
        'source', 'total', 'dec_rate']].copy()
    source_dec_stats.columns = ['graph', 'total', 'rate']
    source_dec_stats['type'] = 'from_dec'

    target_inc_stats = sign_target_stats[[
        'target', 'total', 'inc_rate']].copy()
    target_inc_stats.columns = ['graph', 'total', 'rate']
    target_inc_stats['type'] = 'to_inc'

    target_dec_stats = sign_target_stats[[
        'target', 'total', 'dec_rate']].copy()
    target_dec_stats.columns = ['graph', 'total', 'rate']
    target_dec_stats['type'] = 'to_dec'

    sign_stats = pd.concat([source_inc_stats, source_dec_stats,
                            target_inc_stats, target_dec_stats], ignore_index=True)
    sign_stats = sign_stats.replace(np.nan, 0)

    # calculate the change in performance per target and source respectively
    change_source_stats = stats[['source', 'change']].copy()
    change_source_stats.columns = ['graph', 'change']
    change_source_stats['type'] = 'from'
    change_target_stats = stats[['target', 'change']].copy()
    change_target_stats.columns = ['graph', 'change']
    change_target_stats['type'] = 'to'

    change_stats = pd.concat(
        [change_source_stats, change_target_stats], ignore_index=True)

    # only include the significant changes
    filtered_sign = stats[(stats['inc_suc'] == True) |
                          (stats['dec_suc'] == True)]
    sign_change_source_stats = filtered_sign[['source', 'change']].copy()
    sign_change_source_stats.columns = ['graph', 'change']
    sign_change_source_stats['type'] = 'from'
    sign_change_target_stats = filtered_sign[['target', 'change']].copy()
    sign_change_target_stats.columns = ['graph', 'change']
    sign_change_target_stats['type'] = 'to'

    sign_change_stats = pd.concat(
        [sign_change_source_stats,  sign_change_target_stats], ignore_index=True)

    # test for significant differences between changes
    final_change_stats = pd.DataFrame(
        [], columns=["graph1", "graph2", "type", "p_value", "is_significant"])
    targets = sign_change_stats['graph'].unique()
    for graph1, graph1_df in sign_change_stats.groupby('graph'):
        for graph2, graph2_df in sign_change_stats.groupby('graph'):
            if graph1 != graph2:
                source1_values = graph1_df[graph1_df['type']
                                           == 'from']["change"]
                source2_values = graph2_df[graph2_df['type']
                                           == 'from']["change"]
                target1_values = graph1_df[graph1_df['type']
                                           == 'to']["change"]
                target2_values = graph2_df[graph2_df['type']
                                           == 'to']["change"]
                _, source_p_value = mannwhitneyu(
                    source1_values, source2_values)
                _, target_p_value = mannwhitneyu(
                    target1_values, target2_values)
                corrected_alpha = bonferroni_correction(
                    0.05, len(targets) - 1)
                source_is_significant = source_p_value < corrected_alpha
                target_is_significant = target_p_value < corrected_alpha
                final_change_stats.loc[len(final_change_stats)] = {
                    "graph1": graph1,
                    "graph2": graph2,
                    "type": "from",
                    "p_value": source_p_value,
                    "is_significant": source_is_significant
                }
                final_change_stats.loc[len(final_change_stats)] = {
                    "graph1": graph1,
                    "graph2": graph2,
                    "type": "to",
                    "p_value": target_p_value,
                    "is_significant": target_is_significant
                }
            else:
                source_values = graph1_df[graph1_df['type']
                                          == 'from']["change"]
                target_values = graph1_df[graph1_df['type']
                                          == 'to']["change"]
                _, p_value = mannwhitneyu(source_values, target_values)
                corrected_alpha = bonferroni_correction(
                    0.05, len(targets) - 1)
                is_significant = p_value < corrected_alpha
                final_change_stats.loc[len(final_change_stats)] = {
                    "graph1": graph1,
                    "graph2": graph2,
                    "type": "between",
                    "p_value": p_value,
                    "is_significant": is_significant
                }

    final_change_stats.to_csv(
        f"{path}/graph_performance_drop_significance.csv", index=False)

    # report the number of non-significant changes to avoid publication bias
    non_sign_source_stats = (stats.groupby(['source'])
                             .size()
                             .reset_index(name='total'))
    non_sign_target_stats = (stats.groupby(['target'])
                             .size()
                             .reset_index(name='total'))
    filtered_non_sign = stats[(stats['inc_suc'] == False)
                              & (stats['dec_suc'] == False)]

    non_sign_source_stats["rate"] = (filtered_non_sign.groupby(['source'])
                                     .size()
                                     .reset_index(name='total').total) / non_sign_source_stats["total"]
    non_sign_target_stats["rate"] = (filtered_non_sign.groupby(['target'])
                                     .size()
                                     .reset_index(name='total').total) / non_sign_target_stats["total"]

    non_sign_source_stats = non_sign_source_stats[['source', 'rate']].copy()
    non_sign_source_stats.columns = ['graph', 'rate']
    non_sign_source_stats['type'] = 'source'
    non_sign_target_stats = non_sign_target_stats[['target', 'rate']].copy()
    non_sign_target_stats.columns = ['graph', 'rate']
    non_sign_target_stats['type'] = 'target'

    non_sign_change_stats = pd.concat(
        [non_sign_source_stats, non_sign_target_stats], ignore_index=True)

    # plot barchart for all results
    nrows, ncols = get_grid_size(4)
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10 * ncols, 10 * nrows),
    )
    ax = ax.flatten() if nrows * ncols > 1 else [ax]

    sns.barplot(ax=ax[0], data=sign_stats, x="graph", y="rate", hue="type")
    ax[0].set_title("Significance Rate")
    sns.boxplot(ax=ax[1], data=change_stats, x="graph", y="change", hue="type")
    ax[1].set_title("Accuracy Change")
    sns.boxplot(ax=ax[2], data=sign_change_stats,
                x="graph", y="change", hue="type")
    ax[2].set_title("Significant Accuracy Change")
    sns.barplot(ax=ax[3], data=non_sign_change_stats,
                x="graph", y="rate", hue="type")
    ax[3].set_title("Non-Significant Rate")
    save_fig(fig, f"{path}/graph_significance_barchart")

    plt.rcParams.update({'font.size': 14})
    plt.rcParams['legend.fontsize'] = 12
    # add separate figures for thesis
    sign_stats = sign_stats[(sign_stats['type'] == 'from_dec') |
                            (sign_stats['type'] == 'to_dec')]
    sign_stats['type'] = sign_stats['type'].replace(
        {'from_dec': 'source', 'to_dec': 'target'})
    sign_stats["rate"] = sign_stats["rate"] * 100
    ax = sns.barplot(data=sign_stats, x="graph", y="rate",
                     hue="type")
    ax.set_axisbelow(True)
    legend = ax.legend_
    if legend is not None:
        legend.remove()
    plt.xlabel("graph type")
    plt.ylabel("sign. rate of accuracy drop")
    plt.grid(True)
    save_fig(plt, f"{path}/graph_significance_sign_rate", svg=True)

    change_stats['type'] = change_stats['type'].replace(
        {'from': 'source', 'to': 'target'})
    change_stats["change"] = change_stats["change"] * 100
    ax = sns.boxplot(data=change_stats, x="graph", y="change", hue="type")
    ax.set_axisbelow(True)
    legend = ax.legend_
    if legend is not None:
        legend.set_title('')
    plt.xlabel("graph type")
    plt.ylabel("accuracy change (%)")
    plt.grid(True)
    save_fig(plt, f"{path}/graph_significance_accuracy_drop", svg=True)
    plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})


def compute_stats(results: pd.DataFrame, path: str):
    # we do regression only on accuracy of the test bucket
    results = results[results['type'] == 'test']
    results = results[results['metric'] == 'accuracy']

    compute_graph_stats(filter_pretext_models(results=results), path)
    compute_model_stats(filter_pretext_models(results=results), path)
    compute_model_stats(filter_pretext_models(
        results=results, keep=True), f"{path}/pretext")


def get_dataset_classes():
    return [
        SyntheticDataset,
        Covid19Dataset,
        FakeNewsDataset,
        BitcoinBlockPropagationDataset,
        DDoSDataset,
        WavesDataset,
    ]


def create_dataset(dataset_class, source_graph: str, target_graph: str, recreate=False):
    dataset = dataset_class(
        log_progress=True,
        train_behavior_distribution={source_graph: 1},
        test_behavior_distribution={target_graph: 1},
        recreate=recreate,
    )
    return dataset


def get_models():
    return [
        ClassificationModel.METRICS_LOG,
        ClassificationModel.METRICS_MLP,
        ClassificationModel.METRICS_XGB,
        ClassificationModel.METRICS_GAT,
        ClassificationModel.ATTR_GAT,
        ClassificationModel.CSA_BASE,
        ClassificationModel.CSA_EXT,
        ClassificationModel.CSA_CON,
        ClassificationModel.CSA_PRED,
        ClassificationModel.CSA_CURR,
    ]


def run_experiment(exp_args):
    (
        dataset_class,
        source_graph_type,
        target_graph_type,
        models,
        recreate,
    ) = exp_args

    models = deserialize_enum_values(models, ClassificationModel)
    dataset = create_dataset(
        dataset_class, source_graph_type, target_graph_type, recreate=recreate
    )

    results = []
    for model in models:
        print(f"Running {model.name}...")
        result = model.value["run"](dataset)
        results.append(
            {
                "dataset": dataset.abbreviation.split("&")[0],
                "model": model.name,
                "result": result,
                "more_values": {
                    "source": source_graph_type,
                    "target": target_graph_type,
                },
            }
        )
    return results


def run_experiments(recreate=False, analyze=True, execute=True, stats=False, parallelize=False):
    path = "experiments/transportability"
    results = read_results(
        path, columns=[*default_result_columns(), "source", "target"]
    )
    if execute:
        dataset_classes = get_dataset_classes()
        models = get_models()
        graph_types = ["er", "ws", "ba", "exp"]

        models = serialize_enum_values(models)
        combinations = [
            (
                dataset_class,
                source_graph_type,
                target_graph_type,
                models,
                recreate,
            )
            for source_graph_type in graph_types
            for target_graph_type in graph_types
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
    if stats:
        compute_stats(results, path)


if __name__ == "__main__":
    default_runner(run_experiments)
