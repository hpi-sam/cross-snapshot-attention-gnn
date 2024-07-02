import math
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
import seaborn as sns
from xgboost import plot_importance

from src.utils.path import remove_after_last_slash


def draw_heatmap(
    data,
    path,
    title="",
    xlabels=None,
    ylabels=None,
    vmin=0,
    vmax=1,
    annot=False,
    ax=None,
    rotate_x=None,
    rotate_y=None,
    figsize=None,
    triangle=False,
):
    sns.set_theme()

    xlabels = xlabels if xlabels is not None else []
    ylabels = ylabels if ylabels is not None else []

    def same_sign():
        min_sign = np.sign(vmin)
        max_sign = np.sign(vmax)
        return min_sign == max_sign or min_sign == 0 or max_sign == 0

    matrix = None
    if triangle:
        matrix = np.triu(data)

    if figsize:
        plt.figure(figsize=figsize)

    cmap = "Greens" if same_sign() else "BrBG"
    g = sns.heatmap(
        data,
        xticklabels=ylabels,
        yticklabels=xlabels,
        annot=annot,
        annot_kws={"fontsize": 8},
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        mask=matrix,
        ax=ax,
    )
    g.set(title=title)
    if rotate_x is not None:
        g.tick_params(axis="x", rotation=rotate_x)
    else:
        g.tick_params(axis="x", rotation=90)
    if rotate_y is not None:
        g.tick_params(axis="y", rotation=rotate_y)
    g.figure.tight_layout()
    if ax is None:
        Path(remove_after_last_slash(path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{path}.png")
        plt.style.use("default")
        if figsize:
            # Reset the figure size to the default size of (6.4, 4.8) inches
            plt.rcParams["figure.figsize"] = (6.4, 4.8)
        plt.close()


def draw_multiple_heatmaps(
    data,
    path,
    titles=None,
    xlabels=None,
    ylabels=None,
    vmin=0,
    vmax=1,
    annot=False,
    **kwargs,
):
    titles = titles if titles is not None else []
    xlabels = xlabels if xlabels is not None else []
    ylabels = ylabels if ylabels is not None else []

    nrows, ncols = get_grid_size(len(data))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(8 * nrows, 8 * ncols))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]
    for i, sample in enumerate(data):
        draw_heatmap(
            sample,
            "",
            titles[i],
            xlabels,
            ylabels if i < ncols else [],
            vmin,
            vmax,
            annot,
            axes[i],
            **kwargs,
        )
    fig.tight_layout()
    Path(remove_after_last_slash(path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{path}.png")
    plt.close()


def draw_importances(
    model,
    path,
    title="",
):
    plot_importance(model)
    plt.tight_layout()
    Path(remove_after_last_slash(path)).mkdir(parents=True, exist_ok=True)
    plt.title(title)
    plt.savefig(f"{path}.png")
    plt.close()


def to_nx_color(rgba_color):
    return [c / 255.0 for c in rgba_color]


def draw_propagation_graph(propagation, title="", path=None, pos=None, labels=True, svg=False, timestamps=None, pretty=False):
    behavior_graphs = propagation.behavior_graph.snapshots
    propagation_graphs = propagation.snapshots

    covered_new = to_nx_color([238, 75, 43, 255])
    covered = to_nx_color([238, 75, 43, 128])
    uncovered = to_nx_color([105, 105, 105, 128])

    if pos is None:
        pos = nx.spring_layout(behavior_graphs[-1], iterations=100, seed=39775)

    num_snapshots = len(
        behavior_graphs) if timestamps is None else len(timestamps)
    max_col = 5 if timestamps is None else len(timestamps)
    cols = min(num_snapshots, max_col)
    rows = math.ceil(num_snapshots / max_col)
    fig, all_axes = plt.subplots(rows, cols, figsize=(
        cols * 3, 8) if not pretty else (cols * 4, 4))
    ax = all_axes.flat

    def includes_edge(edge: Tuple[int, int], edges: List[Tuple[int, int]]):
        return (edge[0], edge[1]) in edges or (
            edge[1],
            edge[0],
        ) in edges

    for i, (behavior_graph, propagation_graph) in enumerate(
        zip(behavior_graphs, propagation_graphs)
    ):
        if timestamps is not None and i not in timestamps:
            continue

        ax_ind = i if timestamps is None else timestamps.index(i)

        ax[ax_ind].set_title(
            f"t = {i}", fontsize="small" if not pretty else "xx-large")
        covered_nodes = list(propagation_graph.nodes())
        covered_edges = list(propagation_graph.edges())
        new_covered_nodes = covered_nodes
        new_covered_edges = []
        if i > 0:
            new_covered_nodes, new_covered_edges = propagation.diff(i - 1, i)
        node_colors = [
            covered_new
            if x in new_covered_nodes
            else (covered if x in covered_nodes else uncovered)
            for x in behavior_graph.nodes()
        ]
        node_sizes = [
            350 if x in new_covered_nodes else 250 for x in behavior_graph.nodes()
        ] if not pretty else [500 for x in behavior_graph.nodes()]
        edge_colors = [
            covered_new
            if includes_edge(x, new_covered_edges)
            else (covered if includes_edge(x, covered_edges) else uncovered)
            for x in behavior_graph.edges()
        ]
        edge_widths = [
            2 if includes_edge(x, new_covered_edges) else 1
            for x in behavior_graph.edges()
        ]
        nx.draw(
            behavior_graph,
            node_color=node_colors,
            edge_color=edge_colors,
            node_size=node_sizes,
            width=edge_widths,
            with_labels=labels,
            font_size=9,
            pos=pos,
            ax=ax[ax_ind],
        )

    if not pretty:
        for a in ax:
            a.margins(0.10)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if path is None:
        plt.show()
    else:
        Path(remove_after_last_slash(path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{path}.png" if not svg else f"{path}.svg")
        plt.close()
    return pos


def draw_propagation_graph_with_attention_weights(propagation, attention_weights, title="", local=False, svg=False, path=None, pos=None, timestamps=None, pretty=False):
    behavior_graphs = propagation.behavior_graph.snapshots
    propagation_graphs = propagation.snapshots

    uncovered = to_nx_color([105, 105, 105, 128])

    if pos is None:
        pos = nx.spring_layout(behavior_graphs[-1], iterations=100, seed=39775)

    num_snapshots = len(
        behavior_graphs) if timestamps is None else len(timestamps)
    max_col = 5 if timestamps is None else len(timestamps)
    cols = min(num_snapshots, max_col)
    rows = math.ceil(num_snapshots / max_col)
    fig, all_axes = plt.subplots(rows, cols, figsize=(
        cols * 3, 8) if not pretty else (cols * 4, 4))
    ax = all_axes.flat

    snapshot_attentions = []
    for snapshot in attention_weights:
        snapshot_attentions.append([sum(node) for node in snapshot])

    highest_attention = np.max(snapshot_attentions)
    lowest_attention = np.min(snapshot_attentions)

    for i, (behavior_graph, propagation_graph) in enumerate(
        zip(behavior_graphs, propagation_graphs)
    ):
        if timestamps is not None and i not in timestamps:
            continue
        ax_ind = i if timestamps is None else timestamps.index(i)

        ax[ax_ind].set_title(
            f"Δ t{i-1},{i}", fontsize="small" if not pretty else "xx-large")

        snapshot_attention = snapshot_attentions[i - 1] if i > 0 else [
            lowest_attention for x in behavior_graph.nodes()]

        if local:
            highest_attention = np.max(snapshot_attention)
            lowest_attention = np.min(snapshot_attention)

        node_attention_normalized = [
            ((x - lowest_attention) /
             (highest_attention - lowest_attention)) if highest_attention != lowest_attention else 0
            for x in snapshot_attention
        ]
        cmap = plt.cm.Greens
        node_colors = [cmap(val) for val in node_attention_normalized]
        edge_colors = [
            uncovered
            for x in behavior_graph.edges()
        ]
        node_sizes = None if not pretty else [
            500 for x in behavior_graph.nodes()]
        nx.draw(
            behavior_graph,
            node_color=node_colors,
            edge_color=edge_colors,
            node_size=node_sizes,
            with_labels=False,
            font_size=9,
            pos=pos,
            ax=ax[ax_ind],
        )

    if not pretty:
        for a in ax:
            a.margins(0.10)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if path is None:
        plt.show()
    else:
        Path(remove_after_last_slash(path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{path}.png" if not svg else f"{path}.svg")
        plt.close()


def draw_histograms(
    data: List[List[List[float]]],
    metrics: List[str],
    classes: List[str],
    _title="",
    path=None,
    borders: Union[List[Tuple[int, str]], None] = None,
    svg=False
):
    """
    Plots histograms for multiple metrics with different classes.
    Args:
        data: a list of lists, where each top level list is a metric, second level is the class and each value in this sublist is an observation
        metrics: a list of metrics to plot
        classes: a list of classes to plot
    """
    nrows, ncols = get_grid_size(len(data))
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8 * nrows, 8 * ncols),
    )
    ax = ax.flat if nrows * ncols > 1 else [ax]
    for i, metric in enumerate(metrics):
        for j, class_name in enumerate(classes):
            class_mean = round(np.mean(data[i][j]), 2)
            class_std = round(np.std(data[i][j]), 2)
            class_label = f"{class_name} (μ={class_mean}, σ={class_std})"
            ax[i].hist(data[i][j], alpha=0.5, label=class_label)
            if borders is not None:
                for border in borders:
                    plt.axvline(x=border[0], color="red", linestyle="--")
                    plt.text(
                        border[0] - 0.1 * borders[0][0],
                        plt.gca().get_ylim()[1] * 0.9,
                        f"{border[1]}",
                        fontsize=12,
                        ha="center",
                        va="bottom",
                        color="red",
                    )

        ax[i].set_axisbelow(True)
        ax[i].grid(True)
        ax[i].set_title(metric)
        ax[i].legend()
    fig.tight_layout()
    if path is None:
        plt.show()
    else:
        Path(remove_after_last_slash(path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{path}.png" if not svg else f"{path}.svg")
        plt.close()


def surfaceplot(data: pd.DataFrame, x=None, y=None, z=None, cmap='viridis', custom_cmap=None, figsize=(8, 6), ax=None):
    # Extract x, y, z data from the DataFrame
    X = data[x].values if x is not None else np.arange(data.shape[0])
    Y = data[y].values if y is not None else np.arange(data.shape[1])
    Z = data[z].values if z is not None else data.values

    # Create a meshgrid if x and y are specified
    if x is not None and y is not None:
        X, Y = np.meshgrid(X, Y)
        new_Z = np.empty(X.shape)
        new_Z[:, :] = Z[:]
        Z = new_Z

    # Create the surface plot
    sns.set_style("darkgrid")
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    if custom_cmap is not None:
        custom_cmap = np.array(custom_cmap)
        facecolors = np.empty(X.shape + (4,))
        for i in range(4):
            facecolors[:, :, i] = custom_cmap[:, i]
        ax.plot_surface(
            X, Y, Z, facecolors=facecolors, shade=False)
    else:
        ax.plot_surface(X, Y, Z, cmap=cmap)

    # Set labels and title
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)


def get_grid_size(n):
    n = n - 1
    nrows = int(n**0.5) + 1
    ncols = int(n / nrows) + 1
    return nrows, ncols


def save_fig(fig, path: str = None, svg=False):
    fig.tight_layout()
    if path is None:
        plt.show()
    else:
        Path(remove_after_last_slash(path)).mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{path}.png" if not svg else f"{path}.svg")
        plt.close()
