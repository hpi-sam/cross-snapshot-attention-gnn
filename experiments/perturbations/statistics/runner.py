from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiments.utils import (default_runner, parallelize_experiment_runs,
                               read_results, save_results)
from src.datasets.random.synthetic import SyntheticDataset
from src.generation.behavior.behavior import BehaviorGraphMetrics
from src.generation.propagation.propagation import PropagationMetrics
from src.perturbations.enum import (PerturbationFactorMethod,
                                    PerturbationMethod, get_perturbation_class)
from src.utils.correlation import bayesian_regression
from src.utils.drawing import get_grid_size, save_fig

metrics = [
    *BehaviorGraphMetrics().get_all_keys(), *PropagationMetrics().get_all_keys()]


def analyze_results(results: pd.DataFrame, path: str):
    for perturbation, perturbation_df in results.groupby("perturbation"):
        perturbation_path = f"{path}/{perturbation}"
        Path(perturbation_path).mkdir(parents=True, exist_ok=True)
        for metric in metrics:
            nrows, ncols = get_grid_size(
                len(perturbation_df["graph"].unique()))
            fig, ax = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(8 * ncols, 8 * nrows),
            )
            ax = ax.flatten() if nrows * ncols > 1 else [ax]
            metric_path = f"{perturbation_path}/{metric}"
            Path(metric_path).mkdir(parents=True, exist_ok=True)
            for j, graph in enumerate(perturbation_df["graph"].unique()):
                graph_results = perturbation_df[perturbation_df["graph"] == graph]
                sns.boxplot(
                    ax=ax[j],
                    data=graph_results,
                    x="factor",
                    y=metric,
                )
                ax[j].set_title(f"{graph}")
            save_fig(fig, f"{metric_path}/boxplot")


def compute_regression(results: pd.DataFrame, regression_results: pd.DataFrame, path: str):
    for graph_type, graph_type_df in results.groupby("graph"):
        for perturbation_name, perturbation_df in graph_type_df.groupby("perturbation"):
            perturbation_path = f"{path}/{perturbation_name}"
            for metric in metrics:
                metric_path = f"{perturbation_path}/{metric}"
                try:
                    result_trace, r_squared = bayesian_regression(perturbation_df, predictor_columns=list(
                        perturbation_df.filter(like='factor')), target_column=metric)
                    az.plot_trace(result_trace)
                    save_fig(plt, f"{metric_path}/br_trace_{graph_type}")
                    az.plot_posterior(result_trace)
                    save_fig(plt, f"{metric_path}/br_posterior_{graph_type}")

                    # Extract parameters from InferenceData object
                    alpha_mean = result_trace.posterior["alpha"].values.mean()
                    beta_mean = result_trace.posterior["beta"].values.mean(
                        axis=(0, 1))
                    sigma_mean = result_trace.posterior["sigma"].values.mean()

                    # Create a dictionary with the mean parameter values and Bayesian R-squared
                    mean_values = {
                        "graph": graph_type,
                        "perturbation": perturbation_name,
                        "alpha": alpha_mean,
                        **{f"beta{i}": beta_mean[i] for i in range(beta_mean.size)},
                        "sigma": sigma_mean,
                        "r_squared": r_squared}
                    regression_results.loc[len(
                        regression_results)] = mean_values
                except Exception as e:
                    print(e)


def compute_stats(results: pd.DataFrame, path: str):
    columns = ["graph", "perturbation",
               "alpha", "beta0", "sigma", "r_squared"]
    regression_results = pd.DataFrame([], columns=columns)

    compute_regression(results, regression_results, path)
    regression_results.to_csv(f"{path}/regression_results.csv")


def run_experiment(exp_args):
    (
        graph_type,
        perturbation,
        factor,
    ) = exp_args

    results = []
    print(f"Run {graph_type}, {perturbation}, {factor}")

    if perturbation.startswith("anchoring"):
        factors = {"anchoring_factor": factor}
        dataset_name = f"syn_{perturbation.split('_')[1]}_anchoring_{factors}"
        dataset = SyntheticDataset(
            log_progress=True,
            train_behavior_distribution={graph_type: 1},
            test_behavior_distribution={graph_type: 1},
            num_samples=200,
            prevent_transform=True,
            name=dataset_name,
            abbreviation=dataset_name,
            anchoring_method=perturbation.split("_")[1],
            anchoring_factor=factor,
        )
    else:
        dataset = SyntheticDataset(
            log_progress=True,
            train_behavior_distribution={graph_type: 1},
            test_behavior_distribution={graph_type: 1},
            num_samples=200,
            prevent_transform=True,
        )
        if factor != 0:
            dataset.perturb(
                perturbation=get_perturbation_class(
                    getattr(PerturbationMethod, perturbation)
                ),
                train=True,
                test=True,
                persist=False,
                **getattr(PerturbationFactorMethod, perturbation)(factor),
            )

    for sample in [*dataset.train, *dataset.test]:
        bg_metrics = BehaviorGraphMetrics(
            sample.propagation.behavior_graph).transform_to_list()
        prop_metrics = PropagationMetrics(
            sample.propagation).transform_to_list()
        results.append(
            [graph_type, perturbation.lower(), factor, *bg_metrics, *prop_metrics])

    return results


def run_experiments(recreate=False, analyze=True, execute=True, stats=False, parallelize=False):
    path = "experiments/perturbations/statistics"
    results = read_results(path, columns=[
                           "graph", "perturbation", "factor", *metrics])
    if execute:
        results = pd.DataFrame(
            [], columns=results.columns
        )
        graph_types = ["er", "ba", "ws", "exp"]
        perturbations = [*list(PerturbationMethod.__members__.keys()),
                         "anchoring_degree", "anchoring_closeness", "anchoring_betweenness"]
        factors = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        anchoring_factors = [0, 1, 2, 3, 4]

        combinations = [
            (
                graph_type,
                perturbation,
                factor,
            )
            for graph_type in graph_types
            for perturbation in perturbations
            for factor in (anchoring_factors if perturbation.startswith("anchoring") else factors)
        ]

        def append_result(result):
            results.loc[len(results)] = result

        parallelize_experiment_runs(
            results=results,
            combinations=combinations,
            run_experiment=run_experiment,
            parallelize=parallelize,
            path=path,
            on_result=append_result,
            timeout_in_min=60,
        )

        save_results(results, path)
    if analyze:
        analyze_results(results, path)
    if stats:
        compute_stats(results, path)


if __name__ == "__main__":
    default_runner(run_experiments)
