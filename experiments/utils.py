import argparse
import itertools
import json
import os
import time
import traceback
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import ray
import scipy.stats as sp
import seaborn as sns
from tqdm import tqdm

from src.utils.correlation import bayesian_regression, bonferroni_correction
from src.utils.drawing import get_grid_size, save_fig
from src.utils.numbers import clamped_sample_fn
from src.utils.objects import safely_parse_list
from src.utils.path import path_exists, remove_after_last_slash
from src.utils.results import Result


def default_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument(
        "--recreate", action="store_true", help="Recreate datasets for each run"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Analyzes the results of the experiments"
    )
    parser.add_argument(
        "--no-execute", action="store_true", help="Do not execute the experiments"
    )
    parser.add_argument(
        "--parallelize", action="store_true", help="Parallelize the experiments"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Compute more statistics from the results"
    )
    return parser


def default_runner(run_experiments: Callable):
    parser = default_arg_parse()
    args = parser.parse_args()

    num_runs = args.runs if not args.no_execute else 1
    for i in range(num_runs):
        s = time.time()
        print(f"Start run {i+1}.")
        run_experiments(
            recreate=args.recreate,
            analyze=args.analyze and i == num_runs-1,
            execute=not args.no_execute,
            stats=args.stats and i == num_runs-1,
            parallelize=args.parallelize,
        )
        print(f"Finish run {i+1} after {int(time.time() - s)} seconds.")


def default_result_columns():
    return ["dataset", "type", "model", "metric", "value"]


def read_results(path, columns=None, fileName="results"):
    if columns is None:
        columns = default_result_columns()
    if not path_exists(f"{path}/{fileName}.csv"):
        return pd.DataFrame(columns=columns)
    return pd.read_csv(f"{path}/{fileName}.csv")


def save_results(df, path):
    return df.to_csv(f"{path}/results.csv", index=False)


def append_to_results(
    results: pd.DataFrame,
    model: str,
    dataset: str,
    result: Dict[str, Result],
    more_values: dict = None,
    metric_filter: List[str] = None,
):
    for result_type, result_type_result in result.items():
        for (
            metric,
            metric_value,
        ) in result_type_result.to_dict().items():
            if metric_filter is not None and metric not in metric_filter:
                continue

            new_result = {
                "model": model,
                "dataset": dataset,
                "type": result_type,
                "metric": metric,
                "value": metric_value,
                **(more_values if more_values is not None else {}),
            }
            results.loc[len(results)] = new_result


def compute_means_with_significance(results: pd.DataFrame, group_col: str, value_col="value", path: str = '', file_prefix: str = "", filter_colum="test"):
    complete_test_results = results[results["type"] == filter_colum]
    # first we compute all the means for the group column
    Path(remove_after_last_slash(f"{path}/")
         ).mkdir(parents=True, exist_ok=True)
    groups = complete_test_results.groupby(["metric", group_col])
    means = groups[value_col].mean().reset_index()
    means.to_csv(f"{path}/{file_prefix}{group_col}_means.csv", index=False)
    groups = complete_test_results.groupby(["metric", group_col])
    stds = groups[value_col].std().reset_index()
    stds.to_csv(f"{path}/{file_prefix}{group_col}_stds.csv", index=False)

    # then we compute if the means are significantly different
    unique_col_values = complete_test_results[group_col].unique()
    unique_metric_values = complete_test_results["metric"].unique()
    sign = pd.DataFrame(
        [], columns=["metric", "left", "right", "stat", "p", "significant"])
    for val1, val2 in list(itertools.combinations(unique_col_values, 2)):
        for metric in unique_metric_values:
            values1 = complete_test_results[(complete_test_results[group_col]
                                            == val1) & (complete_test_results["metric"] == metric)][value_col]
            values2 = complete_test_results[(complete_test_results[group_col]
                                            == val2) & (complete_test_results["metric"] == metric)][value_col]
            stat_value, p_value = sp.mannwhitneyu(
                values1, values2, alternative='two-sided')
            sign.loc[len(sign)] = {
                "metric": metric,
                "left": val1,
                "right": val2,
                "stat": stat_value,
                "p": p_value,
                "significant":  p_value < bonferroni_correction(0.05, len(unique_col_values)-1),
            }
    sign = sign.sort_values(by=["metric", "left"], ascending=True)
    sign.to_csv(
        f"{path}/{file_prefix}{group_col}_significance.csv", index=False)


def compute_bay_regression_per_group(results: pd.DataFrame, group_key: str, entity_key: str, predictor_column: str, path: str):
    regression_results = pd.DataFrame([], columns=["dataset", group_key, entity_key,
                                                   "means", "stds", "size", "r_squared"])

    # we do regression only on accuracy of the test bucket
    test_results = results[results['type'] == 'test']
    test_results = test_results[test_results['metric'] == 'accuracy']

    for dataset_name, dataset_df in test_results.groupby("dataset"):
        dataset_path = f"{path}/{dataset_name}"
        for group_name, group_df in dataset_df.groupby(group_key):
            group_path = f"{dataset_path}/{group_name}"

            # Convert a stringified list column to a list column if necessary
            group_df[predictor_column] = group_df[predictor_column].apply(
                safely_parse_list)
            group_df = group_df.reset_index()

            # Create new columns for each value in the list column
            max_list = max([len(x) if isinstance(x, list)
                            else 1 for x in group_df[predictor_column]])
            for j in range(max_list):
                group_df[f'{predictor_column}{j}'] = group_df[predictor_column].apply(
                    lambda x: x[j] if isinstance(x, list) else x)
            # Drop the original column
            group_df.drop(columns=[predictor_column], inplace=True)

            for entity_name, entity_df in group_df.groupby(entity_key):
                try:
                    result_trace, r_squared = bayesian_regression(entity_df, predictor_columns=list(
                        entity_df.filter(like=predictor_column)), target_column="value")
                    az.plot_trace(result_trace)
                    save_fig(plt, f"{group_path}/br_trace_{entity_name}")
                    az.plot_posterior(result_trace)
                    save_fig(
                        plt, f"{group_path}/br_posterior_{entity_name}")

                    beta_means = result_trace.posterior["beta"].values.mean(
                        axis=(0, 1))
                    beta_stds = result_trace.posterior["beta"].values.std(
                        axis=(0, 1))
                    size = len(entity_df)
                    regression_results.loc[len(regression_results)] = [
                        dataset_name, group_name, entity_name, list(beta_means), list(beta_stds), size, r_squared]
                except Exception:
                    print(
                        f"Could not compute regression for {dataset_name} {group_name} {entity_name}")
    regression_results.to_csv(f"{path}/{entity_key}_regression_results.csv")
    return regression_results


def compute_t_test_significance_from_bay_regression_results(results: pd.DataFrame, group_key: str, entity_key: str, path: str):
    significance_results = pd.DataFrame([], columns=[
                                        "dataset", group_key, entity_key, f"other_{entity_key}", "t_statistic", "p_value", "is_sign"])

    def sample_coefficient(mean, std):
        return clamped_sample_fn(
            mean=abs(mean), std=std, round_int=False)()

    def samples_from_coefficients(means, stds):
        inputs = [int(x/10) for x in range(100)]
        samples = []
        for x in inputs:
            coeffs = [sample_coefficient(mean, std)
                      for mean, std in zip(means, stds)]
            samples.append(sum([coeff * x for i, coeff in enumerate(coeffs)]))
        return samples

    for dataset_name, dataset_df in results.groupby("dataset"):
        for group_name, group_df in dataset_df.groupby(group_key):
            entities = group_df[entity_key].unique()
            for entity_name, entity_df in group_df.groupby(entity_key):
                for other_entity_name, other_entity_df in group_df.groupby(entity_key):
                    if entity_name == other_entity_name:
                        continue

                    my_samples = samples_from_coefficients(
                        means=safely_parse_list(entity_df["means"].values[0]), stds=safely_parse_list(entity_df["stds"].values[0]))
                    other_samples = samples_from_coefficients(
                        means=safely_parse_list(other_entity_df["means"].values[0]), stds=safely_parse_list(other_entity_df["stds"].values[0]))

                    # test if slope is smaller for my samples
                    t_statistic, p_value = sp.ttest_ind(
                        my_samples, other_samples, alternative="less")

                    is_sign = p_value < bonferroni_correction(
                        0.05, len(entities)-1)
                    significance_results.loc[len(significance_results)] = [
                        dataset_name, group_name, entity_name, other_entity_name, t_statistic, p_value, is_sign]

    significance_results.to_csv(
        f"{path}/{entity_key}_significance_results.csv")
    return significance_results


def plot_t_test_significance_rate(results: pd.DataFrame, group_key: str, entity_key: str,  path: str):
    results['is_sign'] = results['is_sign'].astype(
        bool)
    sign_count = results.groupby(
        ["dataset", group_key, entity_key])['is_sign'].sum()
    total_count = results.groupby(
        ["dataset", group_key, entity_key]).size()
    sign_rate_df = (sign_count / total_count).reset_index(name='sign_rate')

    # non-grouped by dataset results
    global_sign_rate_df = sign_rate_df.groupby(
        [group_key, entity_key])['sign_rate'].mean().reset_index()
    global_sign_rate_df.to_csv(
        f"{path}/{entity_key}_sign_rate_combined.csv")

    for group_name, group_df in sign_rate_df.groupby(group_key):
        nrows, ncols = get_grid_size(len(group_df["dataset"].unique()))
        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(10 * ncols, 10 * nrows),
        )
        ax = ax.flatten() if nrows * ncols > 1 else [ax]
        for j, (dataset_name, dataset_df) in enumerate(group_df.groupby("dataset")):
            sns.barplot(data=dataset_df, x=entity_key, y="sign_rate", ax=ax[j])
            ax[j].set_title(f"{dataset_name}")
        save_fig(fig, f"{path}/{entity_key}_sign_rate_{group_name}")


# Ray requires that the function you want to parallelize is a standalone function, not a method on an object.
# So, here's your run_experiment_wrapper function, slightly modified to handle Ray's requirements.


@ray.remote
def run_experiment_wrapper(run_experiment, combination):
    try:
        return run_experiment(combination)
    except BaseException as exception:
        # Return the custom error class with the error message
        return str(exception)


def parallelize_experiment_runs(
    results: pd.DataFrame,
    combinations: List,
    run_experiment: Callable,
    parallelize: bool,
    path: str,
    timeout_in_min=60,
    on_result=None,
    metric_filter=None,
):

    # Execute experiments in parallel
    if parallelize:
        # NOTE: reduce cpu count by 1 to avoid overloading the system
        num_cpus = max(os.cpu_count() - 2, 1)
        print(f"Parallelize with {num_cpus} workers")
        os.environ["RAY_PROFILING"] = "1"
        os.environ["RAY_task_events_report_interval_ms"] = "1000"
        ray.init(num_cpus=num_cpus)
        result_ids = [run_experiment_wrapper.remote(
            run_experiment, comb) for comb in combinations]

        all_results = []
        for i, result_id in enumerate(result_ids):
            try:
                result = ray.get(result_id, timeout=timeout_in_min * 60)
                all_results.append(result)
            except ray.exceptions.GetTimeoutError:
                print(
                    f"TIMEOUT: Task {result_id} with data {combinations[i]} did not complete within the specified time.")

        # Capture the timeline
        time.sleep(1)
        timeline = ray.timeline()

        if timeline is not None:
            with open(f"{path}/timeline.json", "w", encoding="utf-8") as f:
                json.dump(timeline, f)

        # Combine results into a single DataFrame
        for j, result_list in tqdm(enumerate(all_results), 'Collect Results'):
            if isinstance(result_list, str):  # An error occurred
                print(f"Worker raised: {result_list}")
                print(f"Failed combination: {combinations[j]}")
            else:
                try:
                    for result in result_list:
                        if on_result is not None:
                            on_result(result)
                        else:
                            append_to_results(
                                results=results, metric_filter=metric_filter, **result)
                except Exception as e:
                    print(f"Error appending results: {e}")
                    print(traceback.format_exc())
        save_results(results, path)
        ray.shutdown()
    else:
        # Execute experiments sequentially
        for combination in combinations:
            result_list = run_experiment(combination)
            for result in result_list:
                if on_result is not None:
                    on_result(result)
                else:
                    append_to_results(
                        results=results,  metric_filter=metric_filter, **result)
                save_results(results, path)


def serialize_enum_values(enum_values: List):
    return [{"name": value.name} for value in enum_values]


def deserialize_enum_values(enum_values_serialized: List, enum: Enum):
    return [enum[value["name"]] for value in enum_values_serialized]
