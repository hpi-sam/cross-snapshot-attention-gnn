from itertools import combinations
from typing import List
import random
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, wasserstein_distance
from tqdm import tqdm

from src.datasets.dataset import Dataset, Sample
from src.generation.behavior.behavior import BehaviorGraphMetrics
from src.generation.propagation.propagation import (
    PropagationMetrics,
)
from src.utils.correlation import bonferroni_correction
from src.utils.drawing import (
    draw_histograms,
    draw_multiple_heatmaps,
    draw_propagation_graph,
)


class DatasetAnalyzer:
    """Creates analysis of a dataset."""

    def __init__(
        self,
        dataset: Dataset,
        num_sample_plots=3,
        path: str = None,
    ) -> None:
        self.dataset = dataset
        self.num_sample_plots = num_sample_plots
        self.path = f"{path}/{dataset.name}" if path else None
        if self.path is not None:
            self.analyze()

    def analyze(self, samples=True, metrics=True, collisions=True):
        if samples:
            self.plot_samples()
        if metrics:
            self.plot_propagation_metrics()
            self.plot_behavior_metrics()
        if collisions:
            self.plot_collisions()

    def compute_behavior_metrics(self):
        samples = [*self.dataset.train, *self.dataset.test]
        metrics = [[[]] for _ in BehaviorGraphMetrics().get_all_keys()]
        for sample in tqdm(samples, "Computing behavior metrics"):
            results = BehaviorGraphMetrics(
                sample.propagation.behavior_graph
            ).transform_to_list()
            for i, result in enumerate(results):
                metrics[i][0].append(result)
        return metrics

    def compute_propagation_metrics(self):
        samples = [*self.dataset.train, *self.dataset.test]
        metrics = [
            [[] for _ in self.dataset.labels]
            for _ in PropagationMetrics().get_all_keys()
        ]
        for sample in tqdm(samples, "Computing propagation metrics"):
            results = PropagationMetrics(
                sample.propagation).transform_to_list()
            for i, result in enumerate(results):
                metrics[i][sample.label].append(result)
        return metrics

    def compute_distribution_similarity(self, data: List[List[List[float]]]):
        """Computes the similarity between distributions of metrics.
        Uses the kolmogorov-smirnov test and the wasserstein distance.
        """
        num_classes = len(data[0][0])
        num_tests = int(len(data[0]) * ((num_classes * (num_classes - 1)) / 2))
        alpha_value = bonferroni_correction(0.05, num_tests)

        ks_data = []
        ks_p = []
        ws_data = []
        for _, distribution_data in enumerate(data):
            ks_data.append([])
            ks_p.append([])
            ws_data.append([])
            for _, label_data_1 in enumerate(distribution_data):
                ks_data[-1].append([])
                ks_p[-1].append([])
                ws_data[-1].append([])
                for _, label_data_2 in enumerate(distribution_data):
                    ks_score, ks_p_value = ks_2samp(label_data_1, label_data_2)
                    ws_score = wasserstein_distance(label_data_1, label_data_2)
                    ks_data[-1][-1].append(ks_score)
                    ks_p[-1][-1].append(1 if ks_p_value <= alpha_value else 0)
                    ws_data[-1][-1].append(ws_score)

        return ks_data, ks_p, ws_data

    def compute_collisions(
        self,
        samples_l: List[Sample],
        samples_r: List[Sample],
        name: str,
    ):
        collisions = []
        for sample_l in tqdm(samples_l, f"Computing {name} collisions"):
            _, max_thresold = sample_l.propagation.matches_most(
                [sample_r.propagation for sample_r in samples_r]
            )
            collisions.append(max_thresold)
        return collisions

    def compute_label_collisions(self, left_samples, right_samples):
        label_collisions = []
        label_combinations = list(combinations(
            range(len(self.dataset.labels)), 2))
        for label_l, label_r in label_combinations:
            samples_l = [
                sample for sample in left_samples if sample.label == label_l]
            samples_r = [
                sample for sample in right_samples if sample.label == label_r]
            collisions = self.compute_collisions(
                samples_l,
                samples_r,
                f"{self.dataset.labels[label_l].name} - {self.dataset.labels[label_r].name}",
            )
            label_collisions.append(collisions)
        return label_collisions, [
            f"{self.dataset.labels[label_l].name} - {self.dataset.labels[label_r].name}"
            for label_l, label_r in label_combinations
        ]

    def compute_test_collisions(self):
        collisions = self.compute_collisions(
            self.dataset.test,
            self.dataset.test,
            "test - test",
        )
        label_collisions, label_names = self.compute_label_collisions(
            self.dataset.test, self.dataset.test
        )
        return [collisions, label_collisions], [["test - test"], label_names]

    def compute_test_train_collisions(self):
        collisions = self.compute_collisions(
            self.dataset.test,
            self.dataset.train,
            "test - train",
        )
        label_collisions, label_names = self.compute_label_collisions(
            self.dataset.test, self.dataset.train
        )
        return [collisions, label_collisions], [["test - train"], label_names]

    def plot_samples(self):
        if self.path is None:
            return
        samples = []
        for label in range(len(self.dataset.labels)):
            filtered_samples = filter(
                lambda sample: sample.label == label, self.dataset.test
            )
            samples += list(
                random.sample(list(filtered_samples), self.num_sample_plots)
            )

        for i, sample in enumerate(tqdm(samples, "Plot samples")):
            label = self.dataset.labels[sample.label]
            path = (
                f"{self.path}/samples/{i%self.num_sample_plots}_{label.name}"
                if self.path is not None
                else None
            )
            draw_propagation_graph(
                sample.propagation, f"Label={label.name}", path)

    def plot_metrics_similarity(
        self, data: List[List[List[float]]], metrics: List[str], path: str
    ):
        ks_data, ks_p, ws_data = self.compute_distribution_similarity(data)
        classes = [label.name for label in self.dataset.labels]

        draw_multiple_heatmaps(
            data=ks_data,
            path=f"{path}/kolmogorov_smirnov_statistic",
            titles=[
                f"Kolmogorov-Smirnov Score - {title}" for title in metrics],
            xlabels=classes,
            ylabels=classes,
            vmin=-1,
            vmax=1,
            annot=True,
            triangle=True,
        )
        draw_multiple_heatmaps(
            data=ks_p,
            path=f"{path}/kolmogorov_smirnov_p",
            titles=[
                f"Kolmogorov-Smirnov P Values - {title}" for title in metrics],
            xlabels=classes,
            ylabels=classes,
            triangle=True,
        )
        draw_multiple_heatmaps(
            data=ws_data,
            path=f"{path}/wasserstein_distance",
            titles=[f"Wasserstein Distance - {title}" for title in metrics],
            xlabels=classes,
            ylabels=classes,
            vmin=-1,
            vmax=1,
            annot=True,
            triangle=True,
        )

    def plot_behavior_metrics(self):
        data = self.compute_behavior_metrics()
        metrics = BehaviorGraphMetrics().get_all_keys()
        classes = [self.dataset.name]
        path = f"{self.path}/behavior_metrics" if self.path is not None else None
        if path is not None:
            draw_histograms(
                data,
                metrics,
                classes,
                f"{self.dataset.name} - Behavior Metrics",
                f"{path}/histogram",
            )

    def plot_propagation_metrics(self):
        data = self.compute_propagation_metrics()
        metrics = PropagationMetrics().get_all_keys()
        classes = [label.name for label in self.dataset.labels]
        path = f"{self.path}/propagation_metrics" if self.path is not None else None
        if path is not None:
            draw_histograms(
                data,
                metrics,
                classes,
                f"{self.dataset.name} - Propagation Metrics",
                f"{path}/histogram",
            )
            self.plot_metrics_similarity(data, metrics, path)

    def plot_collisions(self):
        test_collisions, test_collision_names = self.compute_test_collisions()
        (
            test_train_collisions,
            test_train_collision_names,
        ) = self.compute_test_train_collisions()
        test_label_similarity = self.compute_distribution_similarity(
            [test_collisions[-1]]
        )

        plt.rcParams.update({'font.size': 14})
        plt.rcParams['legend.fontsize'] = 14

        path = f"{self.path}/collisions" if self.path is not None else None
        if path is not None:
            draw_histograms(
                [[test_collisions[0]]],
                ["Test Collisions (Complete)"],
                test_collision_names[0],
                f"{self.dataset.name} - Test Collisions (Complete)",
                f"{path}/histogram_test_complete",
            )
            draw_histograms(
                [test_collisions[1]],
                ["Test Collisions (Labels)"],
                test_collision_names[1],
                f"{self.dataset.name} - Test Collisions (Labels)",
                f"{path}/histogram_test_labels",
            )
            borders = [0, 0.25, 0.5, 0.75, 1]
            bucket_sizes = [
                (
                    borders[i + 1],
                    sum(
                        borders[i] <= value < borders[i + 1]
                        for value in test_train_collisions[0]
                    ),
                )
                for i in range(len(borders) - 1)
            ]
            draw_histograms(
                [[test_train_collisions[0]]],
                ["Test-Train Collisions (Complete)"],
                test_train_collision_names[0],
                f"{self.dataset.name} - Test-Train Collisions (Complete)",
                f"{path}/histogram_test_train_complete",
                borders=bucket_sizes,
                svg=True,
            )
            draw_histograms(
                [test_train_collisions[1]],
                ["Test-Train Collisions (Labels)"],
                test_train_collision_names[1],
                f"{self.dataset.name} - Test-Train Collisions (Labels)",
                f"{path}/histogram_test_train_labels",
            )
            draw_multiple_heatmaps(
                data=test_label_similarity[0],
                path=f"{path}/kolmogorov_smirnov_statistic_test_labels",
                titles=["Kolmogorov-Smirnov Statistic"],
                xlabels=test_collision_names[-1],
                ylabels=test_collision_names[-1],
                vmin=-1,
                vmax=1,
                annot=True,
                triangle=True,
            )
            draw_multiple_heatmaps(
                data=test_label_similarity[1],
                path=f"{path}/kolmogorov_smirnov_p_test_labels",
                titles=["Kolmogorov-Smirnov P Values"],
                xlabels=test_collision_names[-1],
                ylabels=test_collision_names[-1],
                triangle=True,
            )
            draw_multiple_heatmaps(
                data=test_label_similarity[2],
                path=f"{path}/wasserstein_distance_test_labels",
                titles=["Wasserstein Distance"],
                xlabels=test_collision_names[-1],
                ylabels=test_collision_names[-1],
                vmin=-1,
                vmax=1,
                annot=True,
                triangle=True,
            )
        plt.rcParams.update(
            {'font.size': plt.rcParamsDefault['font.size']})
