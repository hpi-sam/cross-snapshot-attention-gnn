from typing import Callable, Union
import torch
import copy
import random

from src.datasets.dataset import Dataset
import src.models.metrics.linear_regression as linear_regression
import src.models.metrics.xgboost as xgboost
import src.models.metrics.mlp as mlp
from src.generation.behavior.behavior import BehaviorGraphMetrics
from src.generation.propagation.anchoring import (
    BetweennessCentralityAnchoringMetrics,
    ClosenessCentralityAnchoringMetrics,
    DegreeCentralityAnchoringMetrics,
)
from src.generation.propagation.propagation import Propagation, PropagationMetrics


def transform_data(
    dataset: Dataset,
    compute_metrics: Callable,
    is_tensor=False,
    training_share=None,
):
    if training_share is not None:
        data = copy.deepcopy(dataset.train)
        random.shuffle(data)
        train_data = data[: int(len(data) * training_share)]
    else:
        train_data = dataset.train

    x_train = [
        compute_metrics(sample.propagation).transform_to_list() for sample in train_data
    ]
    y_train = [
        sample.label if not is_tensor else [sample.label] for sample in train_data
    ]

    x_test = [
        compute_metrics(sample.propagation).transform_to_list()
        for sample in dataset.test
    ]
    y_test = [
        sample.label if not is_tensor else [sample.label] for sample in dataset.test
    ]
    x_train_collision_score = [
        sample.max_collision_score for sample in train_data]
    x_test_collision_score = [
        sample.max_collision_score for sample in dataset.test]

    if is_tensor:
        return (
            torch.FloatTensor(x_train),
            torch.FloatTensor(y_train),
            torch.FloatTensor(x_test),
            torch.FloatTensor(y_test),
            x_train_collision_score,
            x_test_collision_score,
        )
    return (
        x_train,
        y_train,
        x_test,
        y_test,
        x_train_collision_score,
        x_test_collision_score,
    )


def get_propagation_metrics(x: Union[Propagation, None] = None):
    return PropagationMetrics(x)


def get_anchoring_metrics(x: Union[Propagation, None] = None):
    if x is None:
        return DegreeCentralityAnchoringMetrics().update(
            ClosenessCentralityAnchoringMetrics().update(
                BetweennessCentralityAnchoringMetrics()
            )
        )
    return DegreeCentralityAnchoringMetrics(
        x.behavior_graph.snapshots[0], x.anchor
    ).update(
        ClosenessCentralityAnchoringMetrics(
            x.behavior_graph.snapshots[0], x.anchor
        ).update(
            BetweennessCentralityAnchoringMetrics(
                x.behavior_graph.snapshots[0], x.anchor
            )
        )
    )


def get_behavior_metrics(x: Union[Propagation, None] = None):
    if x is None:
        return BehaviorGraphMetrics()
    return BehaviorGraphMetrics(x.behavior_graph)


def get_combined_metrics(x: Union[Propagation, None] = None):
    return get_propagation_metrics(x).update(
        get_anchoring_metrics(x).update(get_behavior_metrics(x))
    )


def execute(
    dataset: Dataset,
    method: str,
    metrics_flavor: str,
    training_share=None,
    path: str = "",
    return_model=False,
    profile=False,
):
    if metrics_flavor == "propagation":
        compute_metrics = get_propagation_metrics
    elif metrics_flavor == "anchoring":
        compute_metrics = get_anchoring_metrics
    elif metrics_flavor == "behavior":
        compute_metrics = get_behavior_metrics
    else:
        compute_metrics = get_combined_metrics
    if method == "linear_regression":
        return linear_regression.train_and_test(
            *transform_data(dataset, compute_metrics,
                            training_share=training_share),
            return_model=return_model,
            profile=profile,
        )
    elif method == "xgboost":
        feature_names = compute_metrics().get_all_keys()
        return xgboost.train_and_test(
            *transform_data(dataset, compute_metrics,
                            training_share=training_share),
            feature_names,
            path,
            metrics_flavor,
            return_model=return_model,
            profile=profile,
        )
    elif method == "mlp":
        return mlp.train_and_test(
            *transform_data(
                dataset,
                compute_metrics,
                is_tensor=True,
                training_share=training_share,
            ),
            return_model=return_model,
            profile=profile,
        )
    else:
        raise ValueError(f"Unknown method {method}")


def test(
    dataset: Dataset,
    method: str,
    metrics_flavor: str,
    model,
):
    if metrics_flavor == "propagation":
        compute_metrics = get_propagation_metrics
    elif metrics_flavor == "anchoring":
        compute_metrics = get_anchoring_metrics
    elif metrics_flavor == "behavior":
        compute_metrics = get_behavior_metrics
    else:
        compute_metrics = get_combined_metrics
    if method == "linear_regression":
        data = transform_data(
            dataset,
            compute_metrics,
        )
        return linear_regression.test(model, data[2], data[3], data[5])
    elif method == "xgboost":
        data = transform_data(
            dataset,
            compute_metrics,
        )
        return xgboost.test(model, data[2], data[3], data[5])
    elif method == "mlp":
        data = transform_data(
            dataset,
            compute_metrics,
            is_tensor=True,
        )
        return mlp.test(model, data[2], data[3], data[5])
    else:
        raise ValueError(f"Unknown method {method}")
