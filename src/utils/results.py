from typing import Dict, List

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


class Result:
    """Holds the evaluation metrics results."""

    def __init__(self, labels=None, predictions=None):
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

        self.labels = labels if labels is not None else []
        self.predictions = predictions if predictions is not None else []
        self.threshold = None
        self.last = None
        self.computed = False

    def add(self, labels, predictions):
        self.labels += labels
        self.predictions += predictions

    def compute(self):
        if len(self.labels) == 0 or len(self.predictions) == 0:
            return

        accuracy = accuracy_score(self.labels, self.predictions)
        precision = precision_score(
            self.labels, self.predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            self.labels, self.predictions, average="macro", zero_division=0
        )
        f1 = f1_score(self.labels, self.predictions,
                      average="macro", zero_division=0)

        if accuracy > self.accuracy:
            self.accuracy = accuracy
        if precision > self.precision:
            self.precision = precision
        if recall > self.recall:
            self.recall = recall
        if f1 > self.f1:
            self.f1 = f1

        self.labels = []
        self.predictions = []
        self.last = (accuracy, precision, recall, f1)
        self.computed = True

    def to_dict(self):
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    def clear(self):
        # delete temporary attributes, as we don't want to persist them
        del self.labels
        del self.predictions
        del self.threshold
        del self.last
        del self.computed

    def __str__(self):
        return (
            f"Accuracy: {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall: {self.recall:.4f}\n"
            f"F1 Score: {self.f1:.4f}"
        )


class FakeResult():
    def __init__(self, result: dict):
        self.result = result

    def add_to_property(self, key, value):
        self.result[key] += value

    def to_dict(self):
        return self.result


def create_result_buckets(num_buckets=4):
    """Creates a dictionary of results for each bucket"""
    results: Dict[str, Result] = {}
    results["train"] = Result()
    results["test"] = Result()

    if num_buckets > 0:
        steps = 1 / num_buckets
        buckets = np.arange(0 + steps, 1 + steps, steps)
        for bucket in buckets:
            results[f"test_{bucket}"] = Result()
            results[f"test_{bucket}"].threshold = bucket

        results["test_avg"] = Result()
    return results


def compute_result_buckets(
    results: Dict[str, Result],
    labels: List[int],
    predictions: List[int],
    collision_scores: List[float] = None,
    train=False,
):
    """Add labels and predictions to a result bucket depending on given threshold."""
    if train:
        results["train"].add(labels=labels, predictions=predictions)

    elif collision_scores is None:
        results["test"].add(labels=labels, predictions=predictions)

    else:
        results["test"].add(labels=labels, predictions=predictions)

        # add the test labels to the correct bucket by using the threshold
        buckets = list(
            filter(lambda bucket: bucket.threshold is not None, results.values())
        )
        buckets = list(sorted(buckets, key=lambda bucket: bucket.threshold))
        for label, prediction, collision_score in zip(
            labels, predictions, collision_scores
        ):
            for bucket in buckets:
                if collision_score is not None and collision_score < bucket.threshold:
                    bucket.add(labels=[label], predictions=[prediction])
                    break

    for bucket in results.values():
        bucket.compute()

    # add the average of all buckets to the results
    if collision_scores is not None:
        threshold_buckets = list(
            filter(
                lambda bucket: bucket.threshold is not None and bucket.computed,
                results.values(),
            )
        )
        for attr_name in vars(results["test_avg"]):
            if not isinstance(getattr(results["test_avg"], attr_name), (int, float)):
                continue
            bucket_values = [getattr(bucket, attr_name)
                             for bucket in threshold_buckets]
            setattr(
                results["test_avg"],
                attr_name,
                np.mean(bucket_values) if len(bucket_values) > 0 else 0,
            )


def clear_result_buckets(results: Dict[str, Result]):
    for bucket in results.values():
        bucket.clear()
