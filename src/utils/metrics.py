from __future__ import annotations
from typing import List, Union
import numpy as np

from src.utils.numbers import get_change_in_percent, round_down
from src.utils.objects import (
    recompute_dict_values,
    flatten_dict,
    get_all_key_combinations,
)


class Metrics:
    def __init__(self) -> None:
        self._ = {}

    def calculate_change(self, value1, value2, threshold):
        change = get_change_in_percent(value2, value1)
        if abs(change) > threshold:
            return round_down(change)

    def compare(self, other: Metrics, threshold: float = 5.0) -> dict:
        return recompute_dict_values(
            self._, other._, lambda v1, v2: self.calculate_change(
                v1, v2, threshold)
        )

    def merge(self, other: Metrics) -> None:
        self._ = recompute_dict_values(self._, other._, lambda v1, v2: v1 + v2)

    def compute_from_list(self, data: Union[dict, List]) -> dict:
        values = list(data.values()) if isinstance(data, dict) else list(data)
        return {
            # "min": min(values) if len(values) > 0 else 0,
            # "max": max(values) if len(values) > 0 else 0,
            "avg": np.mean(values) if len(values) > 0 else 0,
            # "median": np.median(values) if len(values) > 0 else 0,
        }

    def transform_to_list(self, keys: Union[List, None] = None) -> List:
        flat = flatten_dict(self._)
        return (
            list(flat.values())
            if keys is None
            else [flat[key] for key in filter(lambda k: k in keys, flat.keys())]
        )

    def get_all_keys(self) -> List[str]:
        return list(flatten_dict(self._).keys())

    def get_combinations(self):
        return get_all_key_combinations(flatten_dict(self._))

    def update(self, other: Metrics) -> Metrics:
        self._.update(other._)
        return self

    def __str__(self) -> str:
        if len(list(self._.values())) == 0:
            return "No metrics computed."
        return str(self._)
