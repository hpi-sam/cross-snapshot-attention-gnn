import ast
from itertools import chain, combinations
from typing import Callable, List

import numpy as np


def recompute_dict_values(obj1: dict, obj2: dict, func: Callable) -> dict:
    res = {}
    for key in obj1.keys():
        v1 = obj1[key]
        v2 = obj2[key]
        if type(v1) is dict:
            res[key] = recompute_dict_values(v1, v2, func)
        else:
            new_v = func(v1, v2)
            if new_v is not None:
                res[key] = new_v
    return res


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_all_key_combinations(dictionary, limit=None):
    keys = list(dictionary.keys()) if isinstance(
        dictionary, dict) else dictionary
    if limit is not None:
        all_comb = [list(comb) for comb in combinations(keys, len(keys))]
        other_comb = [
            list(comb)
            for comb in chain.from_iterable(
                combinations(keys, r) for r in range(1, limit + 1)
            )
        ]
        return [*all_comb, *other_comb]
    else:
        return [
            list(comb)
            for comb in chain.from_iterable(
                combinations(keys, r) for r in range(1, len(keys) + 1)
            )
        ]


def pad_list(lst, size, fill_value):
    """
    Pad the given list with the specified fill value until it reaches the given size.

    Args:
        lst (list): The list to be padded.
        size (int): The desired size of the padded list.
        fill_value (any): The value to be used for padding.

    Returns:
        list: The padded list.
    """
    if len(lst) >= size:
        return lst
    else:
        padding = [fill_value] * (size - len(lst))
        return lst + padding


def flatten_list(nested_list):
    """
    Flattens a list of nested lists recursively.

    Args:
    nested_list (list): A nested list to be flattened.

    Returns:
    A flattened list.
    """

    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)

    return flattened_list


def sample_non_consecutive(arr: List[int], n: int):
    """Samples n non-consecutive values from a given array."""

    if n <= 0:
        raise ValueError("Sample size must be a positive integer.")

    if n > (len(arr) + 1) // 2:
        raise ValueError(
            f"It's impossible to find a non-consecutive sample of length {n} for an array of size {len(arr)}."
        )

    sorted_arr = sorted(arr)

    def generate_sample(start_index, num_samples_left, last_value):
        """Checks if the current element is non-consecutive with the last value, and if so,
        it proceeds to generate sub-samples by recursively calling itself."""
        if num_samples_left == 0:
            return [[]]

        samples = []
        for i in range(start_index, len(sorted_arr)):
            if sorted_arr[i] - last_value > 1:
                sub_samples = generate_sample(
                    i + 1, num_samples_left - 1, sorted_arr[i]
                )
                for sub_sample in sub_samples:
                    samples.append([sorted_arr[i]] + sub_sample)
        return samples

    # We first obtain all possible samples, then pick one random sample
    all_possible_samples = generate_sample(0, n, float("-inf"))

    if not all_possible_samples:
        raise ValueError(
            "Couldn't find a non-consecutive sample. This should never happen."
        )

    random_sample_index = int(np.random.randint(len(all_possible_samples)))
    return all_possible_samples[random_sample_index]


def safely_parse_list(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    elif isinstance(value, list):
        return value
    else:
        return value
