import math
from typing import Callable, List, Union
import random

import numpy as np
from scipy import stats


def get_change_in_percent(current, previous) -> float:
    if current == previous:
        return 0.0
    if previous == 0.0:
        return 0.0
    return ((current - previous) / previous) * 100.0


def round_down(num, digits=2, format_e=True):
    if math.isnan(num):
        return "nan"
    result = int(num * (10**digits)) / (10**digits)
    if result == 0.0 and format_e:
        return format(num, "e")
    else:
        return str(result)


def round_up(number: float, decimals: int = 2):
    multiplier = 10 ** decimals
    return math.ceil(number * multiplier) / multiplier


def first_index_above_threshold(arr: float, x: float, default=0):
    result = [i for i, element in enumerate(arr) if element >= x]
    return result[0] if result else default


def is_float(value):
    """
    Returns True if value is a float, and False if value is a string that can be
    formatted into a float.
    """
    try:
        float_value = float(value)
    except (ValueError, TypeError):
        return False
    else:
        return isinstance(float_value, float)


def confidence_interval_to_std_dev(ci: List[float]):
    if len(ci) < 3:
        raise ValueError("Must specify confidence level and ranges")
    return (ci[2] - ci[1]) / (2 * stats.norm.ppf((1 + ci[0]) / 2))


def clamped_sample_fn(
    mean: Union[float, int],
    std: Union[float, int] = None,
    ci: Union[List[float], None] = None,
    bounds: Union[List[float], None] = None,
    fn: Callable[[Union[float, int], Union[float, int]],
                 float] = np.random.normal,
    round_int=True,
) -> Callable[..., Union[float, int]]:
    """
    Returns a function that samples from a given distribution with the given mean and standard deviation,
    and clamps the result to the given bounds. If the bounds are not specified, the result is not clamped.
    """

    if std is None and ci is None:
        raise ValueError("Must specify either std or ci.")
    if std is None:
        std = confidence_interval_to_std_dev(ci)

    def sample(*_):
        sampled = fn(mean, std)
        sampled = round(sampled) if round_int else sampled
        min_val = bounds[0] if bounds is not None and len(bounds) > 0 else None
        max_val = bounds[1] if bounds is not None and len(bounds) > 1 else None
        if min_val is None and max_val is None:
            return sampled
        elif min_val is None:
            return min(sampled, max_val)
        elif max_val is None:
            return max(min_val, sampled)
        return max(min_val, min(sampled, max_val))

    return sample


def add_percent_increase(x, y):
    """
    Returns the sum of x and y% of x, where y represents the percent increase of x.

    Args:
        x (float): The starting value of x.
        y (float): The percent increase of x.

    Returns:
        float: The sum of x and y% of x.
    """
    return x + x * (y / 100.0)


def get_random_seed():
    return random.randint(0, 2**32 - 1)
