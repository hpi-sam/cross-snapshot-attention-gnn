import time
import functools
import types
import torch.nn as nn


class ClassProfiler:
    """Helper to profile the execution time of methods of a class.

    Re-assigns the methods of a given class to wrappers that measure the execution time.
    Afterward, whenever the function is called, execution time is measured and added to the total execution time.
    Calling evaluate yields an overview of the execution times of the methods.
    """

    def __init__(self, keywords=None, levels=2):
        self.execution_times = {}
        self.total_time = 0
        self.levels = levels
        self.keywords = keywords
        self.class_name = None

    def _time_function(self, func, name, level, prefix=""):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            elapsed_time = (end_time - start_time) * 1000
            func_name = f"{prefix}{name}"

            if func_name not in self.execution_times:
                self.execution_times[func_name] = 0
            self.execution_times[func_name] += elapsed_time
            if level == 0:
                self.total_time += elapsed_time

            return result

        return wrapper

    def _profile_attributes(self, obj, prefix="", level=0):
        if not hasattr(obj, "__dict__") or level > self.levels:
            return None

        for name in dir(obj):
            try:
                attr = getattr(obj, name)
                if isinstance(attr, types.MethodType):
                    try:
                        setattr(
                            obj, name, self._time_function(
                                attr, name, level, prefix)
                        )
                    except Exception:
                        continue
                elif isinstance(attr, (list, nn.ModuleList)):
                    for i, item in enumerate(attr):
                        if isinstance(item, object):
                            self._profile_attributes(
                                item, f"{prefix}{name}_{i}.", level + 1
                            )
                elif isinstance(attr, object):
                    self._profile_attributes(
                        attr, f"{prefix}{name}.", level + 1)

            except Exception:
                continue

    def profile(self, original_instance):
        self.class_name = original_instance.__class__.__name__
        self._profile_attributes(original_instance)
        return original_instance

    def evaluate(self):
        header_format = "{:<75}{:<20}{:<10}"
        row_format = "{:<75}{:<20.2f}{:<10.2f}"

        print(f"--- Profile for {self.class_name} ---")
        print(header_format.format("Function Name", "Execution Time", "Percentage"))
        for func_name, exec_time in dict(
            sorted(self.execution_times.items(), key=lambda x: -x[1])
        ).items():
            percentage = (exec_time / self.total_time) * 100
            if percentage > 0.01:
                print(row_format.format(func_name, exec_time, percentage))
        print(f"--- End Profile for {self.class_name} ---")

        if self.keywords is not None:
            keyword_times = []
            for keyword_list in self.keywords:
                keyword_time = 0
                for func_name, exec_time in self.execution_times.items():
                    if all(keyword in func_name for keyword in keyword_list):
                        keyword_time += exec_time
                keyword_times.append(keyword_time)
            return keyword_times
        return self.total_time
