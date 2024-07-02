from sklearn.linear_model import LogisticRegression
from src.utils.results import (
    create_result_buckets,
    compute_result_buckets,
    clear_result_buckets,
    FakeResult
)
import time


def train_and_test(
    x_train,
    y_train,
    x_test,
    y_test,
    x_train_collision_score,
    x_test_collision_score,
    return_model=False,
    profile=False,
):
    s = time.time()
    model = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=1000
    ).fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    e = time.time()

    results = create_result_buckets()
    compute_result_buckets(
        results=results,
        labels=y_train,
        predictions=y_train_pred.tolist(),
        collision_scores=x_train_collision_score,
        train=True,
    )
    compute_result_buckets(
        results=results,
        labels=y_test,
        predictions=y_test_pred.tolist(),
        collision_scores=x_test_collision_score,
    )

    clear_result_buckets(results=results)
    if profile:
        results = {"runtime": FakeResult(
            {'runtime': (e - s) * 1000}), **results}
    if return_model:
        return model, results
    return results


def test(
    model,
    x_test,
    y_test,
    x_test_collision_score,
):
    y_test_pred = model.predict(x_test)
    results = create_result_buckets()
    compute_result_buckets(
        results=results,
        labels=y_test,
        predictions=y_test_pred.tolist(),
        collision_scores=x_test_collision_score,
    )
    clear_result_buckets(results=results)
    return results
