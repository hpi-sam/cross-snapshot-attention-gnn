import time

from xgboost import XGBClassifier

from src.utils.drawing import draw_importances
from src.utils.results import (clear_result_buckets, compute_result_buckets,
                               create_result_buckets, FakeResult)


def train_and_test(
    x_train,
    y_train,
    x_test,
    y_test,
    x_train_collision_score,
    x_test_collision_score,
    feature_names,
    path="",
    prefix="",
    return_model=False,
    profile=False,
):
    s = time.time()
    model = XGBClassifier().fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    y_test_pred = [round(value) for value in y_test_pred]
    y_train_pred = model.predict(x_train)
    y_train_pred = [round(value) for value in y_train_pred]
    e = time.time()

    results = create_result_buckets()
    compute_result_buckets(
        results=results,
        labels=y_train,
        predictions=y_train_pred,
        collision_scores=x_train_collision_score,
        train=True,
    )
    compute_result_buckets(
        results=results,
        labels=y_test,
        predictions=y_test_pred,
        collision_scores=x_test_collision_score,
    )

    if len(path) > 0:
        try:
            model.get_booster().feature_names = feature_names
            draw_importances(model, f"{path}/{prefix}_feature_importance")
        except RuntimeError:
            print("Error")

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
    y_test_pred = [round(value) for value in y_test_pred]

    results = create_result_buckets()
    compute_result_buckets(
        results=results,
        labels=y_test,
        predictions=y_test_pred,
        collision_scores=x_test_collision_score,
    )

    clear_result_buckets(results=results)
    return results
