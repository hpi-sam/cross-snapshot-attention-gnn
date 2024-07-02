import torch
from torch import nn, optim
from tqdm import tqdm
import numpy as np

from src.utils.numbers import round_down
from src.utils.results import (
    create_result_buckets,
    compute_result_buckets,
    clear_result_buckets,
    FakeResult,
)
from src.utils.profiler import ClassProfiler


class SimpleMLP(nn.Module):
    def __init__(self, feature_size: int, output_size: int) -> None:
        super(SimpleMLP, self).__init__()
        self.layers = create_mlp([feature_size, 16, output_size])
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        if self.layers[-1].out_features == 1:
            return self.sigmoid(self.layers(x))
        return self.softmax(self.layers(x))


def create_mlp(dims, dropout=None):
    layers = [] if dropout is None else [nn.Dropout(dropout)]
    for i, dim in enumerate(dims[:-1]):
        layers.append(nn.Linear(dim, dims[i + 1]))
        if i < len(dims[:-1]) - 1:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def one_hot_encode(tensor, num_classes):
    one_hot = torch.zeros(tensor.size(0), num_classes)
    one_hot.scatter_(1, tensor.view(-1, 1).long(), 1)
    return one_hot


def train_and_test(
    x_train,
    y_train,
    x_test,
    y_test,
    x_train_collision_score,
    x_test_collision_score,
    tqdm_name: str = "",
    log_epochs=False,
    return_model=False,
    profile=False,
):
    num_epochs = 10
    highest_label = int(torch.max(y_train, dim=0)[0].tolist()[0])
    model = SimpleMLP(
        len(x_train[0]), 1 if highest_label == 1 else highest_label + 1)
    profiler = None
    if profile:
        profiler = ClassProfiler()
        model = profiler.profile(model)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = (
        nn.BCELoss(
        ) if model.layers[-1].out_features == 1 else nn.CrossEntropyLoss()
    )
    results = create_result_buckets()

    def get_preds(outputs: torch.Tensor):
        if model.layers[-1].out_features == 1:
            preds = (outputs > 0.5).float()
        else:
            preds = torch.FloatTensor([np.argmax(outputs.tolist())])
        return preds

    def get_labels(label: torch.Tensor):
        if model.layers[-1].out_features == 1:
            out = label
        else:
            out = one_hot_encode(
                label, model.layers[-1].out_features).squeeze()
        return out

    for epoch in (
        tqdm(range(num_epochs), tqdm_name) if len(
            tqdm_name) != 0 else range(num_epochs)
    ):
        model.train()
        train_loss = 0
        train_steps = 0
        train_preds = []
        train_labels = []
        for x, y in zip(x_train, y_train):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, get_labels(y))
            train_loss += loss.item()
            train_steps += 1
            train_preds += get_preds(out).tolist()
            train_labels += y.tolist()
            loss.backward()
            optimizer.step()

        compute_result_buckets(
            results=results,
            labels=train_labels,
            predictions=train_preds,
            collision_scores=x_train_collision_score,
            train=True,
        )
        train_accuracy = results["train"].last[0]

        model.eval()
        eval_loss = 0
        eval_steps = 0
        eval_preds = []
        eval_labels = []
        for x, y in zip(x_test, y_test):
            out = model(x)
            loss = criterion(out, get_labels(y))
            eval_loss += loss.item()
            eval_steps += 1
            eval_preds += get_preds(out).tolist()
            eval_labels += y.tolist()

        compute_result_buckets(
            results=results,
            labels=eval_labels,
            predictions=eval_preds,
            collision_scores=x_test_collision_score,
        )
        test_accuracy = results["test"].last[0]

        if log_epochs:
            print(
                f"Epoch {epoch +1 } Test: {round_down(test_accuracy, 4)}, Train: {round_down(train_accuracy, 4)}, Test Loss: {eval_loss/eval_steps}, Train Loss: {train_loss/train_steps}"
            )

    clear_result_buckets(results)
    if profiler is not None:
        profile = profiler.evaluate()
        profile = sum(profile) if isinstance(profile, list) else profile
        results = {"runtime": FakeResult({'runtime': profile}), **results}

    if return_model:
        return model, results
    return results


def test(
    model,
    x_test,
    y_test,
    x_test_collision_score,
):
    model.eval()

    def get_preds(outputs: torch.Tensor):
        if model.layers[-1].out_features == 1:
            preds = (outputs > 0.5).float()
        else:
            preds = torch.FloatTensor([np.argmax(outputs.tolist())])
        return preds

    eval_preds = []
    eval_labels = []
    for x, y in zip(x_test, y_test):
        out = model(x)
        eval_preds += get_preds(out).tolist()
        eval_labels += y.tolist()

    results = create_result_buckets()
    compute_result_buckets(
        results=results,
        labels=eval_labels,
        predictions=eval_preds,
        collision_scores=x_test_collision_score,
    )

    clear_result_buckets(results=results)
    return results
