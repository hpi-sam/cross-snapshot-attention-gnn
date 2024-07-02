import random

from torch import nn, optim
from torch_geometric.loader import DataLoader

from src.datasets.dataset import Dataset
from src.datasets.transform import (PropagationGraphTransform,
                                    TemporalGraphAttributesTransform)
from src.training.losses import KLDLoss
from src.utils.profiler import ClassProfiler
from src.utils.results import (clear_result_buckets, compute_result_buckets,
                               create_result_buckets, FakeResult)


class GNNTrainer:
    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        transform: PropagationGraphTransform = TemporalGraphAttributesTransform,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        criterion=None,
        optimizer=None,
        scheduler=None,
        profile=False,
        profile_keywords=None,
        use_buckets=True,
        do_backward=True,
        training_share=None,
        distributional_model=False,
        force_feats=None,
    ) -> None:
        self.model = model
        self.profiler = None
        self.use_buckets = use_buckets
        self.do_backward = do_backward
        self.training_share = training_share
        self.distributional_model = distributional_model

        if profile:
            self.profiler = ClassProfiler(keywords=profile_keywords)
            self.model = self.profiler.profile(model)

        self.dataset = dataset
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5
        ) if optimizer is None else optimizer
        self.criterion = (
            criterion
            if criterion is not None
            else (
                nn.CrossEntropyLoss()
                if not distributional_model
                else KLDLoss(base_loss=nn.CrossEntropyLoss())
            )
        )
        self.scheduler = scheduler
        self.device = "cpu"

        train_dataset, test_dataset = dataset.transform(
            transform=transform, feats=force_feats)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

    def train(self, epochs: int):
        sum_loss = 0
        step = 0
        if self.use_buckets:
            results = create_result_buckets()
        else:
            results = create_result_buckets(num_buckets=0)

        # NOTE: hack for training share
        train_share_data = []
        if self.training_share is not None:
            for i, data in enumerate(self.train_loader):
                if (
                    self.training_share > 0
                    and i < len(self.train_loader) * self.training_share
                ):
                    train_share_data.append(data)

        for epoch in range(epochs):
            self.model.train()
            collision_scores = []
            inputs = []
            labels = []
            predictions = []
            random.shuffle(train_share_data)
            for i, data in enumerate(
                self.train_loader if self.training_share is None else train_share_data
            ):
                inputs.append(data)
                x, y = self.get_input(data.to(self.device), idx=i)
                out = self.model(x)
                loss = self.criterion(*self.get_criterion_input(out, y, x))
                if self.distributional_model:
                    out = out[0]
                sum_loss += loss.item()
                step += 1
                preds = self.get_preds(out)
                collision_scores += (
                    x.max_collision_score.detach().cpu().numpy().tolist()
                )
                predictions += preds.detach().cpu().numpy().tolist()
                labels += y.detach().cpu().numpy().tolist()
                if self.do_backward:
                    loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()

            self.score(
                results=results,
                predictions=predictions,
                labels=labels,
                collision_scores=collision_scores,
                data=inputs,
            )
            self.model.eval()
            collision_scores = []
            inputs = []
            labels = []
            predictions = []
            for i, data in enumerate(self.test_loader):
                inputs.append(data)
                x, y = self.get_input(data.to(self.device), idx=i, train=False)
                out = self.model(x)
                if self.distributional_model:
                    out = out[0]
                preds = self.get_preds(out)
                collision_scores += (
                    x.max_collision_score.detach().cpu().numpy().tolist()
                )
                predictions += preds.detach().cpu().numpy().tolist()
                labels += y.detach().cpu().numpy().tolist()

            self.score(
                results=results,
                predictions=predictions,
                labels=labels,
                collision_scores=collision_scores,
                data=inputs,
                train=False,
            )
            test_accuracy = results["test"].last[0]
            test_precision = results["test"].last[1]
            test_recall = results["test"].last[2]
            test_f1 = results["test"].last[3]

            print(
                f"Epoch: {epoch + 1}, Acc: {test_accuracy:.4f}, Prec: {test_precision:.4f}, Rec: {test_recall:.4f}, F1: {test_f1:.4f}, Loss: {sum_loss / (step if step > 0 else 1):.4f}"
            )

        clear_result_buckets(results=results)
        if self.profiler is not None:
            profile = self.profiler.evaluate()
            profile = sum(profile) if isinstance(profile, list) else profile
            results = {"runtime": FakeResult(
                {'runtime': profile}), **results}

        return results

    def get_input(self, data, idx=None, train=True):
        return data, data.y

    def get_criterion_input(self, out, y, x):
        if self.distributional_model:
            # out[0] is embedding, 1 is mean, 2 is logvar
            return out[0], y, out[1], out[2]
        return out, y

    def get_preds(self, out):
        return out.argmax(dim=1)

    def score(
        self,
        results: dict,
        predictions,
        labels,
        collision_scores,
        data=None,
        train=True,
    ):
        compute_result_buckets(
            results=results,
            labels=labels,
            predictions=predictions,
            collision_scores=collision_scores,
            train=train,
        )
