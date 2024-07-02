import os
import random

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.loader import DataLoader

from src.augmentations.augmentation import \
    create_augmented_views_from_graph_batches
from src.datasets.dataset import Dataset
from src.datasets.random.pretext import PreTextDataset
from src.datasets.transform import (PropagationGraphTransform,
                                    TemporalSnapshotListTransform)
from src.models.cross_snapshot_attention import load_csa_net
from src.pretext.contrastive.model import ContrastivePreTextModel
from src.training.losses import ContrastiveLoss
from src.training.storing import load_model, save_model
from src.utils.results import FakeResult


class ContrastivePreTextTrainer:
    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        transform: PropagationGraphTransform = TemporalSnapshotListTransform,
        training_share=1,
        neg_pos_ratio=1,
        batch_size=64,
        learning_rate=0.01,
        distributional_model=False,
        augmentations=None,
        force_feats=None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.training_share = training_share
        self.distributional_model = distributional_model
        self.augmentations = augmentations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.neg_pos_ratio = neg_pos_ratio

        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.criterion = ContrastiveLoss()
        self.device = "cpu"

        train_dataset, test_dataset = dataset.transform(
            transform=transform, feats=force_feats)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        # Sample test set to calculate test loss,
        # seeded to ensure that each model runs on same test set
        random.seed(42)
        self.test_data = []
        for data in self.test_loader:
            x = data.to(self.device)
            x1, x2 = create_augmented_views_from_graph_batches(
                x, keep_first_unchanged=True)
            self.test_data.append((x, x1, x2))
        random.seed(None)

    def train(self, epochs: int):
        train_loss = 0
        train_step = 0
        test_loss = 0
        test_step = 0
        train_losses = []
        num_negative_samples = 1 * self.neg_pos_ratio  # in batches

        # NOTE: hack for training share
        train_share_data = []
        for i, data in enumerate(self.train_loader):
            if (
                self.training_share > 0
                and i < len(self.train_loader) * self.training_share
            ):
                train_share_data.append(data)

        for epoch in range(epochs):
            self.model.train()
            random.shuffle(train_share_data)

            epoch_loss = 0
            for i, data in enumerate(train_share_data):
                x = data.to(self.device)
                x1, x2 = create_augmented_views_from_graph_batches(
                    x, self.augmentations
                )

                positive_i = self.get_output(self.model(x1))
                positive_j = self.get_output(self.model(x2))
                other_samples = [
                    train_share_data[j] for j in range(len(train_share_data)) if j != i
                ]
                negative_samples = random.sample(
                    other_samples, num_negative_samples)
                negative_samples = torch.cat(
                    [self.get_output(self.model(x)) for x in negative_samples], dim=0
                )
                loss = self.criterion(positive_i, positive_j, negative_samples)

                epoch_loss += loss.item()
                train_loss += loss.item()
                train_step += 1

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            train_losses.append(epoch_loss / len(train_share_data))
            self.model.eval()
            num_correct = 0
            num_total = 0
            max_accuracy = 0
            for i, data in enumerate(self.test_data):
                _, x1, x2 = data
                positive_i = self.get_output(self.model(x1))
                positive_j = self.get_output(self.model(x2))
                other_samples = [
                    self.test_data[j] for j in range(len(self.test_data)) if j != i and self.test_data[j][0].batch.max() >= self.test_data[i][0].batch.max()
                ]
                negative_samples = random.sample(
                    other_samples, num_negative_samples)
                negative_samples = torch.cat(
                    [self.get_output(self.model(x[0])) for x in negative_samples], dim=0
                )[:positive_i.shape[0], :]
                loss = self.criterion(positive_i, positive_j, negative_samples)

                # Compute cosine similarity for each pair
                pos_similarity = F.cosine_similarity(
                    positive_i, positive_j, dim=-1)
                neg_similarity = F.cosine_similarity(
                    positive_i, negative_samples, dim=-1)

                is_greater = pos_similarity > neg_similarity
                num_correct += is_greater.sum().item()
                num_total += is_greater.shape[0]

                test_loss += loss.item()
                test_step += 1

            accuracy = num_correct / num_total
            if accuracy > max_accuracy:
                max_accuracy = accuracy

            print(
                f"Epoch: {epoch + 1}, Accuracy: {accuracy:.4f}, Test Loss: {test_loss / (test_step if test_step > 0 else 1):.4f}, Loss: {train_loss / (train_step if train_step > 0 else 1):.4f}"
            )

        return {"test": FakeResult({"accuracy": max_accuracy, "train_losses": train_losses})}

    def get_output(self, out):
        if self.distributional_model:
            return out[0]
        return out


def load_from_contrastive_pretext_model(
    model_name: str,
    pretext_dataset=None,
    force_retrain=False,
    force_load=False,
    encoder_only=True,
    freeze=True,
    train_epochs=10,
    train_feats=1,
    **encoder_kwargs,
):
    dataset = (
        pretext_dataset
        if pretext_dataset is not None
        else PreTextDataset(log_progress=True)
    )
    path = (
        f"src/pretext/contrastive/models/model_{dataset.abbreviation}_{model_name}.pt"
    )

    if not os.path.exists(path) and force_load:
        raise ValueError(f"Model {model_name} can't be loaded.")

    if os.path.exists(path) and not force_retrain:
        if encoder_only:
            return load_csa_net(path=path, freeze=freeze, **encoder_kwargs)
        return load_model(
            path=path,
            model_class=ContrastivePreTextModel,
            **encoder_kwargs,
        )

    print(f"Training model {model_name}...")
    model = ContrastivePreTextModel(
        **encoder_kwargs,
    )
    distributional_model = encoder_kwargs.get("distributional", False)
    trainer = ContrastivePreTextTrainer(
        model=model, dataset=dataset, distributional_model=distributional_model, force_feats=train_feats
    )
    trainer.train(epochs=train_epochs)
    if encoder_only:
        save_model(path=path, model=model.encoder)
        return load_csa_net(path=path, freeze=freeze, **encoder_kwargs)
    save_model(path=path, model=model)
    return model
