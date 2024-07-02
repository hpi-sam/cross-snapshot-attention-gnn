import os
import random
from typing import List

import torch
from torch_geometric.data import Data

from src.datasets.dataset import Sample
from src.datasets.random.pretext import PreTextDataset
from src.datasets.transform import TemporalSnapshotListTransform
from src.generation.propagation.propagation import Propagation
from src.models.cross_snapshot_attention import load_csa_net
from src.pretext.predictive.model import PredictivePreTextModel
from src.training.losses import KLDLoss, WeightedWithFilteredPriorsBCELoss
from src.training.storing import load_model, save_model
from src.training.trainer import GNNTrainer
from src.utils.objects import flatten_list
from src.utils.results import compute_result_buckets


class PredictivePreTextTrainer(GNNTrainer):
    def __init__(self, **kwargs):
        self.max_t = kwargs["dataset"].max_t

        self.available_train_prompts = self.compute_available_prompt_sizes_per_sample(
            kwargs["dataset"].train)
        self.available_test_prompts = self.compute_available_prompt_sizes_per_sample(
            kwargs["dataset"].test)
        self.imbalance = self.compute_imbalance(
            samples=kwargs["dataset"].train, prompt_sizes=self.available_train_prompts)
        self.pos_weight = int(1/self.imbalance)

        print(
            f"Imbalance of {self.imbalance:.4f}, using weight of {self.pos_weight} for positive labels")

        # Sample test prompt sizes to test on same prompts between epochs,
        # seeded to ensure that each model runs on same test set
        random.seed(42)
        self.test_prompt_sizes = [
            self.sample_prompt_size(
                available=self.available_test_prompts, idx=i)
            for i, sample in enumerate(kwargs["dataset"].test)
        ]
        random.seed(None)
        # just init
        self.train_prompt_sizes = [
            self.sample_prompt_size(
                available=self.available_train_prompts, idx=i)
            for i, sample in enumerate(kwargs["dataset"].train)
        ]

        weighted_loss = WeightedWithFilteredPriorsBCELoss(
            pos_weight=self.pos_weight,
        )
        if "distributional_model" in kwargs and kwargs["distributional_model"]:
            criterion = KLDLoss(base_loss=weighted_loss)
        else:
            criterion = weighted_loss

        # NOTE: Starts to overfit pretty quickly, so we use a low learning rate
        learning_rate = 0.001

        # no batching, as we want to determine prompt size individually per sample
        # no buckets, as not needed for this task
        super().__init__(
            **kwargs,
            transform=TemporalSnapshotListTransform,
            learning_rate=learning_rate,
            criterion=criterion,
            batch_size=1,
            use_buckets=False,
        )

    def compute_available_prompt_sizes_per_sample(self, samples: List[Sample]):
        prompt_sizes = []
        for i, sample in enumerate(samples):
            prompt_sizes.append([])
            for t in range(2, len(sample.propagation.snapshots) - 1):
                node_diff, _ = sample.propagation.diff(t - 1, t)
                if len(node_diff) > 0:
                    prompt_sizes[i].append(t)
        return prompt_sizes

    def compute_imbalance(self, samples: List[Sample], prompt_sizes: List[List[int]]):
        total_predictions_to_make = 0
        # the number of cases where nodes change from 0 to 1
        num_change_case = 0

        def get_all_covered_nodes(propagation: Propagation, t: int):
            return [node for node in propagation.behavior_graph.snapshots[t].nodes() if propagation.node_is_covered(node, t)]

        def get_all_uncovered_nodes(propagation: Propagation, t: int):
            return [node for node in propagation.behavior_graph.snapshots[t].nodes() if not propagation.node_is_covered(node, t)]

        for i, sample in enumerate(samples):
            for t in prompt_sizes[i]:
                uncovered_nodes = get_all_uncovered_nodes(
                    sample.propagation, t-1)
                new_covered_nodes = get_all_covered_nodes(
                    sample.propagation, t)
                num_change_case += len(
                    set(uncovered_nodes).intersection(set(new_covered_nodes)))
                total_predictions_to_make += len(uncovered_nodes)

        return num_change_case / total_predictions_to_make

    def sample_prompt_size(self, available: List[List[int]], idx: int):
        choices = available[idx]
        if len(choices) == 0:
            choices = list(range(2, self.max_t - 1))
        return random.choice(choices)

    def get_label_from_snapshot(self, snapshot: Data):
        # Use x attribute of the sample to retrieve the propagation attribute for each node
        return snapshot.x[:, 0].detach().cpu().numpy().tolist()

    def get_input(self, data, idx=None, train=True):
        if train:
            prompt_size = self.sample_prompt_size(
                available=self.available_train_prompts, idx=idx)
            self.train_prompt_sizes[idx] = prompt_size
        else:
            prompt_size = self.test_prompt_sizes[idx]

        labels = []
        new_data = data.clone()
        for j, sample_snapshots in enumerate(new_data.snapshots):
            new_data.snapshots[j] = sample_snapshots[:prompt_size]
            new_data.current_snapshot = sample_snapshots[prompt_size - 1]
            new_data.next_snapshot = sample_snapshots[prompt_size]
            label = self.get_label_from_snapshot(sample_snapshots[prompt_size])
            labels.append(label)

        return new_data, torch.tensor(labels, dtype=torch.float32)

    def get_preds(self, out):
        return (out > 0.5).float()

    def get_criterion_input(self, out, y, x):
        prior_attributes = x.current_snapshot.x[:, 0]
        if self.distributional_model:
            # out[0] is embedding, 1 is mean, 2 is logvar
            return out[0], y, prior_attributes, out[1], out[2]
        return out, y, prior_attributes

    def get_previous_snapshot_labels(self, data_loader, prompt_sizes):
        previous_labels = []
        for data, prompt_size in zip(data_loader, prompt_sizes):
            for sample_snapshots in data.snapshots:
                label = self.get_label_from_snapshot(
                    sample_snapshots[prompt_size - 1])
                previous_labels.append(label)
        return previous_labels

    def score(
        self, results, predictions, labels, collision_scores, data=None, train=True
    ):
        if train:
            previous_labels = self.get_previous_snapshot_labels(
                data, self.train_prompt_sizes
            )
        else:
            previous_labels = self.get_previous_snapshot_labels(
                data, self.test_prompt_sizes
            )

        previous_labels = flatten_list(previous_labels)
        next_labels = flatten_list(labels)
        next_predictions = flatten_list(predictions)

        # filter the previously uncovered nodes, as the already covered nodes won't change anyways
        relevant_items = [i for i, val in enumerate(
            previous_labels) if val == 0]

        next_labels = [next_labels[i] for i in relevant_items]
        next_predictions = [next_predictions[i] for i in relevant_items]

        compute_result_buckets(
            results=results,
            labels=next_labels,
            predictions=next_predictions,
            collision_scores=None,
            train=train,
        )


def load_from_predictive_pretext_model(
    model_name: str,
    model_version: str,
    pretext_dataset=None,
    pretext_dropout=None,
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
    path = f"src/pretext/predictive/models/model_{dataset.abbreviation}_{model_name}.pt"

    if not os.path.exists(path) and force_load:
        raise ValueError(f"Model {model_name} can't be loaded.")

    if os.path.exists(path) and not force_retrain:
        if encoder_only:
            return load_csa_net(path=path, freeze=freeze, **encoder_kwargs)
        return load_model(
            path=path,
            model_class=PredictivePreTextModel,
            version=model_version,
            max_num_nodes=dataset.get_max_num_nodes(),
            **({"pretext_dropout": pretext_dropout} if pretext_dropout is not None else {}),
            **encoder_kwargs,
        )

    print(f"Training model {model_name}...")
    model = PredictivePreTextModel(
        version=model_version,
        max_num_nodes=dataset.get_max_num_nodes(),
        **encoder_kwargs,
    )
    distributional_model = encoder_kwargs.get("distributional", False)
    trainer = PredictivePreTextTrainer(
        model=model, dataset=dataset, distributional_model=distributional_model, force_feats=train_feats
    )
    trainer.train(epochs=train_epochs)
    if encoder_only:
        save_model(path=path, model=model.encoder)
        return load_csa_net(path=path, freeze=freeze, **encoder_kwargs)
    save_model(path=path, model=model)
    return load_model(
        path=path,
        model_class=PredictivePreTextModel,
        version=model_version,
        max_num_nodes=dataset.get_max_num_nodes(),
        **({"pretext_dropout": pretext_dropout} if pretext_dropout is not None else {}),
        **encoder_kwargs,
    )
