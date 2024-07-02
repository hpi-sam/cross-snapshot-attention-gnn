import gzip
import random
import time
from pathlib import Path
from typing import Callable, List, Union

import dill
from tqdm import tqdm

from src.datasets.transform import (PropagationGraphTransform,
                                    PropagationMetricsGraphTransform,
                                    TemporalGraphAttributesTransform,
                                    TemporalSnapshotListTransform)
from src.datasets.transform_factory import TransformedDatasetFactory
from src.generation.propagation.propagation import (Propagation,
                                                    PropagationConfig)
from src.perturbations.perturbation import Perturbation
from src.utils.objects import sample_non_consecutive
from src.utils.path import path_exists


class Label:
    """Generic label interface."""

    def __init__(
        self,
        name: str,
        config: Union[PropagationConfig,
                      Callable[..., PropagationConfig], None] = None,
    ) -> None:
        self.name = name
        self.config = config

    def instantiate(self, *args):
        return self.config(*args) if callable(self.config) else self.config


class Sample:
    """Generic sample interface for propagations."""

    def __init__(
        self,
        propagation: Propagation,
        label: int,
        max_collision_score: Union[float, None] = None,
    ):
        self.propagation = propagation
        self.label = label
        self.max_collision_score = max_collision_score


class Dataset:
    """Generic dataset interface."""

    def __init__(
        self,
        name: str,
        labels: List[Label],
        abbreviation: str = None,
        recreate: bool = False,
        prevent_transform=False,
        mask_method=None,
        mask_amount=None,
        mask_train=False,
        mask_test=True,
        skeleton=False,
    ) -> None:
        self.name = name
        self.abbreviation = abbreviation if abbreviation else name
        self.labels = labels
        self.train: List[Sample] = []
        self.test: List[Sample] = []
        self.size = 0
        self.loaded = False
        self.prevent_transform = prevent_transform

        self.mask_method = mask_method
        self.mask_amount = mask_amount
        self.mask_train = mask_train
        self.mask_test = mask_test

        # don't do anything if we only want the skeleton
        if skeleton:
            return

        if recreate:
            self.create()
        else:
            self.load()

    def sample(self, train=True) -> List[Sample]:
        raise NotImplementedError("Sample function not implemented.")

    def get_max_t(self):
        return max(
            [
                len(sample.propagation.snapshots)
                for sample in [
                    *self.train,
                    *self.test,
                ]
            ]
        )

    def get_max_num_nodes(self):
        return max(
            [
                len(snapshot.nodes())
                for sample in [
                    *self.train,
                    *self.test,
                ]
                for snapshot in sample.propagation.behavior_graph.snapshots
            ]
        )

    def after_sample(self):
        """Called after samples have been created. Can be used to compute additional properties of the samples."""
        # we want to compute the collision threshold for each sample in the test set to estimate the "hardness/uniqueness" of the sample
        propagations = [sample.propagation for sample in self.train]
        for sample in tqdm(self.test, "After sample"):
            _, max_threshold = sample.propagation.matches_most(
                others=propagations)
            sample.max_collision_score = max_threshold

    def create(self, masking_only=False):
        """Creates and persists the dataset."""
        print(
            f"Creating {self.name} dataset{' masking' if masking_only else ''}...")
        start_time = time.time()
        if not masking_only:
            self.train = self.sample()
            self.test = self.sample(train=False)
            self.after_sample()
            self.persist(mask=False)
            if not self.prevent_transform:
                self.transform(mask=False)
        if self.mask_method is not None and self.mask_amount is not None:
            self.mask(
                amount=self.mask_amount,
                mask=self.mask_method,
                train=self.mask_train,
                test=self.mask_test,
            )
            # NOTE: We don't recompute the after sample here, as we want to keep the original collision scores, as collisions are a mediator for masking
            # self.after_sample()
            self.persist()
            if not self.prevent_transform:
                self.transform()
        self.size = len(self.train) + len(self.test)
        end_time = time.time()
        duration = end_time - start_time
        print(
            f"Created {self.name} dataset with {self.size} samples in {int(duration)}s."
        )

    def transform(self, transform: PropagationGraphTransform = None, mask=True, feats=None):
        """Transforms the dataset into a given format.

        If none given, pre-transforms the dataset into all available formats.
        """
        if transform is None:
            for transform in [
                TemporalGraphAttributesTransform,
                TemporalSnapshotListTransform,
                PropagationMetricsGraphTransform
            ]:
                factory = TransformedDatasetFactory(
                    dataset=self,
                    transform=transform(),
                    prefix=self.get_mask_file_prefix() if mask else "",
                    suffix=self.get_mask_file_suffix() if mask else "",
                )
                factory.build(force=True)
            return None
        factory = TransformedDatasetFactory(
            dataset=self,
            transform=transform(),
            prefix=self.get_mask_file_prefix(),
            suffix=self.get_mask_file_suffix(),
        )
        data = factory.build()
        # only use a subset of the features if given
        if feats is not None:
            if transform is TemporalSnapshotListTransform:
                for sample in data:
                    for snapshot in sample.snapshots:
                        snapshot.x = snapshot.x[:, :feats]
            else:
                for sample in data:
                    sample.x = sample.x[:, :feats]

        return data[:data.split], data[data.split:]

    def get_mask_file_prefix(self):
        return (
            f"masked_{self.mask_method}"
            if self.mask_method is not None and self.mask_amount is not None
            else ""
        )

    def get_mask_file_suffix(self):
        return (
            f"_{self.mask_amount}"
            if self.mask_method is not None and self.mask_amount is not None
            else ""
        )

    def get_file_names(self, mask=True):
        path = f"tmp/{self.name}/raw"
        mask_file_prefix = self.get_mask_file_prefix() if mask else ""
        path = f"{path}/{mask_file_prefix}" if len(
            mask_file_prefix) > 0 else path
        Path(path).mkdir(parents=True, exist_ok=True)
        mask_file_suffix = self.get_mask_file_suffix() if mask else ""
        return f"{path}/train{mask_file_suffix}.pt", f"{path}/test{mask_file_suffix}.pt"

    def persist(self, mask=True):
        train_file, test_file = self.get_file_names(mask)
        self.save_samples(self.train, train_file)
        self.save_samples(self.test, test_file)

    def load(self):
        """Loads the dataset from disk."""
        base_train_file, base_test_file = self.get_file_names(mask=False)

        if not path_exists(base_train_file) or not path_exists(base_test_file):
            self.create()
            return

        print(f"Loading {self.name} dataset...")
        start_time = time.time()
        self.train = self.load_samples(base_train_file)
        self.test = self.load_samples(base_test_file)

        if self.mask_method is not None and self.mask_amount is not None:
            masked_train_file, masked_test_file = self.get_file_names()
            if not path_exists(masked_train_file) or not path_exists(masked_test_file):
                self.create(masking_only=True)
                return
            self.train = self.load_samples(masked_train_file)
            self.test = self.load_samples(masked_test_file)

        self.size = len(self.train) + len(self.test)
        end_time = time.time()
        duration = end_time - start_time
        self.loaded = True
        print(
            f"Loaded {self.name} dataset with {self.size} samples in {int(duration)}s."
        )

    def save_samples(self, samples: List[Sample], path: str):
        with gzip.open(path, "wb") as f:
            serialized_samples = dill.dumps(samples)
            f.write(serialized_samples)

    def load_samples(self, path: str):
        with gzip.open(path, "rb") as f:
            serialized_samples = f.read()
            return dill.loads(serialized_samples)

    def sample_mask_amount(self, amount: int):
        # if -1 given, sample a random amount from 1 to max_t - 1
        if amount == -1:
            max_t = self.get_max_t()
            return random.choice(range(1, max_t - 1))
        return amount

    def mask_spots(self, samples: List[Sample], amount: int):
        for sample in tqdm(samples, f"Mask {amount} snapshots using spots method"):
            max_t = len(sample.propagation.snapshots)
            masked_amount = self.sample_mask_amount(amount)
            # shift by one as we need the first snapshot to be unmasked
            masked_t = sample_non_consecutive(range(1, max_t), masked_amount)
            for t in masked_t:
                sample.propagation.mask_snapshot(t)

    def mask_gaps(self, samples: List[Sample], amount: int):
        for sample in tqdm(samples, f"Mask {amount} snapshots using gaps method"):
            max_t = len(sample.propagation.snapshots)
            masked_amount = self.sample_mask_amount(amount)
            # start == 1 would be the same as mask_start
            # start == max_t - amount would be the same as mask_end
            # thus, we exclude those two cases
            start = random.choice(range(2, max_t - masked_amount - 1))
            for t in range(start, start + masked_amount):
                sample.propagation.mask_snapshot(t)

    def mask_start(self, samples: List[Sample], amount: int):
        for sample in tqdm(samples, f"Mask {amount} snapshots using start method"):
            masked_amount = self.sample_mask_amount(amount)
            # shift by one as we need the first snapshot to be unmasked
            for t in range(1, masked_amount + 1):
                sample.propagation.mask_snapshot(t)

    def mask_end(self, samples: List[Sample], amount: int):
        for sample in tqdm(samples, f"Mask {amount} snapshots using end method"):
            masked_amount = self.sample_mask_amount(amount)
            for t in range(
                len(sample.propagation.snapshots) - masked_amount,
                len(sample.propagation.snapshots),
            ):
                sample.propagation.mask_snapshot(t)

    def mask_nodes(self, samples: List[Sample], amount: float):
        for sample in tqdm(samples, f"Mask {amount} nodes"):
            for t, _ in enumerate(sample.propagation.snapshots):
                if t == 0:
                    continue
                masked_amount = int(
                    sample.propagation.behavior_graph.snapshots[t].number_of_nodes() * amount)
                masked_nodes = random.sample(
                    list(sample.propagation.behavior_graph.snapshots[t].nodes()), masked_amount)
                for node in masked_nodes:
                    sample.propagation.mask_node(node, t)

    def mask(
        self, amount: int, mask="spots", train=False, test=True, debug_smoke_test=False
    ):
        """Masks the given amount of snapshots in the dataset.

        Args:
            amount (int): Amount to mask. A value of -1 will sample a random amount using the max_t of the dataset.
            mask (str, optional): Masking method to be used. Defaults to "spots".
            train (bool, optional): Mask the train set. Defaults to False.
            test (bool, optional): Mask the test set. Defaults to True.
            debug_smoke_test (bool, optional): Small approximate smoke test to check if masking worked. Defaults to False.
        """
        if len(self.train) == 0 or len(self.test) == 0:
            raise RuntimeError("No samples to mask.")

        # approximate smoke test for masking that checks if the diffs between snapshots changed
        if debug_smoke_test:
            before_diffs = []
            for sample in self.train:
                for t in range(1, len(sample.propagation.snapshots)):
                    before_diffs.append(
                        len(sample.propagation.diff(t - 1, t)[0]))

        if mask == "spots":
            if train:
                self.mask_spots(self.train, amount)
            if test:
                self.mask_spots(self.test, amount)
        elif mask == "gaps":
            if train:
                self.mask_gaps(self.train, amount)
            if test:
                self.mask_gaps(self.test, amount)
        elif mask == "start":
            if train:
                self.mask_start(self.train, amount)
            if test:
                self.mask_start(self.test, amount)
        elif mask == "end":
            if train:
                self.mask_end(self.train, amount)
            if test:
                self.mask_end(self.test, amount)
        elif mask == "nodes":
            if train:
                self.mask_nodes(self.train, amount)
            if test:
                self.mask_nodes(self.test, amount)
        else:
            raise ValueError(f"Unknown mask method: {mask}")

        if debug_smoke_test:
            after_diffs = []
            for sample in self.train:
                for t in range(1, len(sample.propagation.snapshots)):
                    after_diffs.append(
                        len(sample.propagation.diff(t - 1, t)[0]))
            assert (
                before_diffs != after_diffs
            ), f"Failed! The masking method {mask} did not change the diffs between snapshots."

    def perturb(
        self, perturbation: Perturbation, train=False, test=True, persist=True, **perturbation_kwargs
    ):
        """Perturb the dataset using the given perturbation.

        Args:
            perturbation (Perturbation): Perturbation to be applied.
        """
        if train:
            for sample in tqdm(self.train, 'Perturb train set'):
                perturbed = perturbation(
                    propagation=sample.propagation, **perturbation_kwargs
                )
                sample.propagation = perturbed.propagation
        if test:
            for sample in tqdm(self.test, 'Perturb test set'):
                perturbed = perturbation(
                    propagation=sample.propagation, **perturbation_kwargs
                )
                sample.propagation = perturbed.propagation
        if persist:
            self.persist()
            self.transform()
