import torch
from torch_geometric.data import InMemoryDataset
from src.utils.path import delete_file, path_exists
from src.datasets.transform import PropagationGraphTransform
from pathlib import Path
import gzip


class TransformedDatasetFactory:
    """A factory for creating transformed datasets in PyG format."""

    def __init__(
        self, dataset, transform: PropagationGraphTransform, prefix="", suffix=""
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.file_name = self.get_file_name(prefix, suffix)

    def get_file_name(self, prefix="", suffix=""):
        Path(
            f"tmp/{self.dataset.name}/processed").mkdir(parents=True, exist_ok=True)
        if len(prefix) == 0:
            return f"data_{self.transform.name}{suffix}.pt"

        Path(f"tmp/{self.dataset.name}/processed/{prefix}").mkdir(
            parents=True, exist_ok=True
        )
        return f"{prefix}/data_{self.transform.name}{suffix}.pt"

    def has_data(self):
        return path_exists(f"tmp/{self.dataset.name}/processed/{self.file_name}")

    def build(self, force=False):
        other = self
        if force:
            path = f"tmp/{self.dataset.name}/processed/{self.file_name}"
            delete_file(path)

        class CustomInMemoryDataset(InMemoryDataset):
            def __init__(
                self, root, transform=None, pre_transform=None, pre_filter=None
            ):
                super().__init__(root, transform, pre_transform, pre_filter)

                with gzip.open(self.processed_paths[0], "rb") as f:
                    collate, self.split = torch.load(f)

                self.data, self.slices = collate

            @property
            def raw_file_names(self):
                return []

            @property
            def processed_file_names(self):
                return [other.file_name]

            def download(self):
                pass

            def process(self):
                max_t = other.dataset.get_max_t()
                train_data = [
                    (other.transform.transform(sample.propagation, max_t), sample)
                    for sample in other.dataset.train
                ]
                test_data = [
                    (other.transform.transform(sample.propagation, max_t), sample)
                    for sample in other.dataset.test
                ]
                split = len(train_data)

                data_list = []
                for data, sample in [*train_data, *test_data]:
                    data.y = sample.label
                    data.max_collision_score = (
                        sample.max_collision_score
                        if sample.max_collision_score is not None
                        else 0
                    )
                    data_list.append(data)

                with gzip.open(self.processed_paths[0], "wb") as f:
                    torch.save((self.collate(data_list), split), f)

        return CustomInMemoryDataset(root=f"tmp/{self.dataset.name}")
