from torch_geometric.loader import DataLoader
from tqdm import tqdm

from experiments.utils import default_runner
from src.datasets.crypto.bitcoin_block_propagation import \
    BitcoinBlockPropagationDataset
from src.datasets.dataset import Dataset
from src.datasets.cybersecurity.ddos import DDoSDataset
from src.datasets.epidemics.covid19 import Covid19Dataset
from src.datasets.epidemics.waves import WavesDataset
from src.datasets.random.synthetic import SyntheticDataset
from src.datasets.social_media.fake_news import FakeNewsDataset
from src.datasets.transform import TemporalSnapshotListTransform
from src.models.cross_snapshot_attention import CrossSnapshotAttentionNet
from src.training.trainer import GNNTrainer
from src.utils.drawing import draw_propagation_graph, draw_propagation_graph_with_attention_weights
from src.training.storing import save_model, load_model


def get_datasets():
    return [
        SyntheticDataset(log_progress=True, num_nodes=20),
        Covid19Dataset(log_progress=True, num_nodes=20),
        FakeNewsDataset(log_progress=True, num_nodes=20),
        BitcoinBlockPropagationDataset(log_progress=True, num_nodes=20),
        DDoSDataset(log_progress=True, num_nodes=20),
        WavesDataset(log_progress=True, num_nodes=20),
    ]


def draw_attention_weights(dataset: Dataset, model: CrossSnapshotAttentionNet, path: str, train=True):
    train_dataset, test_dataset = dataset.transform(
        transform=TemporalSnapshotListTransform)
    dataset_samples = dataset.train if train else dataset.test

    loader = DataLoader(
        train_dataset if train else test_dataset, batch_size=1, shuffle=False
    )
    label_indices = {label_idx: [] for label_idx in range(len(dataset.labels))}
    max_samples = 5

    for i, sample in enumerate(tqdm(loader, "Computing attention weights")):
        _, attention_weights = model.embed(
            sample, return_attention_weights=True)

        node_weights = []
        for t, snapshot in enumerate(attention_weights):
            for n, node in enumerate(snapshot):
                if len(node_weights) < n + 1:
                    node_weights.append([])
                node_weights[n].append(sum(node))

        value = 0
        for sublist in node_weights:
            value += max(sublist) - min(sublist)

        label_indices[dataset_samples[i].label].append(
            (i, value, attention_weights))

    for label_idx, indices in tqdm(label_indices.items(), "Drawing attention weights"):
        count = 1
        for i, _, attention_weights in sorted(indices, key=lambda x: x[1], reverse=True)[:max_samples]:
            propagation = dataset_samples[i].propagation
            label = dataset.labels[dataset_samples[i].label]

            draw_path = f"{path}/{dataset.abbreviation.split('&')[0]}/{label.name}/{count}"
            pos = draw_propagation_graph(
                propagation=propagation, title=f"{label.name}",
                path=f"{draw_path}")
            draw_propagation_graph_with_attention_weights(propagation=propagation, attention_weights=attention_weights,
                                                          title=f"{label.name} ({i})", path=f"{draw_path}_attention_global", pos=pos)
            draw_propagation_graph_with_attention_weights(propagation=propagation, attention_weights=attention_weights,
                                                          title=f"{label.name} ({i})", path=f"{draw_path}_attention_local", local=True, pos=pos)
            count += 1

    # plotting of examples for thesis
    examples = {
        "COV-19": {
            631: [2, 3, 10, 12]
        },
        "FAKE": {
            521: [2, 3, 10, 12],
            338: [1, 2, 3, 4],
        },
        "DDoS": {
            397: [1, 2, 3, 4],
        },
        "SYN": {
            302: [4, 5, 6, 7]
        },
    }

    if dataset.abbreviation.split('&')[0] not in examples:
        return

    examples_indices = examples[dataset.abbreviation.split('&')[0]].keys()
    for label_idx, indices in tqdm(label_indices.items(), "Drawing examples"):
        for i, _, attention_weights in sorted(indices, key=lambda x: x[1], reverse=True)[:max_samples]:
            if i not in examples_indices:
                continue

            timestamps = examples[dataset.abbreviation.split('&')[0]][i]
            propagation = dataset_samples[i].propagation
            label = dataset.labels[dataset_samples[i].label]

            draw_path = f"{path}/examples/{dataset.abbreviation.split('&')[0]}_{label.name}_{i}"
            pos = draw_propagation_graph(
                propagation=propagation, path=f"{draw_path}", timestamps=timestamps, labels=False, pretty=True, svg=True)
            draw_propagation_graph_with_attention_weights(propagation=propagation, attention_weights=attention_weights,
                                                          path=f"{draw_path}_attention_global", pos=pos, timestamps=timestamps, pretty=True, svg=True)
            draw_propagation_graph_with_attention_weights(propagation=propagation, attention_weights=attention_weights,
                                                          path=f"{draw_path}_attention_local", local=True, pos=pos, timestamps=timestamps, pretty=True, svg=True)


def run_experiments(execute=True, **kwargs):
    path = "experiments/explainability/attention_weights"

    if execute:
        for dataset in get_datasets():
            model = CrossSnapshotAttentionNet(
                node_feat_dim=4,
                edge_feat_dim=1,
                attention_layer_dims=[64],
                attention_layer_hidden_dims=[64],
                output_dim=len(dataset.labels),
            )
            GNNTrainer(
                model=model,
                dataset=dataset,
                transform=TemporalSnapshotListTransform,
            ).train(epochs=30)
            model.eval()
            save_model(
                model=model, path=f"{path}/{dataset.abbreviation.split('&')[0]}/model.pt")

    for dataset in get_datasets():
        model = load_model(path=f"{path}/{dataset.abbreviation.split('&')[0]}/model.pt",
                           model_class=CrossSnapshotAttentionNet,
                           node_feat_dim=4,
                           edge_feat_dim=1,
                           attention_layer_dims=[64],
                           attention_layer_hidden_dims=[64],
                           output_dim=len(dataset.labels))
        draw_attention_weights(dataset=dataset, model=model, path=path)


if __name__ == "__main__":
    default_runner(run_experiments)
