from enum import Enum
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops, to_dense_adj
from torch_scatter import scatter

from src.training.storing import load_model


class NodeEmbeddingAggregator(Enum):
    MEAN = "mean"
    GATE = "gate"


class PropagationAggregator(Enum):
    MEAN = "mean"
    LSTM = "lstm"


class PropagationAggregatorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, x):
        output, _ = self.lstm(x)
        return output[-1, :, :]


class NodeEmbeddingAggregatorGate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings, batch_idx):
        gate = torch.sigmoid(self.mlp_gate(node_embeddings))
        graph_embedding = scatter(
            gate * node_embeddings, batch_idx, dim=0, reduce="sum"
        )
        return graph_embedding


class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim: int, num_heads=1, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim, bias=bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.mean(torch.stack([head(x) for head in self.attention_heads]), dim=0)


class CrossSnapshotAttentionLayer(nn.Module):
    """
    Attention layer for computing cross-snapshot dynamics.

    Steps:
    1. Transform node embeddings for each snapshot to hidden dimension
    2. Compute differences between node embeddings of different snapshots
    3. Compute attention weights for each node by using the neighborhood in the previous snapshot
    4. Compute new node embeddings by aggregating with the snapshot differences weighted by the attention weights
    5. Aggregate all snapshot deltas to node propagation embeddings
    6. Aggregate all node embeddings to a graph-level embedding
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        out_dim: Union[int, None],
        node_embedding_aggregator: Union[
            NodeEmbeddingAggregator, None
        ] = NodeEmbeddingAggregator.GATE,
        propagation_aggregator: Union[
            PropagationAggregator, None
        ] = PropagationAggregator.MEAN,
        attention_vision=1,
        attention_heads=1,
        attention_bias=True,
        self_loops=False,
        distributional=False,
        add_last_snapshot_embedding=False,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.attention_vision = attention_vision
        self.attention_heads = attention_heads
        self.attention_bias = attention_bias

        self.self_loops = self_loops
        self.distributional = distributional
        self.add_last_snapshot_embedding = add_last_snapshot_embedding

        # Linear layer for transforming node features to higher dimensional space
        self.node_transform = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
        )

        # Attention-based module for computing cross-snapshot differences
        self.attention = MultiHeadAttention(
            feature_dim=hidden_dim, num_heads=attention_heads, bias=attention_bias) if attention_heads > 0 else None

        # MLP for computing new node embeddings per snapshot using initial node embeddings and attention-weighted snapshot differences
        self.snapshot_delta_concat = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # MLP for computing final node embeddings from hidden dim
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Layers to compute distribution rather than direct embedding
        self.mean_layer = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_layer = nn.Linear(hidden_dim, hidden_dim)

        # Aggregator for combining node embeddings to a graph-level embedding
        self.node_embedding_aggregator = (
            self.get_node_embedding_aggregator(node_embedding_aggregator.value)
            if node_embedding_aggregator is not None
            else None
        )

        # Aggregator for combining snapshot embeddings to a propagation embedding
        self.propagation_aggregator = (
            self.get_propagation_aggregator(propagation_aggregator.value)
            if propagation_aggregator is not None
            else None
        )

    def get_propagation_aggregator(self, aggregator):
        if aggregator == PropagationAggregator.LSTM.value:
            return PropagationAggregatorLSTM(self.hidden_dim, self.hidden_dim, 2)
        return aggregator

    def aggregate_propagation(self, x):
        if type(self.propagation_aggregator) is not str:
            x = torch.stack(x, dim=0)
            return self.propagation_aggregator(x)
        if self.propagation_aggregator == PropagationAggregator.MEAN.value:
            x = torch.stack(x, dim=0)
            return torch.mean(x, dim=0)
        raise ValueError(
            f"Invalid propagation aggregator: {self.propagation_aggregator}"
        )

    def get_node_embedding_aggregator(self, aggregator):
        if aggregator == NodeEmbeddingAggregator.GATE.value:
            return NodeEmbeddingAggregatorGate(self.hidden_dim)
        return aggregator

    def aggregate_node_embeddings(self, x, batch_idx):
        if type(self.node_embedding_aggregator) is not str:
            return self.node_embedding_aggregator(x, batch_idx)
        if self.node_embedding_aggregator == NodeEmbeddingAggregator.MEAN.value:
            return global_mean_pool(x, batch_idx)
        raise ValueError(
            f"Invalid node embeddings aggregator: {self.node_embedding_aggregator}"
        )

    def extract_data_from_snapshots(self, data):
        # we create new batches where the t-batch consists of the t-th snapshot of all batched graphs
        new_data = []
        for _, batch in enumerate(data.snapshots):
            for t, snapshot_data in enumerate(batch):
                if len(new_data) <= t:
                    new_data.append([])
                new_data[t].append(snapshot_data)

        return [Batch.from_data_list(batch) for batch in new_data]

    def get_x(self, x, batch, i):
        if x is None:
            return batch.x
        # when it is the second layer we use the snapshot embeddings from the previous layer as input
        return x[i, :, :] if not self.distributional else x[0][i, :, :]

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def cross_snapshot_attention(self, x, y, adj_matrix, return_attention_weights=False):
        # Compute differences in node embeddings between snapshots
        snapshot_differences = y - x

        if self.attention is not None:
            queries = self.attention(y)
            keys = self.attention(x)
            # Specific attention scores to capture interactions of individual dimensions of specific nodes
            attention_scores = keys * queries
            attention_weights = F.softmax(attention_scores, dim=1)
        else:
            # dummy attention weights
            attention_weights = torch.ones_like(x)

        # Accumulate attention weights from local neighborhoods
        attention_weights = torch.sparse.mm(adj_matrix, attention_weights)

        # Compute weighted cross-snapshot differences
        weighted_snapshot_differences = snapshot_differences * attention_weights

        # Compute new node embeddings by concatenating transformed node features, and weighted cross-snapshot differences
        embeddings = self.snapshot_delta_concat(
            torch.cat(
                [x, weighted_snapshot_differences],
                dim=-1,
            )
        )
        if return_attention_weights:
            return embeddings, attention_weights
        return embeddings

    def transform_node_embeddings(self, x_data, batches):
        node_transformations = []
        for i, batch in enumerate(batches):
            x = self.get_x(x_data, batch, i)
            x_transformed = self.node_transform(x)
            node_transformations.append(x_transformed)
        return node_transformations

    def create_adj_matrix(self, edge_index):
        if self.self_loops:
            edge_index = add_self_loops(edge_index)[0]
        adj_matrix = to_dense_adj(edge_index)[0]
        return adj_matrix.to_sparse().requires_grad_()

    def forward_only(self, x_data, batches, adj_matrices, return_attention_weights=False):
        snapshot_embeddings = []
        attention_weights = []
        batch_idx = batches[0].batch

        # transform the node features of each snapshot
        node_transformations = self.transform_node_embeddings(x_data, batches)

        for i in range(0, len(batches) - self.attention_vision, self.attention_vision):
            adj_matrix = adj_matrices[i]
            node_embeddings = self.cross_snapshot_attention(
                node_transformations[i],
                node_transformations[i + self.attention_vision],
                adj_matrix,
                return_attention_weights=return_attention_weights,
            )
            if return_attention_weights:
                node_embeddings, attention_weight = node_embeddings
                attention_weights.append(
                    attention_weight.detach().cpu().numpy())
            snapshot_embeddings.append(node_embeddings)

        # as we compute the snapshots based on differences, we end up with one less snapshot than we have snapshots
        if self.add_last_snapshot_embedding:
            snapshot_embeddings.append(node_transformations[-1])

        out = (
            self.aggregate_propagation(snapshot_embeddings)
            if self.propagation_aggregator is not None
            else torch.stack(snapshot_embeddings, dim=0)
        )
        out = self.node_mlp(out)

        # compute a distribution to sample from, instead of directly learning the embedding
        if self.distributional:
            mean = self.mean_layer(out)
            logvar = self.logvar_layer(out)
            # During the reparameterization, exponentiating large log-variance can result in large standard deviations,
            # leading to NaN values, clipping them prevents this
            logvar = torch.clamp(logvar, -10, 2)
            out = self.reparameterize(mean, logvar)

        out = (
            self.aggregate_node_embeddings(out, batch_idx)
            if self.node_embedding_aggregator is not None
            else out
        )

        if return_attention_weights:
            return out, attention_weights

        if self.distributional:
            return out, mean, logvar
        return out

    def forward(self, x_data, data, return_attention_weights=False):
        # NOTE: The data needs to be reformated for the snapshotting to work as PyG is not optimized for it
        batches = self.extract_data_from_snapshots(data)
        adj_matrices = [self.create_adj_matrix(
            batch.edge_index) for batch in batches]
        return self.forward_only(x_data, batches, adj_matrices, return_attention_weights)


class CrossSnapshotAttentionNet(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        attention_layer_dims: List[int],
        attention_layer_hidden_dims: List[int],
        output_dim: Union[int, None] = None,
        node_embedding_aggregator: Union[
            NodeEmbeddingAggregator, None
        ] = NodeEmbeddingAggregator.GATE,
        propagation_aggregator: Union[
            PropagationAggregator, None
        ] = PropagationAggregator.MEAN,
        dropout=0.0,
        **layer_kwargs,
    ):
        super().__init__()
        self.activation = nn.ReLU()
        self.distributional = layer_kwargs.get("distributional", False)
        self.dropout = nn.Dropout(dropout)

        layer_sizes = [node_feat_dim] + attention_layer_dims
        self.cross_attention_layers = nn.ModuleList()
        for i, layer_size in enumerate(layer_sizes[:-1]):
            is_not_the_final_attention_layer = i < len(layer_sizes) - 2
            layer = CrossSnapshotAttentionLayer(
                node_feat_dim=layer_size,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=attention_layer_hidden_dims[i],
                out_dim=layer_sizes[i + 1],
                node_embedding_aggregator=None
                if is_not_the_final_attention_layer
                else node_embedding_aggregator,
                propagation_aggregator=None
                if is_not_the_final_attention_layer
                else propagation_aggregator,
                add_last_snapshot_embedding=is_not_the_final_attention_layer,
                **layer_kwargs,
            )
            self.cross_attention_layers.append(layer)

        # Final classification layer
        self.classify = (
            nn.Linear(
                layer_sizes[-1], output_dim) if output_dim is not None else None
        )

    def compute_cross_snapshot_attention(self, data, return_attention_weights=False):
        x = None
        for i, layer in enumerate(self.cross_attention_layers):
            if i == len(self.cross_attention_layers) - 1:
                x = layer(x, data, return_attention_weights)
            else:
                x = layer(x, data, return_attention_weights)
                if self.distributional:
                    x0 = self.activation(x[0])
                    x = (x0, *x[1:])
                else:
                    x = self.activation(x)
        return x

    def embed(self, data, return_attention_weights=False):
        return self.compute_cross_snapshot_attention(data, return_attention_weights=return_attention_weights)

    def forward(self, data):
        x = self.compute_cross_snapshot_attention(data)
        if self.distributional:
            x = (self.dropout(x[0]), *x[1:])
            return (self.classify(x[0]), *x[1:]) if self.classify is not None else x
        x = self.dropout(x)
        return self.classify(x) if self.classify is not None else x


def freeze_attention_layers(
    model: CrossSnapshotAttentionNet
):
    for layer in model.cross_attention_layers:
        for param_name, param in layer.named_parameters():
            if 'attention' in param_name:
                param.requires_grad = False
    return model


def load_csa_net(path: str, freeze: bool = True, **encoder_kwargs):
    # If freeze true, we only load the frozen layers from the stored model
    properties = [["cross_attention_layers", ".attention"]] if freeze else None
    model = load_model(
        path=path, model_class=CrossSnapshotAttentionNet, property_keywords=properties, **encoder_kwargs
    )
    if freeze:
        model = freeze_attention_layers(model)
    return model
