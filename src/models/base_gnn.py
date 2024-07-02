from typing import List

from torch import nn
from torch_geometric.nn import GATConv, GraphConv, global_mean_pool


class BaseGNN(nn.Module):
    def __init__(
        self, layer_sizes: List[int], edge_feat_dim: int = None, conv_type: str = "GCN"
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.conv_type = conv_type
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.0)
        self.edge_feat_dim = edge_feat_dim

        self.conv_layers = nn.ModuleList()
        for i, layer_size in enumerate(layer_sizes[1:-1]):
            layer = self._get_conv_layer(layer_sizes[i], layer_size)
            self.conv_layers.append(layer)
        self.out_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def _get_conv_layer(self, input_size, output_size):
        if self.conv_type == "GCN":
            return GraphConv(input_size, output_size)
        elif self.conv_type == "GAT":
            return GATConv(input_size, output_size, edge_dim=self.edge_feat_dim)
        else:
            raise ValueError(
                f"Invalid convolutional layer type: {self.conv_type}")

    def compute_conv(self, x, edge_index, edge_attr):
        for i, layer in enumerate(self.conv_layers):
            if i == len(self.conv_layers) - 1:
                x = (
                    layer(x, edge_index, edge_attr)
                    if self.conv_type == "GAT" and self.edge_feat_dim is not None
                    else layer(x, edge_index)
                )
            else:
                x = self.activation(
                    layer(x, edge_index, edge_attr)
                    if self.conv_type == "GAT" and self.edge_feat_dim is not None
                    else layer(x, edge_index)
                )
        return x

    def forward_only(self, data):
        x, edge_index, batch_idx, edge_attr = (
            data.x,
            data.edge_index,
            data.batch,
            data.edge_attr,
        )
        x = self.compute_conv(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch_idx)
        x = self.dropout(x)
        return self.out_layer(x)

    def forward(self, data):
        return self.forward_only(data)
