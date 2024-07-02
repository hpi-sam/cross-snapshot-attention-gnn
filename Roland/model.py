import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, GRUCell
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, Linear
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool


class ROLAND(torch.nn.Module):
    def __init__(self, input_dim, num_nodes, num_classes, dropout=0.0, update='mlp', loss=BCEWithLogitsLoss, second_layer=True):

        super(ROLAND, self).__init__()
        # Architecture:
        # 2 MLP layers to preprocess BERT repr,
        # 2 GCN layer to aggregate node embeddings
        # HadamardMLP as link prediction decoder

        # You can change the layer dimensions but
        # if you change the architecture you need to change the forward method too
        # TODO: make the architecture parameterizable

        # Layer dimensions
        hidden_conv_1 = 64
        hidden_conv_2 = 32

        # Preprocessing layers
        self.preprocess1 = Linear(input_dim, 256)
        self.preprocess2 = Linear(256, 128)

        # GCN layers
        if second_layer:
            self.conv1 = GCNConv(128, hidden_conv_1)
            self.conv2 = GCNConv(hidden_conv_1, hidden_conv_2)
        else:
            self.conv1 = GCNConv(128, hidden_conv_2)

        # Post-processing for graph classification
        self.global_pool = global_mean_pool  # Global mean pooling
        # Classifier for graph-level predictions
        self.classifier = Linear(hidden_conv_2, num_classes)

        # Loss function for graph classification
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.second_layer = second_layer

        # Dropout and updates
        self.dropout = dropout
        self.update = update
        if second_layer:
            if update == 'gru':
                self.gru1 = GRUCell(hidden_conv_1, hidden_conv_1)
                self.gru2 = GRUCell(hidden_conv_2, hidden_conv_2)
            elif update == 'mlp':
                self.mlp1 = Linear(hidden_conv_1 * 2, hidden_conv_1)
                self.mlp2 = Linear(hidden_conv_2 * 2, hidden_conv_2)
        else:
            if update == 'gru':
                self.gru1 = GRUCell(hidden_conv_2, hidden_conv_2)
            elif update == 'mlp':
                self.mlp1 = Linear(hidden_conv_2 * 2, hidden_conv_2)

        if second_layer:
            self.previous_embeddings = [
                torch.zeros(num_nodes, hidden_conv_1),
                torch.zeros(num_nodes, hidden_conv_2)
            ]
        else:
            self.previous_embeddings = [
                torch.zeros(num_nodes, hidden_conv_2)
            ]

    def reset_loss(self, loss=BCEWithLogitsLoss):
        self.loss_fn = loss()

    def reset_parameters(self):
        self.preprocess1.reset_parameters()
        self.preprocess2.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.postprocess1.reset_parameters()

    def extract_data_from_snapshots(self, data):
        # we create new batches where the t-batch consists of the t-th snapshot of all batched graphs
        new_data = []
        for _, batch in enumerate(data.snapshots):
            for t, snapshot_data in enumerate(batch):
                if len(new_data) <= t:
                    new_data.append([])
                new_data[t].append(snapshot_data)

        return [Batch.from_data_list(batch) for batch in new_data]

    def forward(self, data):

        batches = self.extract_data_from_snapshots(data)
        batch_idx = batches[0].batch

        outputs = []
        for batch in batches:
            x, edge_index = batch.x, batch.edge_index
            num_nodes = x.size(0)
            # Process initial node features through preprocessing layers
            x = F.leaky_relu(self.preprocess1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.leaky_relu(self.preprocess2(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

            # First GCN layer
            x = self.conv1(x, edge_index)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Update embeddings if required
            if self.update == 'gru':
                x = self.gru1(x, self.previous_embeddings[0][:num_nodes])
            elif self.update == 'mlp':
                combined = torch.cat(
                    (x, self.previous_embeddings[0][:num_nodes]), dim=1)
                x = self.mlp1(combined)

            if self.second_layer:
                # Second GCN layer
                x = self.conv2(x, edge_index)
                x = F.leaky_relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                # Update embeddings if required
                if self.update == 'gru':
                    x = self.gru2(x, self.previous_embeddings[1][:num_nodes])
                elif self.update == 'mlp':
                    combined = torch.cat(
                        (x, self.previous_embeddings[1][:num_nodes]), dim=1)
                    x = self.mlp2(combined)

            # Global pooling to get graph-level representation
            # Aggregate node features to graph-level
            x = self.global_pool(x, batch_idx)

            outputs.append(x)

        x = torch.stack(outputs, dim=1)
        x = torch.mean(x, dim=1).squeeze()

        # Classification layer
        x = self.classifier(x)  # Generate class scores for each graph
        return x

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
