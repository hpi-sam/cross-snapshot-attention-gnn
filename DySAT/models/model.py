import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from DySAT.models.layers import StructuralAttentionLayer, TemporalAttentionLayer


class DySAT(nn.Module):
    def __init__(self, args, num_features, time_length, num_classes):
        super(DySAT, self).__init__()
        self.args = args
        self.num_time_steps = min(
            time_length, args.window + 1) if args.window >= 0 else time_length
        self.num_features = num_features
        self.num_classes = num_classes

        self.structural_head_config = list(
            map(int, args.structural_head_config.split(",")))
        self.structural_layer_config = list(
            map(int, args.structural_layer_config.split(",")))
        self.temporal_head_config = list(
            map(int, args.temporal_head_config.split(",")))
        self.temporal_layer_config = list(
            map(int, args.temporal_layer_config.split(",")))
        self.spatial_drop = args.spatial_drop
        self.temporal_drop = args.temporal_drop

        self.structural_attn, self.temporal_attn = self.build_model()

        # Adding a global pooling layer
        self.global_pooling = nn.AdaptiveAvgPool1d(1)

        # Classifier layer
        self.classifier = nn.Linear(
            self.temporal_layer_config[-1], num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data):
        batches = self.extract_data_from_snapshots(data)
        batch_idx = batches[0].batch

        structural_out = []
        # Process each individual graph if required
        for batch in batches:
            # Assuming batch is a list of Data objects for each snapshot
            structural_out.append(self.structural_attn(batch))

        structural_outputs = [g.x[:, None, :]
                              for g in structural_out]  # list of [Ni, 1, F]

        # padding outputs along with Ni
        maximum_node_num = structural_outputs[-1].shape[0]
        out_dim = structural_outputs[-1].shape[-1]
        structural_outputs_padded = []
        for out in structural_outputs:
            zero_padding = torch.zeros(
                maximum_node_num-out.shape[0], 1, out_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_padded.append(padded)
        structural_outputs_padded = torch.cat(
            structural_outputs_padded, dim=1)  # [N, T, F]

        # Adjust temporal_attn to handle this structure
        temporal_out = self.temporal_attn(structural_outputs_padded)
        temporal_pool = torch.mean(temporal_out, dim=1).squeeze()
        # Apply global pooling to reduce the node dimension
        graph_pool = global_mean_pool(temporal_pool, batch_idx)  # [T, F]

        logits = self.classifier(graph_pool)
        return logits

    def extract_data_from_snapshots(self, data):
        # we create new batches where the t-batch consists of the t-th snapshot of all batched graphs
        new_data = []
        for _, batch in enumerate(data.snapshots):
            for t, snapshot_data in enumerate(batch):
                if len(new_data) <= t:
                    new_data.append([])
                new_data[t].append(snapshot_data)

        return [Batch.from_data_list(batch) for batch in new_data]

    def build_model(self):
        input_dim = self.num_features
        structural_attention_layers = nn.Sequential()
        for i, output_dim in enumerate(self.structural_layer_config):
            layer = StructuralAttentionLayer(input_dim=input_dim, output_dim=output_dim,
                                             n_heads=self.structural_head_config[i],
                                             attn_drop=self.spatial_drop, ffd_drop=self.spatial_drop,
                                             residual=self.args.residual)
            structural_attention_layers.add_module(
                name=f"structural_layer_{i}", module=layer)
            input_dim = output_dim

        temporal_attention_layers = nn.Sequential()
        for i, heads in enumerate(self.temporal_head_config):
            layer = TemporalAttentionLayer(input_dim=input_dim, n_heads=heads,
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop, residual=self.args.residual)
            temporal_attention_layers.add_module(
                name=f"temporal_layer_{i}", module=layer)
            input_dim = self.temporal_layer_config[i]

        return structural_attention_layers, temporal_attention_layers

    def get_loss(self, logits, targets):
        return self.criterion(logits, targets)


class DySatArgs:
    def __init__(self, heads=None):
        self.time_steps = 16  # total time steps used for train, eval and test
        self.dataset = 'Enron'  # dataset name
        self.GPU_ID = 0  # GPU_ID (0/1 etc.)
        self.epochs = 200  # # epochs
        self.val_freq = 1  # Validation frequency (in epochs)
        self.test_freq = 1  # Testing frequency (in epochs)
        self.batch_size = 512  # Batch size (# nodes)
        self.featureless = True  # True if one-hot encoding
        self.early_stop = 10  # patient
        self.residual = True  # Use residual
        self.neg_sample_size = 10  # # negative samples per positive
        self.walk_len = 20  # Walk length for random walk sampling
        self.neg_weight = 1.0  # Weightage for negative samples
        self.learning_rate = 0.01  # Initial learning rate for self-attention model
        # Spatial (structural) attention Dropout (1 - keep probability)
        self.spatial_drop = 0.1
        # Temporal attention Dropout (1 - keep probability)
        self.temporal_drop = 0.5
        self.weight_decay = 0.0005  # Initial learning rate for self-attention model
        # Encoder layer config: # attention heads in each GAT layer
        self.structural_head_config = '16,8,8' if heads is None else heads
        # Encoder layer config: # units in each GAT layer
        self.structural_layer_config = '64'
        # Encoder layer config: # attention heads in each Temporal layer
        self.temporal_head_config = '16' if heads is None else heads
        # Encoder layer config: # units in each Temporal layer
        self.temporal_layer_config = '64'
        self.position_ffn = 'True'  # Position wise feedforward
        # Window for temporal attention (default : -1 => full)
        self.window = -1


def get_model_args(heads=None):
    return DySatArgs(heads=heads)
