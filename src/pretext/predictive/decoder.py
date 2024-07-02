import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, to_dense_adj


class PropagationEmbeddingPredictivePreTextDecoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, max_num_nodes: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_num_nodes = max_num_nodes
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.sigmoid = nn.Sigmoid()
        # Create weight matrix to handle maximum number of nodes to predict the propagation state for different numbers of nodes
        self.final_predict = nn.Parameter(
            torch.Tensor(self.hidden_dim, self.max_num_nodes)
        )
        nn.init.kaiming_uniform_(self.final_predict, a=math.sqrt(5))

    def forward(self, embedding, _current_snapshot, next_snapshot):
        out = self.mlp(embedding)
        num_nodes = next_snapshot.num_nodes
        # Use the required slice of the weight matrix
        out = torch.matmul(out, self.final_predict[:, :num_nodes])
        return self.sigmoid(out)


class NodeEmbeddingsPredictivePreTextDecoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.attention = nn.Linear(embedding_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def create_adj_matrix(self, edge_index, batch=None):
        edge_index = add_self_loops(edge_index)[0]
        adj_matrix = to_dense_adj(edge_index, batch=batch)[0]
        return adj_matrix

    def forward(self, embedding, _current_snapshot, next_snapshot):
        attention_keys = self.attention(embedding)
        attention_values = self.attention(embedding)
        attention_weights = F.softmax(attention_values * attention_keys, dim=1)
        adj_matrix = self.create_adj_matrix(
            next_snapshot.edge_index, batch=next_snapshot.batch
        )
        attention_weights = torch.mm(adj_matrix, attention_weights)
        out = self.mlp(attention_keys * attention_weights).transpose(0, 1)
        return self.sigmoid(out)
