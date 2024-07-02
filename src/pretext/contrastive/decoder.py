import torch.nn as nn


class ContrastivePreTextDecoder(nn.Module):
    """Projection head for the contrastive learning task, implemented as a MLP with two layers, as suggested here: https://arxiv.org/pdf/2010.13902.pdf"""

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        embedding,
    ):
        out = self.mlp(embedding)
        return out
