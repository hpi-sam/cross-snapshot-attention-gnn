import torch.nn as nn
import torch

from src.models.cross_snapshot_attention import CrossSnapshotAttentionNet
from src.pretext.predictive.decoder import (
    PropagationEmbeddingPredictivePreTextDecoder,
    NodeEmbeddingsPredictivePreTextDecoder,
)


class NaivePredictivePreTextModel(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.placeholder = nn.Linear(1, 1)

    def forward(self, data):
        current_snapshot = data.current_snapshot
        # just return previous values
        return torch.unsqueeze(current_snapshot.x[:, 0], dim=0)


class PredictivePreTextModel(nn.Module):
    def __init__(
        self,
        version="prop",
        decoder_hidden_dim=64,
        max_num_nodes: int = 100,
        pretext_dropout=0.2,
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        if "output_dim" in encoder_kwargs:
            encoder_kwargs.pop("output_dim")
        if version != "prop" and "node_embedding_aggregator" in encoder_kwargs:
            encoder_kwargs.pop("node_embedding_aggregator")

        self.encoder = CrossSnapshotAttentionNet(
            **encoder_kwargs,
            output_dim=None,
            **({"node_embedding_aggregator": None} if version != "prop" else {}),
        )
        self.embedding_dim = encoder_kwargs["attention_layer_dims"][-1]
        self.decoder_hidden_dim = decoder_hidden_dim
        self.dropout = nn.Dropout(pretext_dropout)

        if version == "prop":
            self.decoder = PropagationEmbeddingPredictivePreTextDecoder(
                embedding_dim=self.embedding_dim,
                hidden_dim=self.decoder_hidden_dim,
                max_num_nodes=max_num_nodes,
            )
        elif version == "node":
            self.decoder = NodeEmbeddingsPredictivePreTextDecoder(
                embedding_dim=self.embedding_dim,
                hidden_dim=self.decoder_hidden_dim,
            )

    def forward(self, data):
        embedding_out = self.encoder(data)
        embedding = embedding_out
        if self.encoder.distributional:
            embedding = embedding[0]

        current_snapshot = data.current_snapshot
        next_snapshot = data.next_snapshot
        if isinstance(current_snapshot, list):
            current_snapshot = current_snapshot[0]
        if isinstance(next_snapshot, list):
            next_snapshot = next_snapshot[0]
        embedding = self.dropout(embedding)
        out = self.decoder(embedding, current_snapshot, next_snapshot)

        # We want to add a business constraint that covered nodes will always stay covered
        prior_attributes = current_snapshot.x[:, 0]
        out = out * (1 - prior_attributes) + prior_attributes

        if self.encoder.distributional:
            return out, *embedding_out[1:]
        return out
