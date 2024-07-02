import torch.nn as nn

from src.models.cross_snapshot_attention import CrossSnapshotAttentionNet
from src.pretext.contrastive.decoder import ContrastivePreTextDecoder


class ContrastivePreTextModel(nn.Module):
    def __init__(
        self,
        decoder_hidden_dim=64,
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        if "output_dim" in encoder_kwargs:
            encoder_kwargs.pop("output_dim")

        self.encoder = CrossSnapshotAttentionNet(
            **encoder_kwargs,
            output_dim=None,
        )
        self.embedding_dim = encoder_kwargs["attention_layer_dims"][-1]
        self.decoder_hidden_dim = decoder_hidden_dim
        self.dropout = nn.Dropout(0.2)
        self.decoder = ContrastivePreTextDecoder(
            embedding_dim=self.embedding_dim, hidden_dim=self.decoder_hidden_dim
        )

    def forward(self, data):
        embedding_out = self.encoder(data)
        embedding = embedding_out
        if self.encoder.distributional:
            embedding = embedding[0]
        embedding = self.dropout(embedding)
        out = self.decoder(embedding)
        if self.encoder.distributional:
            return out, *embedding_out[1:]
        return out
