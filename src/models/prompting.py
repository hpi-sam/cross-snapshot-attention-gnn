import torch.nn as nn
from torch_geometric.data import Batch

from src.pretext.predictive.model import PredictivePreTextModel
from src.models.cross_snapshot_attention import CrossSnapshotAttentionNet


class PrePromptingModel(nn.Module):
    """This model first predicts the next snapshots using the predictor model, and then does the classification using the
    classifier model on this sample.
    """

    def __init__(
        self,
        predictor: PredictivePreTextModel,
        classifier: CrossSnapshotAttentionNet,
        prediction_amount: int,
    ) -> None:
        super().__init__()
        self.prediction_amount = prediction_amount
        self.predictor = predictor
        self.classifier = classifier

    def predict_snapshot(self, data, i):
        samples = data.clone().to_data_list()
        predictions = []
        for sample in samples:
            idx_to_predict = -self.prediction_amount + i
            old_sample = sample.clone()

            sample.snapshots = old_sample.snapshots[
                :idx_to_predict
            ]
            sample.current_snapshot = old_sample.snapshots[
                idx_to_predict - 1
            ]
            sample.next_snapshot = old_sample.snapshots[
                idx_to_predict
            ]
            predicted_thresholds = self.predictor(
                Batch.from_data_list([sample])).detach()
            predictions.append((predicted_thresholds > 0.5).float())
        return predictions

    def forward(self, data):
        new_data = data.clone()
        if self.training or not self.training:
            for i in range(self.prediction_amount):
                predicted = self.predict_snapshot(new_data, i)
                # update original data with new snapshot for each sample j in the batch
                for j, _ in enumerate(new_data.snapshots):
                    new_data.snapshots[j][-self.prediction_amount + i].x[
                        :, 0
                    ] = predicted[j]

        return self.classifier(new_data)
