import torch
import torch.nn as nn


def weighted_bce_loss(
    y_pred, y_true, prior_attributes, pos_weight=1.0
):
    # filter out the cases where the prior attributes are already 1
    indices = torch.where(prior_attributes != 1)[0]
    if indices.numel() == 0:
        indices = torch.arange(y_pred.shape[0])
    y_pred = y_pred[:, indices]
    y_true = y_true[:, indices]

    # increase the weight for cases where the prior attributes change from 0 to 1
    weights = torch.ones_like(y_true)
    change_case_indices = (y_true == 1).nonzero(as_tuple=True)
    weights[change_case_indices] = pos_weight

    bce_loss = torch.nn.functional.binary_cross_entropy(
        y_pred, y_true, weight=weights, reduction="none"
    )
    return bce_loss.mean()


class WeightedWithFilteredPriorsBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, y_pred, y_true, prior_attributes):
        return weighted_bce_loss(
            y_pred=y_pred,
            y_true=y_true,
            prior_attributes=prior_attributes,
            pos_weight=self.pos_weight,
        )


class KLDLoss(nn.Module):
    def __init__(self, base_loss, beta=1.0):
        super().__init__()
        self.base_loss = base_loss
        self.beta = beta

    def kl_divergence(self, mean, logvar):
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return kl_div

    def forward(self, *args):
        base_loss = self.base_loss(*args[:-2])
        kl_loss = (
            self.kl_divergence(args[-2], args[-1]) / args[-2].numel()
        )  # normalize by number of nodes in graph
        total_loss = base_loss + self.beta * kl_loss
        return total_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss function. See: https://arxiv.org/pdf/2010.13902.pdf"""

    def __init__(self, temperature=0.5, normalize_embeddings=False):
        super().__init__()
        # The temperature parameter is used to control the sharpness of the distribution
        # of similarities, making it more or less concentrated around high values.
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings

    def forward(self, z_i, z_j, negative_samples):
        # Normalize the embeddings to have unit L2 norm.
        # This is done to constrain the embeddings in a hypersphere,
        # making it easier to compare them using dot products.
        if self.normalize_embeddings:
            z_i = torch.nn.functional.normalize(z_i, dim=1)
            z_j = torch.nn.functional.normalize(z_j, dim=1)
            negative_samples = torch.nn.functional.normalize(
                negative_samples, dim=1)

        # Compute the cosine similarity between the positive pair (z_i and z_j)
        # by taking their dot product, since they have been normalized.
        positive_similarity = torch.sum(z_i * z_j, dim=-1) / self.temperature

        # Compute the cosine similarity between the anchor (z_i) and the negative samples.
        # This is done by taking the dot product between z_i and the transposed
        # negative samples matrix.
        negative_similarity = torch.matmul(
            z_i, negative_samples.t()) / self.temperature

        # Use the log-sum-exp trick for numerical stability when computing the log-sum-exp
        # of the negative similarities. This trick helps avoid issues with very large
        # or very small values that could cause numerical instability.
        max_values, _ = torch.max(negative_similarity, dim=-1, keepdim=True)
        negative_similarity = negative_similarity - max_values

        log_sum_exp = torch.logsumexp(negative_similarity, dim=-1)
        log_sum_exp = log_sum_exp + max_values.squeeze()

        # Calculate the final contrastive loss by subtracting the positive similarity
        # from the log-sum-exp of the negative similarities. The objective is to maximize
        # the positive similarity while minimizing the sum of negative similarities.
        loss = -positive_similarity + log_sum_exp

        # Return the mean loss across all examples in the batch.
        return torch.mean(loss)
