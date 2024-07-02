from enum import Enum
from src.perturbations.edge_addition import RandomEdgeAddition
from src.perturbations.edge_deletion import RandomEdgeDeletion
from src.perturbations.node_addition import RandomNodeAddition
from src.perturbations.node_deletion import RandomNodeDeletion
from src.perturbations.edge_rewiring import AssortativityEdgeRewiring, ClusteringEdgeRewiring


class PerturbationMethod(Enum):
    PRIOR_EDGE_DELETION = "PriorEdgeDeletion"
    PRIOR_NODE_DELETION = "PriorNodeDeletion"
    PRIOR_EDGE_ADDITION = "PriorEdgeAddition"
    PRIOR_NODE_ADDITION = "PriorNodeAddition"
    PRIOR_ASSORTATIVITY_INCREASE = "PriorAssortativityIncrease"
    PRIOR_ASSORTATIVITY_DECREASE = "PriorAssortativityDecrease"
    PRIOR_CLUSTERING_INCREASE = "PriorClusteringIncrease"
    PRIOR_CLUSTERING_DECREASE = "PriorClusteringDecrease"
    POSTERIOR_EDGE_DELETION = "PosteriorEdgeDeletion"
    POSTERIOR_NODE_DELETION = "PosteriorNodeDeletion"
    POSTERIOR_EDGE_ADDITION = "PosteriorEdgeAddition"
    POSTERIOR_NODE_ADDITION = "PosteriorNodeAddition"
    POSTERIOR_ASSORTATIVITY_INCREASE = "PosteriorAssortativityIncrease"
    POSTERIOR_ASSORTATIVITY_DECREASE = "PosteriorAssortativityDecrease"
    POSTERIOR_CLUSTERING_INCREASE = "PosteriorClusteringIncrease"
    POSTERIOR_CLUSTERING_DECREASE = "PosteriorClusteringDecrease"


class PerturbationFactorMethod(Enum):
    PRIOR_EDGE_DELETION = lambda factor: {
        "deletion_amount": factor,
        "method": "proportionate",
    }
    PRIOR_NODE_DELETION = lambda factor: {
        "deletion_amount": factor,
        "method": "proportionate",
    }
    PRIOR_EDGE_ADDITION = lambda factor: {
        "addition_amount": factor,
        "method": "proportionate",
    }
    PRIOR_NODE_ADDITION = lambda factor: {
        "addition_amount": factor,
        "method": "proportionate",
    }
    PRIOR_ASSORTATIVITY_INCREASE = lambda factor: {
        "rewiring_amount": factor,
        "increase": True,
    }
    PRIOR_ASSORTATIVITY_DECREASE = lambda factor: {
        "rewiring_amount": factor,
        "increase": False,
    }
    PRIOR_CLUSTERING_INCREASE = lambda factor: {
        "rewiring_amount": factor,
        "increase": True,
    }
    PRIOR_CLUSTERING_DECREASE = lambda factor: {
        "rewiring_amount": factor,
        "increase": False,
    }
    POSTERIOR_EDGE_DELETION = lambda factor: {
        "deletion_amount": factor,
        "method": "proportionate",
        "posterior": True,
    }
    POSTERIOR_NODE_DELETION = lambda factor: {
        "deletion_amount": factor,
        "method": "proportionate",
        "posterior": True,
    }
    POSTERIOR_EDGE_ADDITION = lambda factor: {
        "addition_amount": factor,
        "method": "proportionate",
        "posterior": True,
    }
    POSTERIOR_NODE_ADDITION = lambda factor: {
        "addition_amount": factor,
        "method": "proportionate",
        "posterior": True,
    }
    POSTERIOR_ASSORTATIVITY_INCREASE = lambda factor: {
        "rewiring_amount": factor,
        "increase": True,
        "posterior": True,
    }
    POSTERIOR_ASSORTATIVITY_DECREASE = lambda factor: {
        "rewiring_amount": factor,
        "increase": False,
        "posterior": True,
    }
    POSTERIOR_CLUSTERING_INCREASE = lambda factor: {
        "rewiring_amount": factor,
        "increase": True,
        "posterior": True,
    }
    POSTERIOR_CLUSTERING_DECREASE = lambda factor: {
        "rewiring_amount": factor,
        "increase": False,
        "posterior": True,
    }


def get_perturbation_class(method: PerturbationMethod):
    if (
        method == PerturbationMethod.PRIOR_EDGE_DELETION
        or method == PerturbationMethod.POSTERIOR_EDGE_DELETION
    ):
        return RandomEdgeDeletion
    elif (
        method == PerturbationMethod.PRIOR_NODE_DELETION
        or method == PerturbationMethod.POSTERIOR_NODE_DELETION
    ):
        return RandomNodeDeletion
    elif (
        method == PerturbationMethod.PRIOR_EDGE_ADDITION
        or method == PerturbationMethod.POSTERIOR_EDGE_ADDITION
    ):
        return RandomEdgeAddition
    elif (
        method == PerturbationMethod.PRIOR_NODE_ADDITION
        or method == PerturbationMethod.POSTERIOR_NODE_ADDITION
    ):
        return RandomNodeAddition
    elif (
        method == PerturbationMethod.PRIOR_ASSORTATIVITY_INCREASE
        or method == PerturbationMethod.POSTERIOR_ASSORTATIVITY_INCREASE
        or method == PerturbationMethod.PRIOR_ASSORTATIVITY_DECREASE
        or method == PerturbationMethod.POSTERIOR_ASSORTATIVITY_DECREASE
    ):
        return AssortativityEdgeRewiring
    elif (
        method == PerturbationMethod.PRIOR_CLUSTERING_INCREASE
        or method == PerturbationMethod.POSTERIOR_CLUSTERING_INCREASE
        or method == PerturbationMethod.PRIOR_CLUSTERING_DECREASE
        or method == PerturbationMethod.POSTERIOR_CLUSTERING_DECREASE
    ):
        return ClusteringEdgeRewiring
    else:
        raise ValueError(f"Unknown perturbation method: {method}")
