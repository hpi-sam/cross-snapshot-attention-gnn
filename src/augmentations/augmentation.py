import random
from enum import Enum
from typing import List

import torch
from torch_geometric.data import Batch, Data


class PropagationAugmentation(Enum):
    NODE_RELABELING = "node_relabeling"
    TIME_SHIFTING = "time_shifting"
    PATH_REWIRING = "path_rewiring"


def sample_augmentations(k=2) -> List[PropagationAugmentation]:
    all_enum_values = list(PropagationAugmentation)
    sampled_keys = random.sample(all_enum_values, k)
    return sampled_keys


def create_augmented_views_from_graph_batches(
    data_batch: Batch, augmentations: List[PropagationAugmentation] = None, keep_first_unchanged=False, log_num_augmented=False
):
    unique_batch_indices = data_batch.batch.unique(sorted=True)

    views = []
    count_augmented = 0
    count = 0
    for batch_idx in unique_batch_indices:
        node_indices = torch.nonzero(
            data_batch.batch == batch_idx, as_tuple=True)[0]
        x_subgraph = data_batch.x[node_indices]
        edge_indices_subgraph = data_batch.edge_index[
            :, data_batch.batch[data_batch.edge_index[0]] == batch_idx
        ]
        edge_attr_subgraph = data_batch.edge_attr[edge_indices_subgraph[0]]
        snapshots_subgraph = data_batch.snapshots[batch_idx.item()]
        max_collision_score_subgraph = data_batch.max_collision_score[batch_idx.item(
        )]
        y_subgraph = data_batch.y[batch_idx.item()]

        base = Data(
            x=x_subgraph,
            edge_index=edge_indices_subgraph,
            edge_attr=edge_attr_subgraph,
            snapshots=snapshots_subgraph,
            max_collision_score=max_collision_score_subgraph,
            y=y_subgraph,
        )

        for i, augmentation in enumerate(
            augmentations if augmentations is not None else sample_augmentations()
        ):
            if i == 0 and keep_first_unchanged:
                augmentation = None
            if i >= len(views):
                views.append([])
            if augmentation == PropagationAugmentation.NODE_RELABELING:
                augmented = node_relabeling(
                    x=x_subgraph,
                    edge_index=edge_indices_subgraph,
                    edge_attr=edge_attr_subgraph,
                    snapshots=snapshots_subgraph,
                    max_collision_score=max_collision_score_subgraph,
                    y=y_subgraph,
                )
            elif augmentation == PropagationAugmentation.TIME_SHIFTING:
                augmented = time_shifting(
                    x=x_subgraph,
                    edge_index=edge_indices_subgraph,
                    edge_attr=edge_attr_subgraph,
                    snapshots=snapshots_subgraph,
                    max_collision_score=max_collision_score_subgraph,
                    y=y_subgraph,
                )
            elif augmentation == PropagationAugmentation.PATH_REWIRING:
                augmented = path_rewiring(
                    x=x_subgraph,
                    edge_index=edge_indices_subgraph,
                    edge_attr=edge_attr_subgraph,
                    snapshots=snapshots_subgraph,
                    max_collision_score=max_collision_score_subgraph,
                    y=y_subgraph,
                )
            else:
                augmented = Data(
                    x=x_subgraph,
                    edge_index=edge_indices_subgraph,
                    edge_attr=edge_attr_subgraph,
                    snapshots=snapshots_subgraph,
                    max_collision_score=max_collision_score_subgraph,
                    y=y_subgraph,
                )

            if log_num_augmented and augmentation is not None:
                if not are_data_objects_equal(augmented, base):
                    count_augmented += 1
                count += 1
            views[i].append(augmented)

    if log_num_augmented and count > 0:
        print(f"Augmented {((count_augmented / count) * 100):.2f}% of graphs")
    return [Batch.from_data_list(view) for view in views]


def node_relabeling(
    x: List,
    edge_index: List,
    edge_attr: List,
    snapshots: List[Data],
    max_collision_score: List,
    y: List,
):
    """ Node relabeling augmentation.

    Relabels the nodes in the propagation to a random permutation of the node indices. Keeps 
    augmented propagation isomorphic but the tensors are changed.
    """
    # Find the smallest number of nodes shared across snapshots
    min_num_nodes = min([snapshot.num_nodes for snapshot in snapshots])

    # Generate a random relabeling for the shared node set
    shared_node_mapping = list(range(min_num_nodes))
    random.shuffle(shared_node_mapping)
    updated_snapshots = []

    for snapshot in snapshots:
        num_nodes = snapshot.num_nodes

        # Create a new node mapping by extending the shared_node_mapping with the actual node number in the snapshot
        node_mapping = shared_node_mapping + \
            list(range(min_num_nodes, num_nodes))
        node_mapping = torch.tensor(node_mapping, dtype=torch.long)

        # Update the snapshot's x and edge_index objects corresponding to new node mapping
        updated_x = snapshot.x[node_mapping]
        updated_edge_index = node_mapping[snapshot.edge_index]

        updated_snapshot = Data(
            x=updated_x,
            edge_index=updated_edge_index,
            # no need to update edge_attr, as the order of the edges is not changed
            edge_attr=snapshot.edge_attr,
            num_nodes=num_nodes,
        )
        updated_snapshots.append(updated_snapshot)

    return Data(
        # x and edge_index and edge_attr are placeholder anyways, so no need to update
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        snapshots=updated_snapshots,
        max_collision_score=max_collision_score,
        y=y,
    )


def time_shifting(
    x: List,
    edge_index: List,
    edge_attr: List,
    snapshots: List[Data],
    max_collision_score: List,
    y: List,
):
    """ Time shifting augmentation.

    Shifts complete snapshots without changing their content.
    Forward: Insert a snapshot copy at t in between t and t+1
    Backward: Replace a snapshot at t-1 with a copy of t
    """
    n_snapshots = len(snapshots)
    # Randomly select a subset of snapshots to shift
    shift_indices = random.sample(
        range(1, n_snapshots - 1), random.randint(1, n_snapshots // 2)
    )
    # Randomly select a direction for each snapshot to shift
    shift_directions = [random.choice([-1, 1])
                        for _ in range(len(shift_indices))]
    shifted_snapshots = snapshots.copy()

    for idx, direction in zip(shift_indices, shift_directions):
        # Forward shift: copy current snapshot to the next snapshot (i.e. makes propagation look slower)
        if direction == 1:
            shifted_snapshots.insert(idx + 1, snapshots[idx])
        # Backward shift: copy current snapshot to the previous snapshot (i.e. makes propagation look faster)
        elif direction == -1:
            shifted_snapshots[idx - 1] = snapshots[idx]

    new_snapshots = shifted_snapshots[:n_snapshots]

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        snapshots=new_snapshots,
        max_collision_score=max_collision_score,
        y=y,
    )


def path_rewiring(
    x: List,
    edge_index: List,
    edge_attr: List,
    snapshots: List[Data],
    max_collision_score: List,
    y: List,
):
    """Path rewiring augmentation.

    Looks for following pattern in all snapshots:
    At i-1: There is a covered node a, and there is an edge to another node b. b needs to be uncovered at i, but covered at i+1,
    and edge (a,b) needs to be the edge that produces the new covering. Further, there needs to be a node c that is a common neighbor of
    both nodes a and b. This node needs to be uncovered at i-1 and i. For all these combinations, we sample one and rewire the propagation path
    to be over node c instead of the direct path from a to b. Therefore, we set the edge (a,b) to 0, the edge (a,c) to 1, and the edge (b,c) to 1.
    Also, the node c is covered at i.

    Technically, we could also investigate the reverse operation (merging from a common neighbor), but this seems more complicated,
    as we would need to remove the covered state for a node, which would propagate in the removal of many other nodes (where the initially augmented node was the source).
    This could trickle down so much, that the augmented propagation graph could look very different. Otherwise, in our implementation here, the patterns
    should still be very similar to the original propagation graph, as we only insert one new covered node at a time.

    """

    updated_snapshots = snapshots.copy()
    max_augments = 20
    augments = 0
    break_out = False

    def find_edge_indices(snapshot, source, target):
        mask = (snapshot.edge_index[0] == source) & (
            snapshot.edge_index[1] == target)
        return mask.nonzero(as_tuple=True)[0]

    for i in range(1, len(updated_snapshots) - 1):
        prev_snapshot = updated_snapshots[i - 1]
        curr_snapshot = updated_snapshots[i]
        next_snapshot = updated_snapshots[i + 1]

        min_covered_nodes = int(curr_snapshot.num_nodes * 0.2)
        # When there are too few covered nodes, we continue, as it is pretty unlikely we can augment here anyway
        if (
            len((curr_snapshot.x[:, 0] == 1).nonzero(
                as_tuple=True)[0].tolist())
            < min_covered_nodes
        ):
            continue

        # When there are too few uncovered nodes, we stop augmenting, as it is pretty unlikely we can augment anyway
        min_uncovered_nodes = int(curr_snapshot.num_nodes * 0.2)
        if (
            len((curr_snapshot.x[:, 0] == 0).nonzero(
                as_tuple=True)[0].tolist())
            < min_uncovered_nodes
        ):
            break

        # Find nodes that are covered at the previous and current snapshots
        prev_active_nodes = (
            prev_snapshot.x[:, 0] == 1).nonzero(as_tuple=True)[0]
        next_active_nodes = (
            next_snapshot.x[:, 0] == 1).nonzero(as_tuple=True)[0]

        # Calculate adjacency matrix for the previous snapshot
        adj_matrix_prev = torch.zeros(
            prev_snapshot.x.shape[0], prev_snapshot.x.shape[0]
        )
        adj_matrix_prev[prev_snapshot.edge_index[0],
                        prev_snapshot.edge_index[1]] = 1

        for a in prev_active_nodes:
            for b in next_active_nodes:
                if a == b:
                    continue
                # Only consider cases of a and b, where there is an edge (a,b) and that edge is the one that was used for propagation
                ab_indices = find_edge_indices(curr_snapshot, a, b)
                if ab_indices.numel() == 0 or not torch.all(
                    curr_snapshot.edge_attr[ab_indices] == 1
                ):
                    continue

                # Check if there are some common neighbors of a and b
                common_neighbors = (adj_matrix_prev[a] * adj_matrix_prev[b]).nonzero(
                    as_tuple=True
                )[0]

                if common_neighbors.numel() == 0:
                    continue

                # Filter common neighbors that are uncovered at the previous snapshot and the current snapshot
                valid_common_neighbors = common_neighbors[
                    (prev_snapshot.x[common_neighbors, 0] == 0)
                    & (curr_snapshot.x[common_neighbors, 0] == 0)
                ]

                if valid_common_neighbors.numel() == 0:
                    continue

                augments += 1
                # Randomly sample one valid common neighbor
                c = valid_common_neighbors[
                    random.randint(0, valid_common_neighbors.numel() - 1)
                ]

                # Mark the common neighbor node as becoming covered at the current snapshot
                curr_snapshot.x[c, :2] = 1
                for future_snapshot in updated_snapshots[i + 1:]:
                    future_snapshot.x[c] = torch.tensor([1])

                # Update the edge_attr tensor for the edges (a, b), (a, c), and (b, c)
                for snapshot in snapshots[i:]:
                    ab_indices = find_edge_indices(snapshot, a, b)
                    bc_indices = find_edge_indices(snapshot, b, c)
                    # (a,b) needs to be set to 0, as it is not the propagating edge anymore
                    if ab_indices.numel() > 0:
                        snapshot.edge_attr[ab_indices] = torch.tensor(
                            [0], dtype=snapshot.edge_attr.dtype
                        )
                    # (b,c) needs to be set to 1, as it is a propagating edge now (starting from i)
                    if bc_indices.numel() > 0:
                        snapshot.edge_attr[bc_indices] = torch.tensor(
                            [1], dtype=snapshot.edge_attr.dtype
                        )

                for snapshot in snapshots[i - 1:]:
                    ac_indices = find_edge_indices(snapshot, a, c)
                    # (b,c) needs to be set to 1, as it is a propagating edge now (starting from i-1)
                    if ac_indices is not None:
                        snapshot.edge_attr[ac_indices] = torch.tensor(
                            [1], dtype=snapshot.edge_attr.dtype
                        )

                if augments >= max_augments:
                    break_out = True
                    break

            else:  # Continue if the inner loop wasn't broken
                continue

            break
        if break_out:
            break

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        snapshots=updated_snapshots,
        max_collision_score=max_collision_score,
        y=y,
    )


def are_data_objects_equal(data1: Data, data2: Data) -> bool:
    # Check if they have the same keys (attributes)
    if set(data1.keys) != set(data2.keys):
        return False

    # Check if the attributes are equal
    for key in data1.keys:
        attr1 = data1[key]
        attr2 = data2[key]

        if isinstance(attr1, torch.Tensor):
            if not torch.allclose(attr1, attr2, rtol=1e-05, atol=1e-08):
                return False
        elif isinstance(attr1, Data):
            if not are_data_objects_equal(attr1, attr2):
                return False
        else:
            if attr1 != attr2:
                return False

    return True
