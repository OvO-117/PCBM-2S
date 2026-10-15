"""Validated dual-graph input and batching for P-CBM-2S.

PyTorch ``.pt`` files are pickle-based. Only load files from a trusted source.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


CONCEPT_NAMES = (
    "vina_score",
    "n_contacts",
    "n_hbond",
    "n_hydrophobic",
    "n_contact_residues",
)
RWSE_STEPS = (1, 2, 4, 8, 16)


class DualGraphData(Data):
    """A pair with independent ligand and protein node-index spaces."""

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "prot_edge_index":
            if getattr(self, "prot_node_x", None) is None:
                raise ValueError("prot_node_x is required to batch prot_edge_index")
            return self.prot_node_x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


@lru_cache(maxsize=1)
def _category_dimensions() -> tuple[tuple[int, ...], tuple[int, ...]]:
    # Load feature vocabularies independently of the encoder backend.
    from .compound_tools import (
        CompoundKit,
        get_atom_feature_dims,
        get_bond_feature_dims,
    )

    atoms = tuple(get_atom_feature_dims(list(CompoundKit.atom_vocab_dict)))
    bonds = tuple(get_bond_feature_dims(list(CompoundKit.bond_vocab_dict)))
    if len(atoms) != 18 or len(bonds) != 5:
        raise ValueError("The released feature vocabulary must contain 18 atom and 5 bond fields")
    return atoms, bonds


def _tensor(sample: Data, name: str) -> torch.Tensor:
    value = getattr(sample, name, None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.is_complex() or value.dtype == torch.bool:
        raise ValueError(f"{name} must contain real numeric values")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinite values")
    return value


def _edge_index(sample: Data, name: str, nodes: int) -> torch.Tensor:
    edge = _tensor(sample, name)
    if edge.dtype != torch.long or edge.ndim != 2 or edge.size(0) != 2:
        raise ValueError(f"{name} must be int64 with shape [2, E]")
    if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= nodes):
        raise ValueError(f"{name} contains a node index outside [0, {nodes})")
    return edge


def _categorical(values: torch.Tensor, sizes: Sequence[int], name: str) -> None:
    if values.numel() and values.is_floating_point() and not torch.equal(values, values.round()):
        raise ValueError(f"{name} must contain integer category IDs")
    for column, size in enumerate(sizes):
        column_values = values[:, column]
        if column_values.numel() and (
            int(column_values.min()) < 0 or int(column_values.max()) >= size
        ):
            raise ValueError(f"{name} column {column} has category IDs outside [0, {size})")


def validate_sample(sample: Data, *, require_label: bool = True) -> None:
    """Raise ``ValueError`` for malformed tensors or invalid feature values.

    Both 28- and 31-column protein formats are accepted; the model consumes
    the first 28 columns. Protein RWSE is validated when present.
    """
    if not isinstance(sample, Data):
        raise ValueError("Each sample must be a PyG Data object")
    x = _tensor(sample, "x")
    if x.ndim != 2 or x.size(0) < 1 or x.size(1) != 21:
        raise ValueError("x must have shape [N, 21] with N >= 1")
    atom_sizes, bond_sizes = _category_dimensions()
    _categorical(x[:, :18], atom_sizes, "x")
    edge = _edge_index(sample, "edge_index", x.size(0))
    edge_attr = _tensor(sample, "edge_attr")
    if edge_attr.shape != (edge.size(1), 5):
        raise ValueError("edge_attr must have shape [E, 5]")
    _categorical(edge_attr, bond_sizes, "edge_attr")

    protein = _tensor(sample, "prot_node_x")
    if protein.ndim != 2 or protein.size(0) < 1 or protein.size(1) not in (28, 31):
        raise ValueError("prot_node_x must have shape [M, 28] or [M, 31] with M >= 1")
    if not protein.is_floating_point():
        raise ValueError("prot_node_x must be floating point")
    if bool(((protein[:, :28] < -1e-6) | (protein[:, :28] > 1.000001)).any()):
        raise ValueError("The first 28 protein node features must lie in [0, 1]")
    protein_edges = _edge_index(sample, "prot_edge_index", protein.size(0))
    protein_edge_attr = _tensor(sample, "prot_edge_attr")
    if protein_edge_attr.shape != (protein_edges.size(1), 5):
        raise ValueError("prot_edge_attr must have shape [F, 5]")
    if not protein_edge_attr.is_floating_point():
        raise ValueError("prot_edge_attr must be floating point")
    if protein_edge_attr.numel():
        if bool((protein_edge_attr[:, (0, 4)] < 0).any()):
            raise ValueError("Protein distance and sequence-distance features cannot be negative")
        if bool((protein_edge_attr[:, 1:4].abs() > 1.0001).any()):
            raise ValueError("Protein edge direction components must lie in [-1, 1]")
        if bool((protein_edge_attr[:, 4] > 1.000001).any()):
            raise ValueError("Normalized protein sequence distances must lie in [0, 1]")

    for name, width in (("molt5_emb", 768), ("target", 1024)):
        value = _tensor(sample, name)
        if value.shape != (1, width) or not value.is_floating_point():
            raise ValueError(f"{name} must be floating point with shape [1, {width}]")
    for name in CONCEPT_NAMES:
        value = _tensor(sample, name)
        if value.shape != (1,) or not value.is_floating_point():
            raise ValueError(f"{name} must be floating point with shape [1]")
        if name != "vina_score" and float(value.item()) < 0:
            raise ValueError(f"{name} cannot be negative")
        if name != "vina_score" and float(value.item()) != int(value.item()):
            raise ValueError(f"{name} must contain an integer count")
    if require_label:
        y = _tensor(sample, "y")
        if y.shape != (1,) or float(y.item()) not in (0.0, 1.0):
            raise ValueError("y must have shape [1] and contain a binary label")
    for name, nodes in (("pestat_RWSE", x.size(0)), ("prot_pestat_RWSE", protein.size(0))):
        rwse = getattr(sample, name, None)
        if rwse is not None:
            rwse = _tensor(sample, name)
            if rwse.shape != (nodes, len(RWSE_STEPS)) or not rwse.is_floating_point():
                raise ValueError(f"{name} must be floating point with shape [{nodes}, 5]")
            if bool(((rwse < -1e-6) | (rwse > 1.000001)).any()):
                raise ValueError(f"{name} contains a probability outside [0, 1]")


def validate_dataset(samples: Sequence[Data], *, require_labels: bool = True) -> dict[str, int]:
    """Validate every pair and return an input inventory."""
    if not samples:
        raise ValueError("The dataset is empty")
    positives = 0
    atoms = 0
    residues = 0
    for index, sample in enumerate(samples):
        try:
            validate_sample(sample, require_label=require_labels)
        except (ValueError, TypeError, IndexError) as exc:
            raise ValueError(f"Sample {index}: {exc}") from exc
        atoms += sample.x.size(0)
        residues += sample.prot_node_x.size(0)
        if require_labels:
            positives += int(sample.y.item())
    inventory = {"n_samples": len(samples), "n_atoms": atoms, "n_residues": residues}
    if require_labels:
        inventory.update(n_positive=positives, n_negative=len(samples) - positives)
    return inventory


def compute_rwse(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    steps: Sequence[int] = RWSE_STEPS,
    max_dense_nodes: int = 2048,
    block_size: int = 64,
) -> torch.Tensor:
    """Exact random-walk return probabilities on an undirected graph.

    Binary edges are coalesced before degree normalization using undirected
    connectivity. Existing self-loops are retained.
    Propagating blocks of basis vectors through a sparse transition matrix
    avoids dense N-by-N adjacency/power matrices and sparse-power fill-in.
    State storage is O(N * block_size). ``max_dense_nodes`` is retained as
    the graph-size guard name for API compatibility.
    """
    if num_nodes < 0 or num_nodes > max_dense_nodes:
        raise ValueError(f"num_nodes must be between 0 and {max_dense_nodes}")
    if not steps or any(int(step) != step or step < 1 for step in steps):
        raise ValueError("RWSE steps must be positive integers")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be int64 with shape [2, E]")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes):
        raise ValueError("edge_index contains an out-of-range node index")
    output = torch.zeros((num_nodes, len(steps)), dtype=torch.float32, device=edge_index.device)
    if num_nodes == 0 or edge_index.numel() == 0:
        return output
    indices = torch.unique(torch.cat((edge_index, edge_index.flip(0)), dim=1), dim=1)
    source = indices[0]
    degrees = torch.bincount(source, minlength=num_nodes).to(torch.float64)
    weights = degrees[source].reciprocal()
    transition = torch.sparse_coo_tensor(
        indices, weights, (num_nodes, num_nodes), device=edge_index.device
    ).coalesce()
    requested = set(int(step) for step in steps)
    columns_by_step = {step: [i for i, value in enumerate(steps) if int(value) == step] for step in requested}
    for start in range(0, num_nodes, block_size):
        end = min(start + block_size, num_nodes)
        rows = torch.arange(start, end, device=edge_index.device)
        columns = torch.arange(end - start, device=edge_index.device)
        state = torch.zeros((num_nodes, end - start), dtype=torch.float64, device=edge_index.device)
        state[rows, columns] = 1.0
        for step in range(1, max(requested) + 1):
            state = torch.sparse.mm(transition, state)
            if step in requested:
                diagonal = state[rows, columns].to(torch.float32)
                for column in columns_by_step[step]:
                    output[start:end, column] = diagonal
    return output


def as_dual_graph(
    sample: Data | Mapping[str, Any],
    *,
    add_rwse: bool = True,
    add_protein_rwse: bool = False,
    protein_rwse_max_nodes: int = 4096,
) -> DualGraphData:
    """Copy and normalize legacy scalar/vector tensor shapes."""
    if isinstance(sample, Data):
        values = sample.to_dict()
    elif isinstance(sample, Mapping):
        values = dict(sample)
    else:
        raise ValueError("A sample must be a PyG Data object or a field dictionary")
    # Batching reconstructs protein membership through follow_batch. A cached
    # per-sample all-zero prot_batch must not override that membership.
    for name in ("batch", "ptr", "prot_batch", "prot_node_x_batch", "prot_node_x_ptr", "prot_coord", "prot_coords"):
        values.pop(name, None)
    for name in CONCEPT_NAMES + ("y",):
        if name in values:
            value = values[name]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, dtype=torch.float32)
            if value.numel() != 1:
                raise ValueError(f"{name} must contain exactly one value")
            values[name] = value.reshape(1).float()
    for name in ("target", "molt5_emb"):
        value = values.get(name)
        if isinstance(value, torch.Tensor) and value.ndim == 1:
            values[name] = value.unsqueeze(0)
    data = DualGraphData(**values)
    if add_rwse and getattr(data, "pestat_RWSE", None) is None:
        x = getattr(data, "x", None)
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("x must be a two-dimensional tensor before computing RWSE")
        data.pestat_RWSE = compute_rwse(data.edge_index, x.size(0))
    if add_protein_rwse and getattr(data, "prot_pestat_RWSE", None) is None:
        protein = getattr(data, "prot_node_x", None)
        if not isinstance(protein, torch.Tensor) or protein.ndim != 2:
            raise ValueError("prot_node_x must be a two-dimensional tensor before computing protein RWSE")
        data.prot_pestat_RWSE = compute_rwse(
            data.prot_edge_index, protein.size(0), max_dense_nodes=protein_rwse_max_nodes,
        )
    return data


def load_split(
    path: str | Path,
    *,
    validate: bool = True,
    add_rwse: bool = True,
    add_protein_rwse: bool = False,
    protein_rwse_max_nodes: int = 4096,
) -> list[DualGraphData]:
    """Load a trusted local PyTorch split; no data download is performed."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(payload, Data):
        payload = [payload]
    elif isinstance(payload, Mapping) and "samples" in payload:
        payload = payload["samples"]
    if not isinstance(payload, (list, tuple)):
        raise ValueError("The split must contain a list of Data objects (or a 'samples' list)")
    samples = [
        as_dual_graph(sample, add_rwse=add_rwse, add_protein_rwse=add_protein_rwse,
                      protein_rwse_max_nodes=protein_rwse_max_nodes)
        for sample in payload
    ]
    if validate:
        validate_dataset(samples)
    return samples


def make_loader(
    samples: Sequence[Data],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    *,
    add_protein_rwse: bool = False,
    protein_rwse_max_nodes: int = 4096,
) -> DataLoader:
    """Batch with independent protein offsets and membership vectors."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if not samples:
        raise ValueError("Cannot batch an empty dataset")
    converted = [
        as_dual_graph(sample, add_protein_rwse=add_protein_rwse,
                      protein_rwse_max_nodes=protein_rwse_max_nodes)
        for sample in samples
    ]
    return DataLoader(
        converted,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        follow_batch=["prot_node_x"],
    )


def fit_descriptor_stats(train: Sequence[Data]) -> dict[str, dict[str, float]]:
    """Fit population mean/std from the supplied TRAINING partition only."""
    if not train:
        raise ValueError("Cannot fit descriptor statistics on an empty training set")
    result: dict[str, dict[str, float]] = {}
    for name in CONCEPT_NAMES:
        entries = []
        for index, sample in enumerate(train):
            value = _tensor(sample, name)
            if value.numel() != 1:
                raise ValueError(f"Training sample {index}: {name} must contain one value")
            entries.append(value.reshape(()).double().cpu())
        values = torch.stack(entries)
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False).clamp_min(1e-8)),
        }
    return result
