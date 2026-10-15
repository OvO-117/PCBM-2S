"""Normalize and validate externally supplied, trusted PyTorch pair files.

Example:
    python -m pcbm2s.preprocess normalize --input legacy_train.pt \
        --output data/train.pt --split-name train --stats-output data/stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
from torch_geometric.data import Data

from .data import as_dual_graph, fit_descriptor_stats, load_split, validate_dataset


def _write_target(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; use --overwrite explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_split(
    source: str | Path,
    destination: str | Path,
    *,
    protein_bank: str | Path | None = None,
    protein_rwse: bool = False,
    protein_rwse_max_nodes: int = 4096,
    overwrite: bool = False,
) -> list:
    """Materialize legacy slim records and save validated DualGraphData pairs."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output must be different files")
    _write_target(destination, overwrite)
    if protein_bank is None:
        samples = load_split(source, add_protein_rwse=protein_rwse,
                             protein_rwse_max_nodes=protein_rwse_max_nodes)
    else:
        payload = torch.load(source, map_location="cpu", weights_only=False)
        bank = torch.load(Path(protein_bank), map_location="cpu", weights_only=False)
        if isinstance(payload, Mapping) and "samples" in payload:
            payload = payload["samples"]
        if not isinstance(payload, (list, tuple)) or not isinstance(bank, Mapping):
            raise ValueError("Expected a sample list and a protein-bank dictionary")
        samples = []
        for index, sample in enumerate(payload):
            if isinstance(sample, Data):
                values = sample.to_dict()
            elif isinstance(sample, Mapping):
                values = dict(sample)
            else:
                raise ValueError(f"Sample {index} is not a Data object or field dictionary")
            identity = values.get("target_id")
            if identity not in bank:
                raise ValueError(f"Sample {index} has no matching protein-bank entry")
            protein = bank[identity]
            protein = protein.to_dict() if isinstance(protein, Data) else protein
            if not isinstance(protein, Mapping):
                raise ValueError(f"Sample {index}: protein-bank entry must be a dictionary")
            for name in ("prot_node_x", "prot_edge_index", "prot_edge_attr"):
                if name not in protein:
                    raise ValueError(f"Sample {index}: protein-bank entry lacks {name}")
                values[name] = protein[name]
            if "prot_pestat_RWSE" in protein:
                values["prot_pestat_RWSE"] = protein["prot_pestat_RWSE"]
            samples.append(as_dual_graph(values, add_protein_rwse=protein_rwse,
                                         protein_rwse_max_nodes=protein_rwse_max_nodes))
        validate_dataset(samples)
    torch.save(samples, destination)
    return samples


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    normalize = commands.add_parser("normalize", help="Normalize trusted legacy .pt input")
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--protein-bank", type=Path)
    normalize.add_argument("--split-name", choices=("train", "val", "test"))
    normalize.add_argument("--stats-output", type=Path, help="Training-only descriptor statistics")
    normalize.add_argument("--protein-rwse", action="store_true", help="Compute missing protein RWSE")
    normalize.add_argument("--protein-rwse-max-nodes", type=int, default=4096)
    normalize.add_argument("--overwrite", action="store_true")
    validate = commands.add_parser("validate", help="Validate an existing trusted .pt split")
    validate.add_argument("--input", type=Path, required=True)
    arguments = list(sys.argv[1:] if argv is None else argv)
    # A plain --input/--output invocation is the convenient normalization entry.
    # Explicit subcommands remain supported for reproducible existing scripts.
    if arguments and arguments[0].startswith("--") and arguments[0] != "--help":
        arguments.insert(0, "normalize")
    args = parser.parse_args(arguments)
    if args.command == "validate":
        samples = load_split(args.input)
    else:
        if args.stats_output is not None:
            if args.split_name != "train":
                parser.error("--stats-output is allowed only with --split-name train")
            if args.stats_output.resolve() in (args.input.resolve(), args.output.resolve()):
                parser.error("--stats-output must differ from input and output")
            _write_target(args.stats_output, args.overwrite)
        samples = normalize_split(
            args.input, args.output,
            protein_bank=args.protein_bank, overwrite=args.overwrite,
            protein_rwse=args.protein_rwse,
            protein_rwse_max_nodes=args.protein_rwse_max_nodes,
        )
        if args.stats_output is not None:
            args.stats_output.write_text(
                json.dumps(fit_descriptor_stats(samples), indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    print(json.dumps(validate_dataset(samples), indent=2))


if __name__ == "__main__":
    main()
