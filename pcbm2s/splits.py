"""Deterministic, identity-grouped DTI split utilities (no model imports)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from rdkit import Chem


PARTITIONS = ("train", "val", "test")
PROTOCOLS = ("random-pair", "protein-cold", "drug-cold", "double-cold")


def normalize_sequence(sequence: str) -> str:
    """Uppercase and remove whitespace without collapsing U/Z/O/B to X."""
    if not isinstance(sequence, str):
        raise ValueError("sequence must be a nonempty amino-acid string")
    normalized = "".join(sequence.split()).upper()
    if not normalized or any(character < "A" or character > "Z" for character in normalized):
        raise ValueError("sequence must contain amino-acid letters and optional whitespace only")
    return normalized


def canonical_smiles(smiles: str) -> str:
    """Canonical isomeric SMILES identity, preserving stereochemistry."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a nonempty string")
    text = smiles.strip()
    if any(character.isspace() for character in text):
        raise ValueError("smiles must not include molecule names or internal whitespace")
    molecule = Chem.MolFromSmiles(text)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {text!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _ratios(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("Exactly three train/val/test ratios are required")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise ValueError("Split ratios must be finite and nonnegative")
    if not math.isclose(sum(result), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1")
    return result


def _group_counts(size: int, ratios: Sequence[float]) -> list[int]:
    """Largest remainder, with nonempty positive-ratio groups when feasible."""
    quotas = [size * ratio for ratio in ratios]
    counts = [math.floor(quota) for quota in quotas]
    order = sorted(range(3), key=lambda index: (-(quotas[index] - counts[index]), index))
    for index in order[:size - sum(counts)]:
        counts[index] += 1
    active = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if size >= len(active):
        for index in active:
            if counts[index] == 0:
                donors = [candidate for candidate in active if counts[candidate] > 1]
                donor = max(donors, key=lambda candidate: (counts[candidate] - quotas[candidate], counts[candidate], -candidate))
                counts[donor] -= 1
                counts[index] += 1
    return counts


def _assign(keys: Sequence[Any], ratios: Sequence[float], seed: int, role: str) -> tuple[dict[Any, str], dict[str, int]]:
    unique = sorted(set(keys))
    # Independent named streams keep double-cold ligand/protein assignment
    # independent without depending on Python's process-randomized hash().
    random.Random(f"pcbm2s:{seed}:{role}").shuffle(unique)
    counts = _group_counts(len(unique), ratios)
    assignments: dict[Any, str] = {}
    start = 0
    for partition, count in zip(PARTITIONS, counts):
        for key in unique[start:start + count]:
            assignments[key] = partition
        start += count
    return assignments, dict(zip(PARTITIONS, counts))


def split_pairs(
    frame: pd.DataFrame,
    protocol: str,
    *,
    seed: int = 42,
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    """Split rows without inspecting labels; preserve all original columns.

    The manifest records zero-based input row positions and normalized identity
    inventories. Double-cold excludes rows assigned to different partitions by
    their two entities; all other protocols retain every valid input row.
    """
    ratios = _ratios(ratios)
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    if not isinstance(frame, pd.DataFrame) or not {"smiles", "sequence"}.issubset(frame.columns):
        raise ValueError("Input CSV must include smiles and sequence columns")
    if not frame.columns.is_unique:
        raise ValueError("Input columns must have unique names")
    if frame.empty:
        raise ValueError("Input contains no pairs")
    frame = frame.reset_index(drop=True)
    ligands, proteins = [], []
    for index, row in frame[["smiles", "sequence"]].iterrows():
        try:
            ligands.append(canonical_smiles(row["smiles"]))
            proteins.append(normalize_sequence(row["sequence"]))
        except ValueError as exc:
            raise ValueError(f"Input row {index}: {exc}") from exc
    pairs = list(zip(ligands, proteins))
    assigned_counts: dict[str, dict[str, int]] = {}
    if protocol == "random-pair":
        assigned, counts = _assign(pairs, ratios, seed, "pair")
        assigned_counts["pairs"] = counts
        row_partitions = [assigned[pair] for pair in pairs]
    elif protocol == "protein-cold":
        assigned, counts = _assign(proteins, ratios, seed, "protein")
        assigned_counts["proteins"] = counts
        row_partitions = [assigned[protein] for protein in proteins]
    elif protocol == "drug-cold":
        assigned, counts = _assign(ligands, ratios, seed, "ligand")
        assigned_counts["ligands"] = counts
        row_partitions = [assigned[ligand] for ligand in ligands]
    else:
        drug_groups, assigned_counts["ligands"] = _assign(ligands, ratios, seed, "ligand")
        protein_groups, assigned_counts["proteins"] = _assign(proteins, ratios, seed, "protein")
        row_partitions = [
            drug_groups[ligand] if drug_groups[ligand] == protein_groups[protein] else "excluded"
            for ligand, protein in pairs
        ]
    positions = {
        partition: [index for index, group in enumerate(row_partitions) if group == partition]
        for partition in PARTITIONS + ("excluded",)
    }
    outputs = {partition: frame.iloc[positions[partition]].copy() for partition in PARTITIONS}
    excluded = frame.iloc[positions["excluded"]].copy()
    inventories = {}
    for partition, indices in positions.items():
        inventories[partition] = {
            "rows": len(indices),
            "unique_ligands": len({ligands[index] for index in indices}),
            "unique_proteins": len({proteins[index] for index in indices}),
            "unique_pairs": len({pairs[index] for index in indices}),
        }
    empty = [partition for partition in PARTITIONS if not positions[partition]]
    warnings = []
    if empty:
        warnings.append("Empty partition(s): " + ", ".join(empty) + "; this output is not ready for a full train/validation/test experiment.")
    if protocol == "double-cold":
        warnings.append("Cross-partition pairs are excluded; the retained pair inventory differs from the input.")
    manifest = {
        "schema_version": 1,
        "utility": "pcbm2s.splits",
        "protocol": protocol,
        "seed": int(seed),
        "ratios": dict(zip(PARTITIONS, ratios)),
        "ratio_unit": "unique pairs" if protocol == "random-pair" else "unique held-out identities",
        "identity_rules": {
            "ligand": "RDKit canonical isomeric SMILES; stereochemistry retained",
            "protein": "uppercase sequence with whitespace removed; U/Z/O/B retained without X substitution",
        },
        "assignment": "sorted unique identities; independent named seeded shuffle; largest remainder with nonempty positive-ratio groups when feasible",
        "uses_labels": False,
        "input_rows": len(frame),
        "retained_rows": len(frame) - len(excluded),
        "excluded_rows": len(excluded),
        "retained_fraction": (len(frame) - len(excluded)) / len(frame),
        "assigned_group_counts": assigned_counts,
        "inventories": inventories,
        "row_indices": positions,
        "normalized_identity_sha256": hashlib.sha256(json.dumps(pairs, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest(),
        "training_ready": not empty,
        "warnings": warnings,
        "interpretation": "Identity-disjoint does not imply scaffold-disjoint or homology-disjoint. These generated splits are not claimed to reproduce historical manuscript partitions.",
    }
    return outputs, excluded, manifest


def write_splits(partitions: dict[str, pd.DataFrame], excluded: pd.DataFrame, manifest: dict[str, Any], output_dir: str | Path) -> None:
    """Write a complete split inventory, refusing any existing output file."""
    destination = Path(output_dir)
    names = [f"{partition}.csv" for partition in PARTITIONS] + ["excluded.csv", "manifest.json"]
    existing = [destination / name for name in names if (destination / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite split output: {existing[0]}")
    destination.mkdir(parents=True, exist_ok=True)
    for partition in PARTITIONS:
        with (destination / f"{partition}.csv").open("x", encoding="utf-8", newline="") as handle:
            partitions[partition].to_csv(handle, index=False)
    with (destination / "excluded.csv").open("x", encoding="utf-8", newline="") as handle:
        excluded.to_csv(handle, index=False)
    with (destination / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV containing smiles and sequence columns")
    parser.add_argument("--output-dir", required=True, help="Directory for five new output files")
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    source = Path(args.input)
    frame = pd.read_csv(source, keep_default_na=False)
    partitions, excluded, manifest = split_pairs(frame, args.protocol, seed=args.seed, ratios=args.ratios)
    manifest["input_file"] = source.name
    manifest["input_file_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_splits(partitions, excluded, manifest, args.output_dir)
    for warning in manifest["warnings"]:
        print("Warning: " + warning, file=sys.stderr)
    counts = {name: len(rows) for name, rows in partitions.items()}
    counts["excluded"] = len(excluded)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
