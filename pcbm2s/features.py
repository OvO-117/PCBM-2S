"""Ligand and protein graph feature construction.

Molecules follow create_data_scope_20260406.py/new_mol_to_graph_data.
Residue features/edges follow create_data_scope_20260422_protein_graph.py;
the alternative biochemical table and PDB reader follow
pdbbind_refined/build_pdbbind_train_dataset.py. CompoundKit attribution is
provided in THIRD_PARTY_NOTICES.md and the source copyright header.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from .compound_tools import CompoundKit, new_mol_to_graph_data
from .data import compute_rwse


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: index for index, aa in enumerate(AA_ORDER)}
# Fixed biochemical property tables used by the graph feature schema.
AA_BIOCHEM_7D = {
    "A": [3.854415274463007246e-1, 0., 0., 1., 3.333333333333333703e-1, 3.701467387488588390e-1, 1.843457636052180382e-1],
    "C": [2.684964200477326091e-1, 3.333333333333333148e-1, 0., 1., 3.333333333333333703e-1, 3.701467387488588390e-1, 1.843457636052180382e-1],
    "D": [0., 6.666666666666666297e-1, 5.e-1, 1., 3.333333333333333703e-1, 2.103840483044302490e-1, 6.758466201080510771e-1],
    "E": [5.369928400954654402e-2, 6.666666666666666297e-1, 5.e-1, 1., 3.333333333333333703e-1, 3.473285122516323042e-1, 6.758466201080510771e-1],
    "F": [3.233890214797136564e-1, 0., 0., 0., 0., 8.310398090289965456e-1, 1.843457636052180382e-1],
    "G": [3.818615751789976032e-1, 3.333333333333333148e-1, 0., 0., 0., 2.653935266446673658e-1, 1.843457636052180382e-1],
    "H": [5.608591885441527314e-1, 1., 1., 1., 3.333333333333333703e-1, 3.827845257319384409e-1, 5.622611674792461489e-1],
    "I": [3.782816229116945927e-1, 0., 0., 0., 0., 7.620234501158462681e-1, 1.843457636052181492e-1],
    "K": [8.138424821002385734e-1, 1., 1., 1., 3.333333333333333703e-1, 4.400758267218984887e-1, 5.272104361575966625e-1],
    "L": [3.830548926014320510e-1, 0., 0., 0., 0., 7.620234501158462681e-1, 1.843457636052181492e-1],
    "M": [3.544152744630071905e-1, 0., 0., 1., 0., 6.591659060591167352e-1, 1.843457636052180382e-1],
    "N": [3.150357995226730767e-1, 3.333333333333333148e-1, 0., 1., 3.333333333333333703e-1, 0., 7.521412570826193633e-1],
    "P": [4.212410501193317169e-1, 0., 0., 0., 0., 5.438812048023590195e-1, 0.],
    "Q": [3.436754176610978262e-1, 3.333333333333333148e-1, 0., 1., 3.333333333333333703e-1, 1.369444639472022773e-1, 7.521412570826193633e-1],
    "R": [1., 1., 1., 1., 9.999999999999998890e-1, 1.361616232535282078e-1, 1.],
    "S": [3.472553699284009476e-1, 3.333333333333333148e-1, 0., 1., 3.333333333333333703e-1, 4.103770273116591483e-2, 4.509157991830281542e-1],
    "T": [3.424821002386633784e-1, 3.333333333333333148e-1, 0., 1., 3.333333333333333703e-1, 1.774204872568978519e-1, 4.509157991830281542e-1],
    "V": [3.806682577565632664e-1, 0., 0., 0., 0., 6.250789861686442128e-1, 1.843457636052181492e-1],
    "W": [3.723150357995225757e-1, 0., 0., 0., 3.333333333333333703e-1, 9.999999999999998890e-1, 3.924100672025299108e-1],
    "Y": [3.448687350835322740e-1, 3.333333333333333148e-1, 0., 1., 3.333333333333333703e-1, 7.276907954784807009e-1, 4.509157991830281542e-1],
}
PDBBIND_BIOCHEM_7D = {
    **AA_BIOCHEM_7D,
    "A": [3.560570932880529733e-1, 0., 0., 0., 0., 6.780644811217975087e-1, 1.843457636052180382e-1],
    "C": [3.960837647293264055e-1, 0., 0., 1., 0., 5.984832365261813725e-1, 5.129548016638159534e-1],
    "D": [3.294748576389944272e-1, 3.333333333333333148e-1, 1., 1., 3.333333333333333703e-1, 0., 8.928850187423557664e-1],
    "E": [4.056324181626187109e-1, 3.333333333333333148e-1, 1., 1., 3.333333333333333703e-1, 1.369444639472022773e-1, 7.521412570826193633e-1],
}


def build_ligand_graph(smiles: str, *, add_rwse: bool = True) -> Data:
    """Build the 21-column ligand graph.

    CompoundKit uses unscaled atomic mass,
    computes Gasteiger charges, and appends one self-loop per atom.
    """
    from rdkit import Chem

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a nonempty string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("SMILES did not produce a nonempty RDKit molecule")
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise ValueError("Wildcard atoms are not supported by the released feature schema")
    features = new_mol_to_graph_data(mol)
    if features is None:
        raise ValueError("CompoundKit could not build the molecule features")
    atom_keys = list(CompoundKit.atom_vocab_dict)
    float_keys = list(CompoundKit.atom_float_names)
    bond_keys = list(CompoundKit.bond_vocab_dict)
    int_values = np.stack([features[key] for key in atom_keys], axis=1)
    float_values = np.stack([features[key] for key in float_keys], axis=1)
    x = torch.cat((torch.as_tensor(int_values, dtype=torch.long),
                   torch.as_tensor(float_values, dtype=torch.float32)), dim=1)
    edge_index = torch.as_tensor(features["edges"].T.copy(), dtype=torch.long)
    edge_attr = torch.as_tensor(np.stack([features[key] for key in bond_keys], axis=1), dtype=torch.float32)
    if x.shape[1] != 21 or edge_attr.shape[1] != 5:
        raise ValueError("CompoundKit feature vocabulary does not match the released model")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Molecule features contain nonfinite values")
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles_str=smiles)
    if add_rwse:
        graph.pestat_RWSE = compute_rwse(edge_index, x.size(0))
    return graph


def build_protein_graph(
    sequence: str,
    coords: Any,
    *,
    node_dim: int = 28,
    biochem_profile: str = "scope",
    radius: float = 10.0,
    max_num_neighbors: int | None = 32,
    add_seq_edges: bool = True,
    add_rwse: bool = True,
    max_rwse_nodes: int = 4096,
) -> dict[str, Any]:
    """Build residue features and directed radius/sequence edges.

    `coords` contains one aligned C-alpha position per sequence character.
    Node dimensions: 20 one-hot + 7 properties + relative position, with
    optional 3 standardized coordinates for the legacy 31-column format.
    """
    if not isinstance(sequence, str) or not sequence or not sequence.isalpha() or sequence != sequence.upper():
        raise ValueError("sequence must be a nonempty uppercase amino-acid string")
    if node_dim not in (28, 31):
        raise ValueError("node_dim must be 28 or 31")
    if biochem_profile not in ("scope", "pdbbind"):
        raise ValueError("biochem_profile must be scope or pdbbind")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    if max_num_neighbors is not None and max_num_neighbors < 1:
        raise ValueError("max_num_neighbors must be positive or None")
    coordinates = torch.as_tensor(np.asarray(coords), dtype=torch.float32).clone()
    if coordinates.ndim == 3 and coordinates.size(0) == 1:
        coordinates = coordinates.squeeze(0)
    length = len(sequence)
    if coordinates.shape != (length, 3) or not bool(torch.isfinite(coordinates).all()):
        raise ValueError("coords must be finite with shape [len(sequence), 3]")
    table = AA_BIOCHEM_7D if biochem_profile == "scope" else PDBBIND_BIOCHEM_7D
    onehot = torch.zeros((length, 20), dtype=torch.float32)
    for index, aa in enumerate(sequence):
        if aa in AA_TO_IDX:
            onehot[index, AA_TO_IDX[aa]] = 1.0
    biochemical = torch.tensor([table.get(aa, [0.] * 7) for aa in sequence], dtype=torch.float32)
    position = (torch.arange(length, dtype=torch.float32) / float(length - 1)).unsqueeze(1) if length > 1 else torch.zeros((length, 1))
    node_features = torch.cat((onehot, biochemical, position), dim=1)
    if node_dim == 31:
        if length < 2:
            raise ValueError("Legacy 31-column coordinate normalization requires at least two residues")
        normalized = (coordinates - coordinates.mean(dim=0, keepdim=True)) / (coordinates.std(dim=0, keepdim=True) + 1e-8)
        node_features = torch.cat((node_features, normalized), dim=1)

    # Keep the original numerical predicate, including its self-loop behavior:
    # sqrt(0 + 1e-12) > 1e-8, so diagonal candidates survive the radius filter.
    difference = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)
    distances = torch.sqrt(torch.sum(difference * difference, dim=-1) + 1e-12)
    sources: list[int] = []
    targets: list[int] = []
    attributes: list[list[float]] = []

    def append_edge(source: int, target: int) -> None:
        displacement = coordinates[target] - coordinates[source]
        distance = torch.norm(displacement) + 1e-12
        unit = displacement / distance
        sources.append(source)
        targets.append(target)
        attributes.append([float(distance), float(unit[0]), float(unit[1]), float(unit[2]), float(abs(target - source))])

    for source in range(length):
        neighbors = torch.where((distances[source] < radius) & (distances[source] > 1e-8))[0]
        if max_num_neighbors is not None and neighbors.numel() > max_num_neighbors:
            neighbors = neighbors[torch.argsort(distances[source, neighbors])[:max_num_neighbors]]
        for target in neighbors.tolist():
            append_edge(source, target)
    if add_seq_edges and length > 1:
        existing = set(zip(sources, targets))
        for index in range(length - 1):
            for source, target in ((index, index + 1), (index + 1, index)):
                if (source, target) not in existing:
                    append_edge(source, target)
    if not sources:
        edge_index = torch.arange(length, dtype=torch.long).unsqueeze(0).repeat(2, 1)
        edge_attr = torch.zeros((length, 5), dtype=torch.float32)
    else:
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(attributes, dtype=torch.float32)
        edge_attr[:, 0] /= radius
        if length > 1:
            edge_attr[:, 4] /= float(length - 1)
    output: dict[str, Any] = {
        "prot_node_x": node_features,
        "prot_edge_index": edge_index,
        "prot_edge_attr": edge_attr,
        "sequence_str": sequence,
    }
    if add_rwse:
        output["prot_pestat_RWSE"] = compute_rwse(edge_index, length, max_dense_nodes=max_rwse_nodes)
    return output


def protein_graph_from_pdb(path: str | Path, *, chain: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Read standard amino acids in the first PDB model, retaining residue IDs.

    Default iteration follows all chains in file order, matching the retained
    preparation script. Missing C-alpha atoms raise rather than silently
    deleting residues and shifting the sequence/coordinate alignment.
    """
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.Polypeptide import protein_letters_3to1
    except ImportError as exc:
        raise ImportError("PDB preparation requires biopython; install pcbm2s[features]") from exc
    structure = PDBParser(QUIET=True).get_structure("protein", str(path))
    model = next(iter(structure), None)
    if model is None:
        raise ValueError("PDB contains no model")
    sequence, coordinates, residue_ids = [], [], []
    for current_chain in model:
        if chain is not None and current_chain.id != chain:
            continue
        for residue in current_chain:
            name = residue.get_resname()
            if name not in protein_letters_3to1:
                continue
            if "CA" not in residue:
                raise ValueError(f"Missing C-alpha atom at chain {current_chain.id}, residue {residue.id}")
            sequence.append(protein_letters_3to1[name])
            coordinates.append(np.asarray(residue["CA"].coord, dtype=np.float32))
            residue_ids.append({"chain": current_chain.id, "number": residue.id[1], "insertion_code": residue.id[2].strip(), "resname": name})
    if not sequence:
        raise ValueError("No standard amino acids with C-alpha coordinates were found")
    output = build_protein_graph("".join(sequence), np.stack(coordinates), **kwargs)
    output["residue_ids"] = residue_ids
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ligand or protein graph features")
    commands = parser.add_subparsers(dest="kind", required=True)
    ligand = commands.add_parser("ligand", help="Build a ligand graph from SMILES")
    ligand.add_argument("--smiles", required=True)
    ligand.add_argument("--output", type=Path, required=True)
    ligand.add_argument("--no-rwse", action="store_true")
    protein = commands.add_parser("protein", help="Build a residue graph from a PDB")
    protein.add_argument("--pdb", type=Path, required=True)
    protein.add_argument("--output", type=Path, required=True)
    protein.add_argument("--chain", default=None)
    protein.add_argument("--node-dim", type=int, choices=(28, 31), default=28)
    protein.add_argument("--biochem-profile", choices=("scope", "pdbbind"), default="scope")
    protein.add_argument("--radius", type=float, default=10.)
    protein.add_argument("--max-neighbors", type=int, default=32)
    protein.add_argument("--no-sequence-edges", action="store_true")
    protein.add_argument("--no-rwse", action="store_true")
    args = parser.parse_args()
    if args.kind == "ligand":
        result = build_ligand_graph(args.smiles, add_rwse=not args.no_rwse)
    else:
        result = protein_graph_from_pdb(args.pdb, chain=args.chain, node_dim=args.node_dim,
            biochem_profile=args.biochem_profile, radius=args.radius,
            max_num_neighbors=args.max_neighbors, add_seq_edges=not args.no_sequence_edges,
            add_rwse=not args.no_rwse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"Saved {args.kind} graph to {args.output}")


if __name__ == "__main__":
    main()
