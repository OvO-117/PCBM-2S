"""Five measured interface descriptors from an already prepared pose.

Cutoffs/type sets follow 05_extract_interactions_v4_unidock.py. Polar proximity
is a geometric proxy, not a donor/acceptor- and angle-validated hydrogen bond.
This module does not run docking, generate a pose or estimate a Vina score.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np

POLAR_TYPES = {"N", "O", "S", "NA", "OA", "SA", "NS", "OS"}
HYDROPHOBIC_TYPES = {"C", "A"}


def interface_descriptors(receptor_coords, receptor_residues, receptor_is_polar,
                          ligand_coords, ligand_atom_types, vina_score, *, chunk_size=2048):
    """Count receptor–ligand atom pairs at strict distance cutoffs (angstrom).

    Ligand coordinates and AutoDock atom types MUST have the same atom order.
    Receptor hydrophobic-proxy atoms follow the retained nonpolar mask. All
    receptor atoms passed here should be heavy ATOM records, excluding waters.
    """
    receptor = np.asarray(receptor_coords, dtype=np.float64)
    ligand = np.asarray(ligand_coords, dtype=np.float64)
    polar = np.asarray(receptor_is_polar, dtype=bool)
    if receptor.ndim != 2 or receptor.shape[1] != 3 or not receptor.shape[0]:
        raise ValueError("Receptor coordinates must be nonempty [N,3]")
    if ligand.ndim != 2 or ligand.shape[1] != 3 or not ligand.shape[0]:
        raise ValueError("Ligand coordinates must be nonempty [M,3]")
    if not np.isfinite(receptor).all() or not np.isfinite(ligand).all() or not math.isfinite(float(vina_score)):
        raise ValueError("Coordinates and Vina score must be finite")
    if len(receptor_residues) != len(receptor) or polar.shape != (len(receptor),):
        raise ValueError("Receptor metadata length does not match coordinates")
    if len(ligand_atom_types) != len(ligand):
        raise ValueError("Ligand AutoDock types must match coordinate atom order")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    ligand_polar = np.array([value in POLAR_TYPES for value in ligand_atom_types])
    ligand_hydrophobic = np.array([value in HYDROPHOBIC_TYPES for value in ligand_atom_types])
    contact_residues = set()
    contacts = hbonds = hydrophobic = 0
    for start in range(0, len(receptor), chunk_size):
        part = receptor[start:start + chunk_size]
        distance = np.linalg.norm(part[:, None, :] - ligand[None, :, :], axis=-1)
        contact = distance < 4.0
        contacts += int(contact.sum())
        hbonds += int(((distance < 3.5) & polar[start:start + len(part), None] & ligand_polar[None, :]).sum())
        hydrophobic += int(((distance < 4.5) & ~polar[start:start + len(part), None] & ligand_hydrophobic[None, :]).sum())
        for offset in np.flatnonzero(contact.any(axis=1)):
            contact_residues.add(receptor_residues[start + int(offset)])
    return {"vina_score": float(vina_score), "n_contacts": contacts, "n_hbond": hbonds,
            "n_hydrophobic": hydrophobic, "n_contact_residues": len(contact_residues)}


def receptor_atoms_from_pdb(path):
    """Read heavy ATOM records from the first PDB model, retaining chain IDs.

    Chain/residue/insertion-code keys prevent distinct chains from sharing an
    integer residue index. HETATM metals and water are not counted, matching
    the retained ATOM-only descriptor extractor.
    """
    from Bio.PDB import PDBParser
    structure = PDBParser(QUIET=True).get_structure("receptor", str(path))
    model = next(structure.get_models())
    coordinates, residues, polar = [], [], []
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            for atom in residue:
                element = (atom.element or atom.name[0]).upper()
                if element in {"H", "D"}:
                    continue
                coordinates.append(atom.coord)
                residues.append((chain.id, residue.id[1], residue.id[2]))
                polar.append(element in {"N", "O", "S"})
    return np.asarray(coordinates), residues, np.asarray(polar)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptor", required=True, help="Prepared receptor PDB")
    parser.add_argument("--ligand-geometry", required=True, help="JSON: coordinates [M,3] and atom_types [M], same atom order")
    parser.add_argument("--vina-score", required=True, type=float, help="Precomputed pose-dependent Vina score")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error("Refusing to overwrite existing output")
    payload = json.loads(Path(args.ligand_geometry).read_text(encoding="utf-8"))
    receptor, residues, polar = receptor_atoms_from_pdb(args.receptor)
    result = interface_descriptors(receptor, residues, polar, payload["coordinates"], payload["atom_types"], args.vina_score)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
