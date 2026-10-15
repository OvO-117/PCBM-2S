"""Synthetic input tests; no Mamba, model weights, or research data required."""

import tempfile
import unittest
import contextlib
import io
from pathlib import Path

import torch
from torch_geometric.data import Data

from pcbm2s.data import (
    CONCEPT_NAMES,
    DualGraphData,
    as_dual_graph,
    compute_rwse,
    fit_descriptor_stats,
    load_split,
    make_loader,
    validate_dataset,
    validate_sample,
)
from pcbm2s.preprocess import main as preprocess_main, normalize_split


def synthetic_sample(atoms=3, residues=5, label=1):
    ligand_nodes = torch.arange(atoms, dtype=torch.long)
    protein_nodes = torch.arange(residues, dtype=torch.long)
    values = {
        "x": torch.zeros((atoms, 21)),
        "edge_index": torch.stack((ligand_nodes, ligand_nodes)),
        "edge_attr": torch.zeros((atoms, 5), dtype=torch.long),
        "prot_node_x": torch.zeros((residues, 28)),
        "prot_edge_index": torch.stack((protein_nodes, protein_nodes)),
        "prot_edge_attr": torch.zeros((residues, 5)),
        "molt5_emb": torch.zeros((1, 768)),
        "target": torch.zeros((1, 1024)),
        "y": torch.tensor([float(label)]),
        "target_id": f"synthetic_{residues}",
        "smiles_str": "synthetic-placeholder",
    }
    values.update({name: torch.tensor([-5.0 if name == "vina_score" else 2.0]) for name in CONCEPT_NAMES})
    return DualGraphData(**values)


class DataTests(unittest.TestCase):
    def test_validation_and_inventory(self):
        samples = [synthetic_sample(label=0), synthetic_sample(label=1)]
        self.assertEqual(validate_dataset(samples)["n_positive"], 1)

    def test_bad_categories_rejected(self):
        sample = synthetic_sample()
        sample.x[0, 0] = 10000
        with self.assertRaisesRegex(ValueError, "category"):
            validate_sample(sample)
        sample = synthetic_sample()
        sample.x[0, 0] = 0.25
        with self.assertRaisesRegex(ValueError, "integer category"):
            validate_sample(sample)

    def test_bad_edges_and_nans_rejected(self):
        sample = synthetic_sample()
        sample.prot_edge_index[0, 0] = sample.prot_node_x.size(0)
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_sample(sample)
        sample = synthetic_sample()
        sample.target[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN"):
            validate_sample(sample)

    def test_independent_protein_batch_offsets(self):
        samples = [synthetic_sample(atoms=2, residues=5), synthetic_sample(atoms=3, residues=7)]
        batch = next(iter(make_loader(samples, batch_size=2, shuffle=False)))
        self.assertTrue(torch.equal(batch.prot_edge_index[:, 5:], samples[1].prot_edge_index + 5))
        self.assertEqual(batch.prot_node_x_batch.tolist(), [0] * 5 + [1] * 7)
        self.assertEqual(batch.batch.tolist(), [0] * 2 + [1] * 3)

    def test_rwse_two_node_walk_and_isolate(self):
        edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        rwse = compute_rwse(edge, 3)
        self.assertTrue(torch.equal(rwse[:2], torch.tensor([[0., 1., 1., 1., 1.]]).repeat(2, 1)))
        self.assertTrue(torch.equal(rwse[2], torch.zeros(5)))
        with self.assertRaisesRegex(ValueError, "between"):
            compute_rwse(edge, 3000)

    def test_block_rwse_matches_dense_reference(self):
        nodes = 85
        source = torch.arange(nodes)
        destination = (source + 1) % nodes
        edges = torch.stack((torch.cat((source, destination)), torch.cat((destination, source))))
        observed = compute_rwse(edges, nodes, block_size=16)
        adjacency = torch.zeros((nodes, nodes), dtype=torch.float64)
        adjacency[edges[0], edges[1]] = 1
        transition = adjacency / adjacency.sum(1, keepdim=True)
        expected = torch.stack([transition.matrix_power(step).diagonal() for step in (1, 2, 4, 8, 16)], dim=1)
        self.assertTrue(torch.allclose(observed.double(), expected, atol=1e-7))

    def test_optional_protein_rwse_generation_and_validation(self):
        original = synthetic_sample()
        without = as_dual_graph(original)
        self.assertIsNone(getattr(without, "prot_pestat_RWSE", None))
        with_rwse = as_dual_graph(original, add_protein_rwse=True)
        self.assertEqual(with_rwse.prot_pestat_RWSE.shape, (5, 5))
        self.assertTrue(torch.equal(with_rwse.prot_pestat_RWSE, torch.ones((5, 5))))
        validate_sample(with_rwse)
        with_rwse.prot_pestat_RWSE = torch.ones((4, 5))
        with self.assertRaisesRegex(ValueError, "prot_pestat_RWSE"):
            validate_sample(with_rwse)
        with_rwse.prot_pestat_RWSE = torch.ones((5, 5)) * 1.1
        with self.assertRaisesRegex(ValueError, "probability"):
            validate_sample(with_rwse)

    def test_legacy_conversion_adds_rwse_and_removes_cached_batches(self):
        sample = synthetic_sample()
        sample.molt5_emb = sample.molt5_emb.squeeze(0)
        sample.prot_batch = torch.zeros(5, dtype=torch.long)
        converted = as_dual_graph(Data(**sample.to_dict()))
        self.assertEqual(converted.molt5_emb.shape, (1, 768))
        self.assertEqual(converted.pestat_RWSE.shape, (3, 5))
        self.assertIsNone(getattr(converted, "prot_batch", None))
        self.assertTrue(torch.equal(converted.pestat_RWSE, torch.ones((3, 5))))

    def test_stats_are_population_train_statistics(self):
        train = [synthetic_sample(), synthetic_sample()]
        for name in CONCEPT_NAMES:
            setattr(train[0], name, torch.tensor([1.0]))
            setattr(train[1], name, torch.tensor([3.0]))
        stats = fit_descriptor_stats(train)
        self.assertEqual(stats["vina_score"], {"mean": 2.0, "std": 1.0})
        for name in CONCEPT_NAMES:
            setattr(train[1], name, torch.tensor([1.0]))
        self.assertGreater(fit_descriptor_stats(train)["vina_score"]["std"], 0)

    def test_save_load_and_slim_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            original = synthetic_sample()
            torch.save([original], directory / "full.pt")
            self.assertIsInstance(load_split(directory / "full.pt")[0], DualGraphData)
            protein = {name: getattr(original, name) for name in ("prot_node_x", "prot_edge_index", "prot_edge_attr")}
            slim = original.to_dict()
            for name in protein:
                slim.pop(name)
            torch.save([slim], directory / "slim.pt")
            torch.save({original.target_id: protein}, directory / "bank.pt")
            result = normalize_split(directory / "slim.pt", directory / "normalized.pt", protein_bank=directory / "bank.pt")
            validate_sample(result[0])
            self.assertTrue((directory / "normalized.pt").is_file())
            with self.assertRaises(FileExistsError):
                normalize_split(directory / "full.pt", directory / "normalized.pt")

    def test_preprocess_default_entry_and_protein_rwse(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            torch.save([synthetic_sample()], directory / "input.pt")
            with contextlib.redirect_stdout(io.StringIO()):
                preprocess_main([
                    "--input", str(directory / "input.pt"),
                    "--output", str(directory / "output.pt"),
                    "--protein-rwse", "--split-name", "train",
                    "--stats-output", str(directory / "stats.json"),
                ])
            loaded = load_split(directory / "output.pt")
            self.assertEqual(loaded[0].prot_pestat_RWSE.shape, (5, 5))
            self.assertTrue((directory / "stats.json").is_file())


if __name__ == "__main__":
    unittest.main()
