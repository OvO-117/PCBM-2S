"""Offline checks for chemistry schema, residue construction and T5 pooling."""

from types import SimpleNamespace

import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch

from pcbm2s.compound_tools import CompoundKit, get_bond_feature_dims
from pcbm2s.embeddings import encode_with_loaded_model, masked_mean_pool, prepare_encoder_texts
from pcbm2s.features import AA_BIOCHEM_7D, PDBBIND_BIOCHEM_7D, build_ligand_graph, build_protein_graph, protein_graph_from_pdb


class FeatureTests(unittest.TestCase):
    def test_ligand_schema_charge_mass_and_self_loops(self):
        from rdkit import Chem

        graph = build_ligand_graph("CCO")
        assert graph.x.shape == (3, 21)
        assert graph.edge_index.shape == (2, 7)  # two bonds in both directions + 3 self-loops
        assert graph.edge_attr.shape == (7, 5)
        assert torch.equal(graph.edge_index[:, -3:], torch.arange(3).repeat(2, 1))
        bond_keys = list(CompoundKit.bond_vocab_dict)
        expected_self = torch.tensor([get_bond_feature_dims([key])[0] - 1 for key in bond_keys])
        assert torch.equal(graph.edge_attr[-1].long(), expected_self)
        atoms = list(Chem.MolFromSmiles("CCO").GetAtoms())
        assert torch.allclose(graph.x[:, 20], torch.tensor([atom.GetMass() for atom in atoms]))
        assert graph.x[:, 20].min() > 1  # the active new feature path does not multiply mass by 0.01
        assert graph.x[:, 19].abs().sum() > 0  # computed Gasteiger charges
        assert graph.pestat_RWSE.shape == (3, 5)
        assert bool(torch.isfinite(graph.x).all())


    def test_single_atom_still_has_bond_feature_self_loop(self):
        graph = build_ligand_graph("[Na+]")
        assert graph.x.shape == (1, 21)
        assert graph.edge_index.tolist() == [[0], [0]]
        assert torch.allclose(graph.pestat_RWSE, torch.ones(1, 5))
        with self.assertRaises(ValueError):
            build_ligand_graph("not a smiles")


    def test_protein_feature_order_legacy_coordinates_and_profiles(self):
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        coords = torch.arange(60, dtype=torch.float32).reshape(20, 3)
        graph = build_protein_graph(sequence, coords, add_rwse=False)
        node = graph["prot_node_x"]
        assert node.shape == (20, 28)
        assert torch.equal(node[:, :20], torch.eye(20))
        assert torch.equal(node[:, 20:27], torch.tensor([AA_BIOCHEM_7D[aa] for aa in sequence]))
        assert torch.allclose(node[:, 27], torch.linspace(0, 1, 20))
        legacy = build_protein_graph(sequence, coords, node_dim=31, add_rwse=False)
        assert torch.equal(legacy["prot_node_x"][:, :28], node)
        expected_coords = (coords - coords.mean(0, keepdim=True)) / (coords.std(0, keepdim=True) + 1e-8)
        assert torch.equal(legacy["prot_node_x"][:, 28:], expected_coords)
        alternate = build_protein_graph(sequence, coords, biochem_profile="pdbbind", add_rwse=False)
        assert torch.equal(alternate["prot_node_x"][:, 20:27], torch.tensor([PDBBIND_BIOCHEM_7D[aa] for aa in sequence]))
        assert not torch.equal(alternate["prot_node_x"][:4, 20:27], node[:4, 20:27])


    def test_protein_edge_units_sequence_completion_and_original_diagonal(self):
        coords = [[0., 0., 0.], [3., 0., 0.], [20., 0., 0.]]
        graph = build_protein_graph("ACD", coords, radius=10., add_rwse=True)
        edges = list(zip(*graph["prot_edge_index"].tolist()))
        assert all((index, index) in edges for index in range(3))
        assert (1, 2) in edges and (2, 1) in edges  # sequence edge added beyond radius
        attr = graph["prot_edge_attr"][edges.index((0, 1))]
        assert torch.allclose(attr, torch.tensor([.3, 1., 0., 0., .5]))
        far_attr = graph["prot_edge_attr"][edges.index((1, 2))]
        assert abs(float(far_attr[0]) - 1.7) < 1e-6
        assert graph["prot_pestat_RWSE"].shape == (3, 5)
        without_sequence = build_protein_graph("ACD", coords, radius=10., add_seq_edges=False, add_rwse=False)
        assert (1, 2) not in list(zip(*without_sequence["prot_edge_index"].tolist()))


    def test_protein_invalid_alignment_and_single_residue(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            build_protein_graph("AC", [[0., 0., 0.]])
        with self.assertRaisesRegex(ValueError, "at least two"):
            build_protein_graph("A", [[0., 0., 0.]], node_dim=31)
        single = build_protein_graph("X", [[0., 0., 0.]])
        assert single["prot_node_x"].shape == (1, 28)
        assert torch.equal(single["prot_node_x"], torch.zeros(1, 28))


    def test_pdb_reader_preserves_sequence_and_residue_ids(self):
        if importlib.util.find_spec("Bio") is None:
            self.skipTest("Biopython is not installed")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        lines = [
            "ATOM      1  CA  ALA A   7       0.000   0.000   0.000  1.00 20.00           C  ",
            "ATOM      2  CA  CYS A   8       3.000   0.000   0.000  1.00 20.00           C  ",
            "ATOM      3  CA  ASP B  21       6.000   0.000   0.000  1.00 20.00           C  ",
            "END",
        ]
        path = tmp_path / "synthetic.pdb"
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        graph = protein_graph_from_pdb(path)
        assert graph["sequence_str"] == "ACD"
        assert graph["residue_ids"][2]["chain"] == "B"
        assert graph["residue_ids"][2]["number"] == 21
        assert protein_graph_from_pdb(path, chain="A")["sequence_str"] == "AC"


    def test_prott5_text_processing_and_masked_eos_pooling(self):
        assert prepare_encoder_texts(["AUZOB"], "prot5") == ["A X X X X"]
        sequence = "A" * 500 + "C" * 7 + "D" * 500
        assert prepare_encoder_texts([sequence], "prot5") == [" ".join("A" * 500 + "D" * 500)]
        assert prepare_encoder_texts(["C[C@H](O)N"], "molt5") == ["C[C@H](O)N"]
        hidden = torch.tensor([[[1.], [3.], [11.], [999.]]])
        # Third attended vector represents EOS; it is intentionally included.
        assert abs(masked_mean_pool(hidden, torch.tensor([[1, 1, 1, 0]])).item() - 5.) < 1e-6


    def test_encoder_batches_without_transformers_or_network(self):
        class Tokenizer:
            def __init__(self):
                self.calls = []

            def batch_encode_plus(self, values, **kwargs):
                self.calls.append((values, kwargs))
                return {"input_ids": torch.tensor([[1, 3, 9]] * len(values)), "attention_mask": torch.tensor([[1, 1, 0]] * len(values))}

        class Encoder:
            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, input_ids, attention_mask):
                return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1).expand(-1, -1, 4))

        tokenizer = Tokenizer()
        features = encode_with_loaded_model(["AC", "DE", "FG"], kind="prot5", tokenizer=tokenizer,
            model=Encoder(), batch_size=2, device="cpu", max_length=1024)
        assert features.shape == (3, 4)
        assert torch.equal(features, torch.full((3, 4), 2.))
        assert len(tokenizer.calls) == 2
        assert tokenizer.calls[0][0] == ["A C", "D E"]
        assert tokenizer.calls[0][1]["add_special_tokens"] is True


if __name__ == "__main__":
    unittest.main()
