import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pcbm2s.splits import canonical_smiles, main, normalize_sequence, split_pairs, write_splits


def grid():
    ligands = ["C", "CC", "CCC", "CO", "CCO", "CCN", "CCCl", "c1ccccc1", "CC(=O)O", "CC(C)C"]
    proteins = ["A" * (index + 1) + "G" for index in range(10)]
    return pd.DataFrame([
        {"smiles": ligand, "sequence": protein, "label": (i + j) % 2, "note": "测试样本", "row_id": i * 10 + j}
        for i, ligand in enumerate(ligands) for j, protein in enumerate(proteins)
    ])


def identities(frame, field):
    function = canonical_smiles if field == "smiles" else normalize_sequence
    return set(frame[field].map(function))


class SplitTests(unittest.TestCase):
    def assert_disjoint(self, sets):
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])

    def test_identity_rules_and_stereochemistry(self):
        self.assertEqual(normalize_sequence(" a U\nz o\tb "), "AUZOB")
        self.assertNotEqual(normalize_sequence("AB"), normalize_sequence("AX"))
        self.assertEqual(canonical_smiles("OCC"), canonical_smiles("CCO"))
        self.assertNotEqual(canonical_smiles("N[C@@H](C)C(=O)O"), canonical_smiles("N[C@H](C)C(=O)O"))
        with self.assertRaisesRegex(ValueError, "Invalid SMILES"):
            canonical_smiles("不是SMILES")
        with self.assertRaises(ValueError):
            normalize_sequence("A*B")

    def test_random_pair_duplicates_stay_together_and_columns_survive(self):
        frame = grid()
        duplicate = frame.iloc[[4]].copy()
        duplicate["smiles"] = "[CH4]"
        duplicate["sequence"] = " " + duplicate["sequence"].str.lower() + "\n"
        duplicate["row_id"] = 100
        duplicate["label"] = 1 - duplicate["label"]
        frame = pd.concat([frame, duplicate], ignore_index=True)
        parts, excluded, manifest = split_pairs(frame, "random-pair")
        pairs = [set(zip(part.smiles.map(canonical_smiles), part.sequence.map(normalize_sequence))) for part in parts.values()]
        self.assert_disjoint(pairs)
        for part in parts.values():
            if 4 in set(part.row_id):
                self.assertIn(100, set(part.row_id))
            self.assertEqual(list(part.columns), list(frame.columns))
            self.assertEqual(set(part.note), {"测试样本"})
        self.assertTrue(excluded.empty)
        self.assertFalse(manifest["uses_labels"])
        self.assertEqual(manifest["retained_rows"], len(frame))

    def test_cold_identity_disjoint(self):
        for protocol, field in [("protein-cold", "sequence"), ("drug-cold", "smiles")]:
            with self.subTest(protocol=protocol):
                parts, excluded, manifest = split_pairs(grid(), protocol)
                self.assert_disjoint([identities(part, field) for part in parts.values()])
                self.assertEqual([len(identities(part, field)) for part in parts.values()], [7, 2, 1])
                self.assertTrue(excluded.empty)
                self.assertTrue(manifest["training_ready"])

    def test_double_cold_excludes_exact_cross_partition_inventory(self):
        frame = grid()
        parts, excluded, manifest = split_pairs(frame, "double-cold")
        self.assert_disjoint([identities(part, "smiles") for part in parts.values()])
        self.assert_disjoint([identities(part, "sequence") for part in parts.values()])
        self.assertEqual([len(part) for part in parts.values()], [49, 4, 1])
        self.assertEqual(len(excluded), 46)
        rows = set().union(*(set(part.row_id) for part in parts.values()))
        self.assertFalse(rows & set(excluded.row_id))
        self.assertEqual(rows | set(excluded.row_id), set(frame.row_id))
        drugs = {value: name for name, part in parts.items() for value in identities(part, "smiles")}
        proteins = {value: name for name, part in parts.items() for value in identities(part, "sequence")}
        for row in excluded.itertuples():
            self.assertNotEqual(drugs[canonical_smiles(row.smiles)], proteins[normalize_sequence(row.sequence)])
        self.assertEqual(manifest["excluded_rows"], 46)
        self.assertEqual(manifest["retained_rows"], 54)

    def test_reproducibility_and_seed_changes(self):
        for protocol in ["random-pair", "protein-cold", "drug-cold", "double-cold"]:
            with self.subTest(protocol=protocol):
                first = split_pairs(grid(), protocol, seed=42)
                second = split_pairs(grid(), protocol, seed=42)
                self.assertEqual(first[2], second[2])
                for name in first[0]:
                    pd.testing.assert_frame_equal(first[0][name], second[0][name])
                changed = split_pairs(grid(), protocol, seed=43)
                self.assertNotEqual(first[2]["row_indices"], changed[2]["row_indices"])

    def test_tiny_input_warns_and_three_groups_cover_positive_partitions(self):
        parts, _, manifest = split_pairs(grid().iloc[[0]], "random-pair")
        self.assertEqual(sum(map(len, parts.values())), 1)
        self.assertFalse(manifest["training_ready"])
        self.assertTrue(any("Empty partition" in warning for warning in manifest["warnings"]))
        parts, _, manifest = split_pairs(grid().iloc[[0, 1, 2]], "random-pair")
        self.assertEqual([len(part) for part in parts.values()], [1, 1, 1])
        self.assertTrue(manifest["training_ready"])

    def test_invalid_ratios(self):
        for ratios in [(0.8, 0.2, 0.2), (-0.1, 0.6, 0.5), (float("nan"), 0.5, 0.5), (0.7, 0.3)]:
            with self.subTest(ratios=ratios):
                with self.assertRaisesRegex(ValueError, "ratio"):
                    split_pairs(grid(), "random-pair", ratios=ratios)

    def test_cli_utf8_output_and_refusal_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "输入.csv"
            destination = Path(temp) / "划分"
            grid().to_csv(source, index=False)
            main(["--input", str(source), "--output-dir", str(destination), "--protocol", "double-cold"])
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["input_file"], source.name)
            self.assertEqual(len(manifest["input_file_sha256"]), 64)
            self.assertTrue((destination / "excluded.csv").exists())
            self.assertEqual(pd.read_csv(destination / "train.csv").note.iloc[0], "测试样本")
            parts, excluded, info = split_pairs(grid(), "double-cold")
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                write_splits(parts, excluded, info, destination)

    def test_held_out_identity_assignment_does_not_use_labels(self):
        frame = grid()
        first = split_pairs(frame, "double-cold")[2]["row_indices"]
        frame["label"] = 1 - frame["label"]
        self.assertEqual(split_pairs(frame, "double-cold")[2]["row_indices"], first)


if __name__ == "__main__":
    unittest.main()
