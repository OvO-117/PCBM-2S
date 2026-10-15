import unittest
import numpy as np
from pcbm2s.descriptors import interface_descriptors


class DescriptorTests(unittest.TestCase):
    def test_proxy_counts_and_chunk_invariance(self):
        receptor = [[0, 0, 0], [5, 0, 0]]
        residues = [("A", 1), ("A", 2)]
        ligand = [[1, 0, 0], [7, 0, 0]]
        first = interface_descriptors(receptor, residues, [True, False], ligand, ["OA", "C"], -7.1, chunk_size=1)
        second = interface_descriptors(receptor, residues, [True, False], ligand, ["OA", "C"], -7.1)
        self.assertEqual(first, second)
        self.assertEqual(first, dict(vina_score=-7.1, n_contacts=2, n_hbond=1, n_hydrophobic=1, n_contact_residues=2))

    def test_strict_threshold(self):
        value = interface_descriptors([[0, 0, 0]], [0], [True], [[3.5, 0, 0]], ["N"], -1)
        self.assertEqual(value["n_hbond"], 0)
        self.assertEqual(value["n_contacts"], 1)

    def test_invalid_atom_alignment(self):
        with self.assertRaises(ValueError):
            interface_descriptors([[0, 0, 0]], [0], [True], [[1, 0, 0]], [], -1)


if __name__ == "__main__":
    unittest.main()
