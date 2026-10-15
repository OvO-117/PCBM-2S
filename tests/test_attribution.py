"""Numerical tests independent of graph encoders and real datasets."""

import unittest

import torch
from torch import nn

from pcbm2s.attribute import attribute_sample, exact_gate_shapley


class ToyGateModel(nn.Module):
    n_concept = 5
    struct_feature_names = ["vina", "contacts", "polar", "hydrophobic", "residues"]

    def __init__(self, use_gate=True):
        super().__init__()
        self.use_gate = use_gate
        self.encode_calls = 0
        self.head_calls = 0
        self.register_buffer("weights", torch.tensor([0.7, -0.4, 0.9, 0.3, 0.2]))
        self.observed_struct = None

    def encode_pair(self, batch):
        self.encode_calls += 1
        return batch["z"]

    def normalized_concepts(self, batch):
        return batch["c"]

    def head_from_pair(self, z, c_struct, c_gate=None, use_gate=None):
        self.head_calls += 1
        self.observed_struct = c_struct.detach().clone()
        gate_enabled = self.use_gate if use_gate is None else use_gate
        base = z[:, :1] + 0.15 * c_struct.sum(dim=1, keepdim=True)
        if not gate_enabled:
            return base
        gate = (c_gate * self.weights).sum(dim=1, keepdim=True)
        gate = gate + 0.4 * c_gate[:, :1] * c_gate[:, 2:3]
        return base + gate


def inputs():
    return {"z": torch.tensor([[0.25, -0.1]]), "c": torch.tensor([[1.0, 1.0, 0.5, -0.3, 0.2]])}


class AttributionTests(unittest.TestCase):
    def test_exact_efficiency_negative_contribution_and_single_encoding(self):
        model = ToyGateModel()
        model.frozen_dropout = nn.Dropout()
        batch = inputs()
        model.train()
        model.frozen_dropout.eval()
        result = attribute_sample(model, batch)
        self.assertTrue(model.training)
        self.assertFalse(model.frozen_dropout.training)
        self.assertEqual(model.encode_calls, 1)
        self.assertEqual(model.head_calls, 1)
        self.assertEqual(model.observed_struct.shape, (32, 5))
        self.assertTrue(torch.equal(model.observed_struct, batch["c"].expand(32, -1)))
        self.assertEqual(result["n_coalitions"], 32)
        self.assertAlmostEqual(sum(result["contributions"].values()), result["gate_effect"], delta=1e-12)
        self.assertAlmostEqual(result["efficiency_error"], 0.0, delta=1e-12)
        self.assertLess(result["contributions"]["contacts"], 0)
        self.assertLess(result["contributions"]["hydrophobic"], 0)
        self.assertGreater(result["full_probability"], result["base_probability"])

    def test_zero_concepts_are_dummy_players(self):
        model = ToyGateModel()
        batch = inputs()
        batch["c"][:, 3:] = 0
        result = exact_gate_shapley(model, batch["z"], batch["c"])
        self.assertEqual(result["contributions"]["hydrophobic"], 0.0)
        self.assertEqual(result["contributions"]["residues"], 0.0)
        self.assertAlmostEqual(result["efficiency_error"], 0, delta=1e-12)

    def test_zero_effect_for_disabled_gate(self):
        result = attribute_sample(ToyGateModel(use_gate=False), inputs())
        self.assertEqual(result["gate_effect"], 0.0)
        self.assertTrue(all(value == 0.0 for value in result["contributions"].values()))

    def test_zero_effect_for_all_mean_concepts(self):
        batch = inputs()
        batch["c"].zero_()
        result = attribute_sample(ToyGateModel(), batch)
        self.assertEqual(result["gate_effect"], 0.0)
        self.assertTrue(all(value == 0.0 for value in result["contributions"].values()))

    def test_baseline_is_mean_concepts_not_disabled_gate(self):
        class BiasedGate(ToyGateModel):
            def head_from_pair(self, z, c_struct, c_gate=None, use_gate=None):
                result = super().head_from_pair(z, c_struct, c_gate, use_gate)
                return result + 1.0

        model = BiasedGate()
        batch = inputs()
        result = attribute_sample(model, batch)
        expected = model.head_from_pair(batch["z"], batch["c"], torch.zeros_like(batch["c"])).sigmoid().item()
        self.assertAlmostEqual(result["base_probability"], expected)
        self.assertAlmostEqual(result["efficiency_error"], 0, delta=1e-12)

    def test_rejects_multiple_pairs_and_wrong_concept_count(self):
        model, batch = ToyGateModel(), inputs()
        with self.assertRaisesRegex(ValueError, "shape"):
            exact_gate_shapley(model, batch["z"].expand(2, -1), batch["c"].expand(2, -1))
        with self.assertRaisesRegex(ValueError, "five"):
            exact_gate_shapley(model, batch["z"], batch["c"], ["same"] * 5)
        with self.assertRaisesRegex(ValueError, "finite"):
            exact_gate_shapley(model, batch["z"], torch.full((1, 5), float("nan")))

    def test_eval_mode_is_restored_after_failure(self):
        class FailingHead(ToyGateModel):
            def head_from_pair(self, *args, **kwargs):
                raise RuntimeError("deliberate failure")

        model = FailingHead()
        model.train()
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            exact_gate_shapley(model, inputs()["z"], inputs()["c"])
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
