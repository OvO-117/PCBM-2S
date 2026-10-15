"""Model invariants using toy inputs and explicit GNNPlus diagnostic encoders."""

import unittest
import torch

from pcbm2s.data import CONCEPT_NAMES, fit_descriptor_stats, make_loader
from pcbm2s.model import PCBM2S
from pcbm2s.smoke import diagnostic_config, synthetic_pairs


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.samples = synthetic_pairs()
        cls.stats = fit_descriptor_stats(cls.samples)

    def make_model(self, **kwargs):
        config = diagnostic_config()
        config.update(kwargs)
        return PCBM2S(self.stats, **config)

    def test_zero_initialized_gate_is_identity(self):
        model = self.make_model().eval()
        batch = next(iter(make_loader(self.samples, 4, False)))
        with torch.no_grad():
            z = model.encode_pair(batch)
            c = model.normalized_concepts(batch)
            self.assertEqual(torch.count_nonzero(model.compute_gate(c)).item(), 0)
            torch.testing.assert_close(model.head_from_pair(z, c), model.head_from_pair(z, c, use_gate=False))

    def test_output_shapes_and_backward(self):
        model = self.make_model()
        batch = next(iter(make_loader(self.samples, 4, False)))
        out, auxiliary, concepts = model(batch, return_aux=True)
        self.assertEqual(out.shape, (4, 1))
        self.assertEqual(auxiliary.shape, (4, 5))
        loss = out.square().mean() + (auxiliary - concepts).square().mean()
        loss.backward()
        self.assertIsNotNone(model.prot_rwse_encoder.encoder.weight.grad)
        self.assertIsNotNone(model.gate_net.weight.grad)

    def test_eval_batch_consistency(self):
        model = self.make_model().eval()
        with torch.no_grad():
            together = model(next(iter(make_loader(self.samples, 4, False))))
            separate = torch.cat([model(next(iter(make_loader([sample], 1, False)))) for sample in self.samples])
        torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-5)

    def test_protein_rwse_is_consumed(self):
        model = self.make_model().eval()
        batch = next(iter(make_loader(self.samples, 4, False)))
        with torch.no_grad():
            first = model.encode_pair(batch)
            batch.prot_pestat_RWSE = 1 - batch.prot_pestat_RWSE
            second = model.encode_pair(batch)
        self.assertFalse(torch.allclose(first[:, 32:], second[:, 32:]))

    def test_aux_head_can_be_disabled(self):
        model = self.make_model(use_concept_head=False).eval()
        batch = next(iter(make_loader(self.samples, 4, False)))
        self.assertIsNone(model(batch, return_aux=True)[1])

    def test_missing_protein_rwse_is_explicit_error(self):
        model = self.make_model().eval()
        batch = next(iter(make_loader(self.samples, 4, False)))
        del batch.prot_pestat_RWSE
        with self.assertRaisesRegex(ValueError, "prot_pestat_RWSE"):
            model(batch)

    def test_training_statistics_are_buffers(self):
        model = self.make_model()
        self.assertIn("struct_mean", model.state_dict())
        self.assertIn("struct_std", model.state_dict())
        bad = dict(self.stats)
        bad[CONCEPT_NAMES[0]] = {"mean": 0, "std": 0}
        with self.assertRaises(ValueError):
            PCBM2S(bad, **diagnostic_config())


if __name__ == "__main__":
    unittest.main()
