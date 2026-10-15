import unittest
from pcbm2s.common import classification_metrics, validation_threshold


class MetricTests(unittest.TestCase):
    def test_perfect(self):
        result = classification_metrics([0, 1], [0.1, 0.9])
        self.assertEqual(result["auroc"], 1)
        self.assertEqual(result["ap"], 1)
        self.assertEqual(result["f1"], 1)

    def test_one_class_has_undefined_ranking(self):
        result = classification_metrics([1, 1], [0.2, 0.9])
        self.assertIsNone(result["auroc"])
        self.assertIsNone(result["ap"])

    def test_threshold_uses_validation(self):
        self.assertEqual(validation_threshold([0, 1, 1], [0.1, 0.2, 0.9]), 0.2)

    def test_bad_probabilities_are_rejected(self):
        with self.assertRaises(ValueError):
            classification_metrics([0, 1], [0.5, 2])


if __name__ == "__main__":
    unittest.main()
