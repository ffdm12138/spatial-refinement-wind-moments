"""Unit tests for constant-safe multicase metric helpers.

Run with::

    python -m unittest source/scripts/paper0_multicase/test_metrics.py
"""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mc_moments as M


class MetricContractTests(unittest.TestCase):
    def assert_spearman_matches_scipy(self, x, y):
        expected = float(spearmanr(np.asarray(x, float), np.asarray(y, float)).statistic)
        actual = M.spearman(np.asarray(x, float), np.asarray(y, float))
        if np.isnan(expected):
            self.assertTrue(np.isnan(actual))
        else:
            self.assertAlmostEqual(actual, expected, places=14)

    def test_constant_predictor(self):
        x = np.full(8, 2.0)
        y = np.arange(8, dtype=float)
        self.assertTrue(np.isnan(M.pearson(x, y)))
        self.assertTrue(np.isnan(M.spearman(x, y)))
        metrics = M.cell_metrics(x, y)
        self.assertTrue(np.isnan(metrics["r"]))
        self.assertTrue(np.isnan(metrics["slope"]))

    def test_constant_target(self):
        x = np.arange(8, dtype=float)
        y = np.full(8, 2.0)
        self.assertTrue(np.isnan(M.pearson(x, y)))
        self.assertTrue(np.isnan(M.spearman(x, y)))
        metrics = M.cell_metrics(x, y)
        self.assertTrue(np.isnan(metrics["r"]))
        self.assertTrue(np.isnan(metrics["slope"]))

    def test_perfect_increasing(self):
        x = np.arange(7, dtype=float)
        self.assert_spearman_matches_scipy(x, 3.0 * x + 1.0)
        self.assertAlmostEqual(M.spearman(x, 3.0 * x + 1.0), 1.0)

    def test_perfect_decreasing(self):
        x = np.arange(7, dtype=float)
        self.assert_spearman_matches_scipy(x, -x)
        self.assertAlmostEqual(M.spearman(x, -x), -1.0)

    def test_ties(self):
        x = [1, 1, 2, 3, 3, 4]
        y = [4, 4, 2, 1, 1, 0]
        self.assert_spearman_matches_scipy(x, y)

    def test_random(self):
        rng = np.random.default_rng(20260829)
        x = rng.normal(size=100)
        y = 0.4 * x + rng.normal(size=100)
        self.assert_spearman_matches_scipy(x, y)

    def test_single_value_perturbation(self):
        x = np.ones(12)
        x[-1] += 1.0e-6
        y = np.arange(12, dtype=float)
        self.assert_spearman_matches_scipy(x, y)
        self.assertTrue(np.isfinite(M.spearman(x, y)))


if __name__ == "__main__":
    unittest.main()
