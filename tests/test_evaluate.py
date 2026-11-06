"""Analytic and finite-difference contracts for the independent exact evaluator."""

import unittest

import numpy as np

from pokerrl.evaluate import Evaluator
from pokerrl.game import equilibrium_policy, random_policy


class EvaluationTests(unittest.TestCase):
    def test_equilibrium_has_kuhn_value_and_zero_consistent_exploitability(self):
        metrics = Evaluator().metrics(equilibrium_policy())
        self.assertAlmostEqual(metrics.value, -1 / 18, places=14)
        self.assertAlmostEqual(metrics.exploitability, 0, places=14)
        self.assertAlmostEqual(metrics.best_response_p0, -1 / 18, places=14)
        self.assertAlmostEqual(metrics.best_response_p1, 1 / 18, places=14)

    def test_random_profile_has_known_value_and_best_responses(self):
        metrics = Evaluator().metrics(random_policy())
        self.assertAlmostEqual(metrics.value, 1 / 8)
        self.assertAlmostEqual(metrics.best_response_p0, 1 / 2)
        self.assertAlmostEqual(metrics.best_response_p1, 5 / 12)
        self.assertAlmostEqual(metrics.exploitability, 11 / 24)
        self.assertAlmostEqual(metrics.nash_conv, 11 / 12)

    def test_return_gradient_matches_probability_finite_differences(self):
        evaluator = Evaluator()
        policy = np.random.default_rng(12).uniform(0.1, 0.9, (12, 2))
        analytic = evaluator.gradient(policy)
        numerical = np.zeros((12, 2))
        for index in np.ndindex(policy.shape):
            plus, minus = policy.copy(), policy.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            numerical[index] = (evaluator.values(plus) - evaluator.values(minus)) / 2e-6
        np.testing.assert_allclose(analytic, numerical, atol=1e-10, rtol=1e-7)
