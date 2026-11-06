"""Real learner progress, neural gradients, and resumable checkpoint contracts."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pokerrl.cfr import CFR
from pokerrl.evaluate import Evaluator
from pokerrl.neural import NeuralPG


class TrainingTests(unittest.TestCase):
    def test_cfr_average_reduces_exploitability_toward_equilibrium(self):
        trainer = CFR()
        evaluator = Evaluator()
        for _ in range(2000):
            trainer.step()
        metrics = evaluator.metrics(trainer.average_policy())
        self.assertLess(metrics.exploitability, 0.025)
        self.assertLess(abs(metrics.value + 1 / 18), 0.015)

    def test_neural_backpropagation_matches_weight_finite_difference(self):
        trainer = NeuralPG(3)
        evaluator = Evaluator()
        profile = trainer.current_policy()
        derivative = trainer.networks[0].gradients(evaluator.gradient(profile)[:6])
        for layer, index in ((0, (0, 0)), (1, (2,)), (2, (0, 1)), (3, (0,))):
            with self.subTest(layer=layer):
                parameter = trainer.networks[0].parameters[layer]
                original = parameter[index]
                parameter[index] = original + 1e-6
                plus = float(evaluator.values(trainer.current_policy()))
                parameter[index] = original - 1e-6
                minus = float(evaluator.values(trainer.current_policy()))
                parameter[index] = original
                np.testing.assert_allclose(
                    derivative[layer][index], (plus - minus) / 2e-6, atol=1e-10
                )

    def test_real_weight_training_and_identical_checkpoint_continuation(self):
        trainer = NeuralPG(17)
        before = [array.copy() for network in trainer.networks for array in network.parameters]
        for _ in range(5):
            trainer.step()
        after = [array for network in trainer.networks for array in network.parameters]
        self.assertTrue(
            all(
                np.max(np.abs(left - right)) > 1e-8
                for left, right in zip(before, after, strict=True)
            )
        )
        with tempfile.TemporaryDirectory(prefix="pokerrl-checkpoint-") as directory:
            path = Path(directory) / "checkpoint.npz"
            trainer.save(path)
            restored = NeuralPG.load(path)
        np.testing.assert_array_equal(restored.current_policy(), trainer.current_policy())
        np.testing.assert_array_equal(restored.average_policy(), trainer.average_policy())
        trainer.step()
        restored.step()
        self.assertEqual(restored.steps, 6)
        self.assertEqual(trainer.steps, 6)
        np.testing.assert_array_equal(restored.current_policy(), trainer.current_policy())
        np.testing.assert_array_equal(restored.average_policy(), trainer.average_policy())

    def test_first_player_ascends_reward_against_fixed_opponent(self):
        trainer = NeuralPG(8, learning_rate=0.0001)
        evaluator = Evaluator()
        original = trainer.current_policy()
        derivatives = trainer.networks[0].gradients(evaluator.gradient(original)[:6])
        trainer.networks[0].ascend(derivatives, trainer.learning_rate)
        self.assertGreater(
            float(evaluator.values(trainer.current_policy())), float(evaluator.values(original))
        )
