"""Information boundaries, conditional reward learning, and evaluation contracts."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pokerrl.conditioned import OpponentLearner
from pokerrl.evaluate import Evaluator
from pokerrl.game import State, equilibrium_policy, random_policy
from pokerrl.opponent_experiment import evaluate, seat_metrics
from pokerrl.opponents import Opponent, PublicHistory, population


class OpponentTests(unittest.TestCase):
    def test_history_uses_public_opponent_actions_and_decays_once_per_hand(self):
        history = PublicHistory(decay=0.5)
        history.observe("kbc", 1)
        history.observe("bf", 1)
        expected = np.zeros((4, 2))
        expected[2, 1] = 0.5
        expected[3, 0] = 1
        np.testing.assert_array_equal(history.counts, expected)
        np.testing.assert_allclose(
            history.features(), (expected / (2 + expected.sum(axis=1, keepdims=True))).ravel()
        )

    def test_same_visible_history_is_invariant_to_hidden_cards_and_style(self):
        learner = OpponentLearner(4)
        contexts = [PublicHistory(), PublicHistory()]
        # Opponent cards differ, including the unobserved folded card.
        for context, cards in zip(contexts, ((0, 1), (0, 2)), strict=True):
            state = State(cards).step("bet").step("fold")
            context.observe(state.history, 1)
        np.testing.assert_array_equal(
            learner.policy(0, contexts[0]), learner.policy(0, contexts[1])
        )
        before = learner.policy(0, contexts[0]).copy()
        learner.opponents = [Opponent("different_secret_style", equilibrium_policy())]
        np.testing.assert_array_equal(before, learner.policy(0, contexts[0]))

    def test_conditional_gradients_match_reward_finite_differences_both_seats(self):
        learner = OpponentLearner(5)
        opponent = population(7)[0].policy
        history = PublicHistory()
        history.observe("kbc", 0)
        history.observe("bf", 1)
        evaluator = Evaluator()
        for player in (0, 1):
            section = slice(player * 6, player * 6 + 6)
            profile = opponent.copy()
            profile[section] = learner.policy(player, history)
            network = learner.networks[player]
            sign = 1 - 2 * player
            derivatives = network.gradients(sign * evaluator.gradient(profile)[section])
            for layer, index in ((0, (7, 0)), (1, (2,)), (2, (0, 1)), (3, (0,))):
                with self.subTest(player=player, layer=layer):
                    parameter = network.parameters[layer]
                    original = parameter[index]
                    parameter[index] = original + 1e-6
                    profile[section] = learner.policy(player, history)
                    plus = sign * float(evaluator.values(profile))
                    parameter[index] = original - 1e-6
                    profile[section] = learner.policy(player, history)
                    minus = sign * float(evaluator.values(profile))
                    parameter[index] = original
                    self.assertAlmostEqual(
                        derivatives[layer][index], (plus - minus) / 2e-6, places=9
                    )

    def test_rewards_learn_more_bluffing_against_observed_folding(self):
        learner = OpponentLearner(8)
        folding, calling = PublicHistory(), PublicHistory()
        for _ in range(25):
            folding.observe("bf", 1)
            calling.observe("bc", 1)
        fold_policy, call_policy = random_policy(), random_policy()
        fold_policy[9:] = [1.0, 0.0]
        call_policy[9:] = [0.0, 1.0]
        initial = seat_metrics(learner.policy(0, folding), fold_policy, 0).expected_return
        for _ in range(300):
            learner.update(fold_policy, 0, folding)
            learner.update(call_policy, 0, calling)
        self.assertGreater(learner.policy(0, folding)[0, 1] - learner.policy(0, calling)[0, 1], 0.5)
        self.assertGreater(
            seat_metrics(learner.policy(0, folding), fold_policy, 0).expected_return,
            initial + 0.2,
        )

    def test_separately_trained_zero_history_is_context_invariant(self):
        learner = OpponentLearner(4, use_history=False)
        learner.train_episode(3)
        history = PublicHistory()
        history.observe("bc", 1)
        np.testing.assert_array_equal(
            learner.policy(0, history), learner.policy(0, PublicHistory())
        )

    def test_checkpoint_restores_rng_optimizer_and_historical_continuation(self):
        learner = OpponentLearner(17)
        for _ in range(20):
            learner.train_episode(2)
        with tempfile.TemporaryDirectory(prefix="pokerrl-opponents-") as directory:
            path = Path(directory) / "learner.npz"
            learner.save(path)
            restored = OpponentLearner.load(path)
        learner.train_episode(3)
        restored.train_episode(3)
        history = PublicHistory()
        history.observe("bc", 1)
        for player in (0, 1):
            np.testing.assert_array_equal(
                learner.policy(player, history), restored.policy(player, history)
            )
        np.testing.assert_array_equal(learner.archive, restored.archive)
        self.assertEqual(restored.episodes, 21)

    def test_frozen_equilibrium_has_zero_seat_vulnerability(self):
        for player in (0, 1):
            policy = equilibrium_policy()[player * 6 : player * 6 + 6]
            metric = seat_metrics(policy, random_policy(), player)
            self.assertAlmostEqual(metric.vulnerability, 0, places=12)
            self.assertAlmostEqual(metric.worst_case_return, (2 * player - 1) / 18)
        # Uniform play can be exploited, independent of the nominal opponent.
        for player in (0, 1):
            metrics = [
                seat_metrics(random_policy()[:6], opponent, player)
                for opponent in (equilibrium_policy(), random_policy())
            ]
            self.assertGreater(metrics[0].vulnerability, 0.1)
            self.assertEqual(metrics[0].worst_case_return, metrics[1].worst_case_return)

    def test_shifting_evaluation_retains_history_and_reports_finite_metrics(self):
        opponents = population(99, held_out=True)
        rows = evaluate(
            OpponentLearner(5),
            (opponents[0], opponents[4]),
            100,
            hands=4,
            scenario="shifting",
            algorithm="history",
        )
        for player in (0, 1):
            boundary = next(row for row in rows if row.player == player and row.hand == 4)
            self.assertGreater(boundary.history_support, 0)
            self.assertEqual(boundary.opponent, opponents[4].name)
        self.assertTrue(
            all(
                np.isfinite([row.expected_return, row.worst_case_return, row.vulnerability]).all()
                for row in rows
            )
        )


if __name__ == "__main__":
    unittest.main()
