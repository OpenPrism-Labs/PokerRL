"""Credit assignment across actual multi-player hands, including folded heroes."""

import unittest

import numpy as np

from pokerrl.holdem import Action, HoldemEnv
from pokerrl.holdem_learning import OpponentHistory
from pokerrl.rl_common import Decision, collect_hand


class Calls:
    def act(self, observation, history, rng):
        return Action("check_call")

    def decision(self, inputs, mask, rng, *, training):
        return Decision(1, -0.7, 0.25)


class Folds(Calls):
    def decision(self, inputs, mask, rng, *, training):
        return Decision(0)


class RolloutTests(unittest.TestCase):
    def test_reward_is_assigned_once_after_full_hand_to_same_seat(self):
        env = HoldemEnv(42)
        env.reset()
        history = OpponentHistory()
        trajectory, payoff = collect_hand(
            env, Calls(), (Calls(),) * 6, history, np.random.default_rng(0), hero=2
        )
        self.assertEqual(len(trajectory), 4)
        self.assertEqual([row.done for row in trajectory], [False, False, False, True])
        self.assertEqual([row.reward for row in trajectory[:-1]], [0.0] * 3)
        self.assertEqual(trajectory[-1].reward, payoff * 0.01)
        self.assertEqual(payoff, 2)
        for left, right in zip(trajectory[:-1], trajectory[1:], strict=True):
            np.testing.assert_array_equal(left.next_inputs, right.inputs)
            np.testing.assert_array_equal(left.next_mask, right.mask)
        self.assertFalse(trajectory[-1].next_mask.any())
        self.assertEqual(history.features(0)[1, 0], 1 / 32)

    def test_folded_blind_receives_final_net_loss_without_future_card_features(self):
        env = HoldemEnv(42)
        env.reset()
        trajectory, payoff = collect_hand(
            env,
            Folds(),
            (Calls(),) * 6,
            OpponentHistory(),
            np.random.default_rng(0),
            hero=0,
        )
        self.assertEqual(len(trajectory), 1)
        self.assertTrue(trajectory[0].done)
        self.assertEqual(payoff, -0.5)
        self.assertEqual(trajectory[0].reward, -0.005)
        self.assertEqual(float(trajectory[0].inputs[52:104].sum()), 0)
        self.assertEqual(len(env.result().board), 5)

    def test_no_history_collection_zeros_only_history_input_region(self):
        env = HoldemEnv(42)
        env.reset()
        history = OpponentHistory()
        collect_hand(env, Calls(), (Calls(),) * 6, history, np.random.default_rng(0), hero=2)
        env.reset()
        rows, _ = collect_hand(
            env,
            Calls(),
            (Calls(),) * 6,
            history,
            np.random.default_rng(1),
            hero=2,
            use_history=False,
        )
        self.assertTrue(history.features(2).any())
        self.assertTrue(all(not row.inputs[173:].any() for row in rows))
