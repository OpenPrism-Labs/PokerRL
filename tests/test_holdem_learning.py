"""Numerical and public-information contracts for sampled Hold'em learning."""

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from pokerrl.holdem import (
    Action,
    HandResult,
    HoldemEnv,
    LegalActions,
    Observation,
    PublicAction,
)
from pokerrl.holdem_learning import (
    NeuralPolicy,
    OpponentHistory,
    Policy,
    ScriptedPolicy,
    Trainer,
    abstract_actions,
    encode,
)


def observation() -> Observation:
    return Observation(
        seat=2,
        button=5,
        street=0,
        hole=(40, 49),
        board=(),
        stacks=(199, 198, 200, 200, 200, 200),
        bets=(1, 2, 0, 0, 0, 0),
        contributions=(1, 2, 0, 0, 0, 0),
        active=(True,) * 6,
        all_in=(False,) * 6,
        pot=3,
        big_blind=2,
        legal=LegalActions(True, True, 2, 4, 200),
        actions=(),
    )


def result() -> HandResult:
    return HandResult(
        button=5,
        board=(),
        payoffs=(0,) * 6,
        actions=(
            PublicAction(3, 0, "check_call", 2, None, True),
            PublicAction(3, 0, "raise_to", 6, 8, True),
            PublicAction(3, 1, "check_call", 0, None, False),
            PublicAction(3, 2, "fold", 0, None, True),
            PublicAction(4, 0, "check_call", 0, None, False),
        ),
    )


class LearningTests(unittest.TestCase):
    def test_history_uses_completed_hand_window_and_relative_seats(self) -> None:
        history = OpponentHistory(window=2)
        np.testing.assert_array_equal(history.features(2), np.zeros((6, 6)))
        history.update(result())
        np.testing.assert_allclose(history.features(2)[1], [0.5, 1, 1, 0.25, 1 / 3, 1 / 3])
        np.testing.assert_array_equal(history.features(3)[0], np.zeros(6))
        empty = replace(result(), actions=())
        history.update(empty)
        np.testing.assert_allclose(history.features(2)[1], [1, 0.5, 0.5, 0.25, 1 / 3, 1 / 3])
        history.update(empty)
        np.testing.assert_array_equal(history.features(2)[1], [1, 0, 0, 0, 0, 0])
        history.clear()
        np.testing.assert_array_equal(history.features(2), np.zeros((6, 6)))

    def test_abstract_actions_clamp_deduplicate_and_mask(self) -> None:
        obs = observation()
        self.assertEqual(
            abstract_actions(obs),
            (
                Action("fold"),
                Action("check_call"),
                Action("raise_to", 4),
                None,
                Action("raise_to", 7),
                Action("raise_to", 200),
            ),
        )
        short = replace(obs, legal=LegalActions(True, True, 2, 3, 3))
        self.assertEqual(
            abstract_actions(short),
            (Action("fold"), Action("check_call"), Action("raise_to", 3), None, None, None),
        )
        check = replace(obs, legal=LegalActions(False, True, 0, None, None))
        policy = NeuralPolicy(0)
        np.testing.assert_array_equal(
            policy.probabilities(check, OpponentHistory()), [0, 1, 0, 0, 0, 0]
        )
        probabilities = policy.probabilities(obs, OpponentHistory())
        self.assertEqual(probabilities[3], 0)
        self.assertAlmostEqual(float(probabilities.sum()), 1)

    def test_encoder_uses_public_relative_state_and_switchable_history(self) -> None:
        history = OpponentHistory()
        history.update(result())
        obs = replace(observation(), actions=result().actions)
        encoded = encode(obs, history, use_history=True)
        self.assertEqual(encoded.shape, (209,))
        np.testing.assert_array_equal(np.flatnonzero(encoded[:104]), [40, 49])
        self.assertEqual(encoded[111], 1)  # Button is three seats clockwise from observer.
        self.assertEqual(encoded[123 + 6], 1)  # Relative seat one last folded.
        self.assertAlmostEqual(encoded[168], np.log1p(3 / 2))
        without = encode(obs, history, use_history=False)
        np.testing.assert_array_equal(without[:173], encoded[:173])
        np.testing.assert_array_equal(without[173:], np.zeros(36))
        np.testing.assert_array_equal(
            NeuralPolicy(4, use_history=False).probabilities(obs, history),
            NeuralPolicy(4, use_history=False).probabilities(obs, OpponentHistory()),
        )

    def test_hidden_cards_do_not_change_public_policy_input(self) -> None:
        deck = tuple(range(52))
        first = HoldemEnv(3)
        first.reset(deck=deck)
        obs = first.observation()
        history = OpponentHistory()
        policy = NeuralPolicy(4)
        opponent_cards = [card for card in range(12) if card not in obs.hole]
        # Both opponent hole cards and undealt future cards are outside the boundary.
        for left, right in ((opponent_cards[0], opponent_cards[-1]), (50, 51)):
            with self.subTest(cards=(left, right)):
                changed = list(deck)
                changed[left], changed[right] = changed[right], changed[left]
                second = HoldemEnv(3)
                second.reset(deck=tuple(changed))
                self.assertEqual(obs, second.observation())
                np.testing.assert_array_equal(
                    policy.probabilities(obs, history),
                    policy.probabilities(second.observation(), history),
                )

    def test_score_gradient_matches_directional_finite_difference(self) -> None:
        policy = NeuralPolicy(6)
        inputs = encode(observation(), OpponentHistory(), use_history=True)
        mask = np.array([True, True, True, False, True, True])
        probabilities, hidden = policy.forward(inputs, mask)
        gradients = policy.log_gradient(inputs, probabilities, hidden, 2)
        rng = np.random.default_rng(9)
        for parameter, gradient in zip(policy.parameters, gradients, strict=True):
            direction = rng.normal(size=parameter.shape)
            original = parameter.copy()
            parameter[:] = original + 1e-6 * direction
            plus = np.log(policy.forward(inputs, mask)[0][2])
            parameter[:] = original - 1e-6 * direction
            minus = np.log(policy.forward(inputs, mask)[0][2])
            parameter[:] = original
            self.assertAlmostEqual(
                float(np.sum(gradient * direction)), float((plus - minus) / 2e-6), places=7
            )
        self.assertEqual(gradients[3][3], 0)

    def test_all_policies_produce_legal_actions_through_real_hands(self) -> None:
        policies: tuple[Policy, ...] = (
            NeuralPolicy(2),
            NeuralPolicy(3, use_history=False),
            ScriptedPolicy("random"),
            ScriptedPolicy("tight"),
            ScriptedPolicy("loose"),
            ScriptedPolicy("aggressive"),
        )
        history = OpponentHistory()
        rng = np.random.default_rng(8)
        for seed in range(4):
            env = HoldemEnv(seed)
            env.reset()
            while not env.terminal:
                obs = env.observation()
                for action in abstract_actions(obs):
                    if action is not None:
                        self.assertTrue(env.is_legal(action), (obs, action))
                action = policies[obs.seat].act(obs, history, rng)
                self.assertTrue(env.is_legal(action))
                env.step(action)
            history.update(env.result())

    def test_training_freezes_hand_and_uses_prior_baseline(self) -> None:
        trainer = Trainer(4, use_history=True)
        initial = [parameter.copy() for parameter in trainer.policy.parameters]
        history = OpponentHistory()
        history.update(result())
        prior_history = history.features(1)
        visits: list[int] = []
        case = self

        class Observer:
            def act(
                self,
                observation: Observation,
                history: OpponentHistory,
                rng: np.random.Generator,
            ) -> Action:
                visits.append(observation.seat)
                for actual, original in zip(trainer.policy.parameters, initial, strict=True):
                    np.testing.assert_array_equal(actual, original)
                np.testing.assert_array_equal(history.features(1), prior_history)
                case.assertEqual(trainer.baseline, 0)
                return Action("check_call")

        env = HoldemEnv(7)
        env.reset()
        sample = trainer.train_hand(env, (Observer(),) * 6, history, hero=1)
        self.assertTrue(visits)
        self.assertGreater(sample.decisions, 0)
        self.assertNotEqual(sample.payoff_bb, 0)
        self.assertEqual(trainer.baseline, sample.payoff_bb)
        self.assertTrue(
            any(
                np.any(before != after)
                for before, after in zip(initial, trainer.policy.parameters, strict=True)
            )
        )
        self.assertEqual(history.features(1)[1, 0], 2 / 32)

    def test_checkpoint_roundtrip_and_exact_trainer_continuation(self) -> None:
        trainer = Trainer(10, use_history=False)
        history = OpponentHistory()
        opponents: tuple[Policy, ...] = (ScriptedPolicy("loose"),) * 6
        env = HoldemEnv(30)
        env.reset()
        trainer.train_hand(env, opponents, history, hero=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trainer.npz"
            trainer.save(path)
            restored = Trainer.load(path)
            policy = NeuralPolicy.load(path)
            np.testing.assert_array_equal(
                policy.probabilities(observation(), history),
                trainer.policy.probabilities(observation(), history),
            )
            history_copy = copy.deepcopy(history)
            for seed in (11, 12):
                left, right = HoldemEnv(seed), HoldemEnv(seed)
                left.reset()
                right.reset()
                self.assertEqual(
                    trainer.train_hand(left, opponents, history, hero=seed % 6),
                    restored.train_hand(right, opponents, history_copy, hero=seed % 6),
                )
                for actual, expected in zip(
                    restored.policy.parameters, trainer.policy.parameters, strict=True
                ):
                    np.testing.assert_array_equal(actual, expected)
            inference = Path(directory) / "policy"
            policy.save(inference)
            np.testing.assert_array_equal(
                NeuralPolicy.load(inference).probabilities(observation(), history),
                policy.probabilities(observation(), history),
            )
            with self.assertRaises(ValueError):
                Trainer.load(inference)


if __name__ == "__main__":
    unittest.main()
