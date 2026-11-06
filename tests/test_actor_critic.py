"""Independent numerical targets and real complete-hand actor-critic updates."""

import tempfile
import unittest
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from pokerrl.actor_critic import (
    ActorCritic,
    actor_critic_loss,
    generalized_advantages,
    masked_log_probs,
    normalize_advantages,
)
from pokerrl.holdem import HoldemEnv
from pokerrl.holdem_learning import OpponentHistory, Policy, ScriptedPolicy, encode
from pokerrl.rl_common import Agent, Transition, collect_hand


def hand(agent: ActorCritic, size: int = 2, *, one_action: bool = False) -> list[Transition]:
    rng = np.random.default_rng(19)
    trajectory: list[Transition] = []
    for index in range(size):
        inputs = rng.normal(size=209)
        mask = np.array([False, True, not one_action, False, False, False])
        decision = agent.decision(inputs, mask, rng, training=True)
        done = index == size - 1
        trajectory.append(
            Transition(
                inputs,
                mask,
                decision.index,
                2.0 if done else 0.0,
                np.zeros(209),
                np.zeros(6, dtype=np.bool_),
                done,
                decision.log_prob,
                decision.value,
            )
        )
    return trajectory


class ActorCriticMathTests(unittest.TestCase):
    def test_gae_independent_hand_targets(self) -> None:
        advantages, returns = generalized_advantages(
            np.array([0.0, 3.0, 0.0, -2.0]),
            np.array([0.5, 1.0, 10.0, 4.0]),
            np.array([False, True, False, True]),
        )
        # delta=(0.5,2,-6,-6); each terminal resets the lambda recurrence.
        np.testing.assert_allclose(advantages, [2.4, 2.0, -11.7, -6.0])
        np.testing.assert_allclose(returns, [2.9, 3.0, -1.7, -2.0])
        terminal, target = generalized_advantages(
            np.array([7.0]), np.array([100.0]), np.array([True])
        )
        np.testing.assert_array_equal(terminal, [-93.0])
        np.testing.assert_array_equal(target, [7.0])
        with self.assertRaises(ValueError):
            generalized_advantages(np.zeros(1), np.zeros(1), np.array([False]))

    def test_ppo_all_clipping_branches_and_frozen_statistics(self) -> None:
        # New action probability .5, old probabilities produce ratios 1.5,.5,1.5,.5.
        logits = torch.zeros((4, 2), dtype=torch.float64, requires_grad=True)
        old = torch.tensor([1 / 3, 1.0, 1 / 3, 1.0], dtype=torch.float64).log()
        old.requires_grad_()
        advantages = torch.tensor([2.0, 2.0, -2.0, -2.0], requires_grad=True)
        returns = torch.ones(4, requires_grad=True)
        original = old.detach().clone()
        loss = actor_critic_loss(
            logits,
            torch.zeros(4),
            torch.ones((4, 2), dtype=torch.bool),
            torch.zeros(4, dtype=torch.long),
            old,
            advantages,
            returns,
            algorithm="ppo",
            value_coef=0.0,
            entropy_coef=0.0,
        )
        # min(r*A,clip(r)*A) = [2.4, 1, -3, -1.6], negative mean = .3.
        self.assertAlmostEqual(float(loss.detach()), 0.3, places=7)
        torch.autograd.backward(loss)
        self.assertIsNone(old.grad)
        self.assertIsNone(advantages.grad)
        self.assertIsNone(returns.grad)
        torch.testing.assert_close(old.detach(), original)
        gradient = logits.grad
        assert gradient is not None
        torch.testing.assert_close(gradient[[0, 3]], torch.zeros((2, 2), dtype=torch.float64))
        self.assertLess(float(gradient[1, 0]), 0)
        self.assertGreater(float(gradient[2, 0]), 0)

    def test_a2c_value_and_entropy_coefficients(self) -> None:
        loss = actor_critic_loss(
            torch.tensor([[np.log(0.25), np.log(0.75), 100.0]]),
            torch.tensor([1.0], requires_grad=True),
            torch.tensor([[True, True, False]]),
            torch.tensor([0]),
            torch.tensor([999.0]),
            torch.tensor([2.0]),
            torch.tensor([3.0]),
            algorithm="a2c",
        )
        entropy = -0.25 * np.log(0.25) - 0.75 * np.log(0.75)
        expected = -2 * np.log(0.25) + 0.5 * 4 - 0.01 * entropy
        self.assertAlmostEqual(float(loss.detach()), expected, places=7)

    def test_one_action_mask_has_finite_loss_and_zero_actor_gradient(self) -> None:
        for algorithm in ("ppo", "a2c"):
            logits = torch.tensor([[100.0, -3.0, 200.0]], requires_grad=True)
            masks = torch.tensor([[False, True, False]])
            torch.testing.assert_close(masked_log_probs(logits, masks).exp(), masks.float())
            loss = actor_critic_loss(
                logits,
                torch.tensor([0.0]),
                masks,
                torch.tensor([1]),
                torch.tensor([0.0]),
                torch.tensor([2.0]),
                torch.tensor([2.0]),
                algorithm=algorithm,
            )
            self.assertTrue(bool(torch.isfinite(loss)))
            torch.autograd.backward(loss)
            torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        with self.assertRaises(ValueError):
            masked_log_probs(torch.zeros(6), torch.zeros(6, dtype=torch.bool))

    def test_normalization_preserves_small_and_constant_batch_signal(self) -> None:
        for values in ([3.0], [2.0, 2.0]):
            tensor = torch.tensor(values)
            torch.testing.assert_close(normalize_advantages(tensor), tensor)
        torch.testing.assert_close(
            normalize_advantages(torch.tensor([1.0, 3.0])), torch.tensor([-1.0, 1.0])
        )


class ActorCriticTrainingTests(unittest.TestCase):
    def test_complete_hands_threshold_true_updates_and_old_stats(self) -> None:
        for algorithm in ("ppo", "a2c"):
            agent = ActorCritic(11, algorithm)
            initial = [parameter.detach().clone() for parameter in agent.model.parameters()]
            first = hand(agent, 255)
            agent.learn(first)
            self.assertEqual(agent.updates, 0)
            second = hand(agent, 2)
            old_stats = [(step.log_prob, step.value) for step in first + second]
            agent.learn(second)
            # 257 transitions remain one rollout, never splitting a hand at 256.
            self.assertEqual(agent.updates, 20 if algorithm == "ppo" else 1)
            self.assertEqual(agent.rollout, [])
            self.assertEqual(old_stats, [(step.log_prob, step.value) for step in first + second])
            self.assertTrue(
                any(
                    not torch.equal(before, after)
                    for before, after in zip(initial, agent.model.parameters(), strict=True)
                )
            )
            self.assertTrue(np.isfinite(agent.losses).all())
            norm = (
                sum(
                    float(parameter.grad.square().sum())
                    for parameter in agent.model.parameters()
                    if parameter.grad is not None
                )
                ** 0.5
            )
            self.assertLessEqual(norm, 0.50001)

    def test_finish_singleton_and_reject_partial_hand(self) -> None:
        for algorithm in ("ppo", "a2c"):
            agent = ActorCritic(2, algorithm)
            with self.assertRaises(ValueError):
                agent.learn(hand(agent)[:1])
            self.assertFalse(agent.rollout)
            agent.learn(hand(agent, 1, one_action=True))
            agent.finish()
            self.assertEqual(agent.updates, 4 if algorithm == "ppo" else 1)
            self.assertTrue(np.isfinite(agent.losses).all())
            count = agent.updates
            agent.finish()
            agent.learn([])
            self.assertEqual(agent.updates, count)

    def test_real_hands_interface_and_inference_reload(self) -> None:
        algorithms: tuple[Literal["ppo", "a2c"], ...] = ("ppo", "a2c")
        for algorithm in algorithms:
            agent = ActorCritic(7, algorithm, hidden=32, use_history=False)
            contract: Agent = agent
            opponents: tuple[Policy, ...] = (ScriptedPolicy("loose"),) * 6
            rng = np.random.default_rng(34)
            history = OpponentHistory()
            for seed in range(3):
                env = HoldemEnv(seed)
                env.reset()
                trajectory, _ = collect_hand(
                    env, contract, opponents, history, rng, hero=seed, use_history=False
                )
                self.assertTrue(env.terminal)
                self.assertTrue(trajectory[-1].done)
                self.assertEqual(agent.updates, 0)
                contract.learn(trajectory)
            contract.finish()
            self.assertGreater(contract.updates, 0)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "agent.pt"
                contract.save(path)
                restored = ActorCritic.load(path)
                self.assertEqual(restored.algorithm, algorithm)
                self.assertFalse(restored.use_history)
                self.assertEqual(restored.updates, 0)
                self.assertFalse(restored.rollout)
                assert isinstance(restored.optimizer, torch.optim.Adam)
                self.assertFalse(restored.optimizer.state)
                env = HoldemEnv(42)
                env.reset()
                obs = env.observation()
                inputs = torch.tensor(encode(obs, history, use_history=False), dtype=torch.float32)
                with torch.no_grad():
                    for actual, expected in zip(
                        restored.model.forward(inputs), agent.model.forward(inputs), strict=True
                    ):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(
                    restored.act(obs, history, np.random.default_rng(22)),
                    agent.act(obs, history, np.random.default_rng(22)),
                )

    def test_decision_samples_only_legal_slots_with_caller_rng(self) -> None:
        agent = ActorCritic(1, "ppo")
        mask = np.array([False, True, False, False, False, False])
        for training in (True, False):
            decision = agent.decision(
                np.zeros(209), mask, np.random.default_rng(2), training=training
            )
            self.assertEqual(decision.index, 1)
            self.assertEqual(decision.log_prob, 0.0)
            self.assertTrue(np.isfinite(decision.value))
        first = np.random.default_rng(20)
        second = np.random.default_rng(20)
        for _ in range(20):
            mask = np.array([True, True, False, True, False, True])
            self.assertEqual(
                agent.decision(np.ones(209), mask, first, training=True),
                agent.decision(np.ones(209), mask, second, training=False),
            )


if __name__ == "__main__":
    unittest.main()
