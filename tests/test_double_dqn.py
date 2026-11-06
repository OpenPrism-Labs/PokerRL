"""Numerical Double DQN targets and observable replay/training contracts."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pokerrl.double_dqn import DoubleDQN, double_dqn_targets
from pokerrl.holdem import HoldemEnv
from pokerrl.holdem_learning import OpponentHistory, Policy, ScriptedPolicy
from pokerrl.rl_common import Agent, Transition, collect_hand


def terminal_transition() -> Transition:
    return Transition(
        inputs=np.ones(209),
        mask=np.array([True, True, False, False, False, False]),
        action=1,
        reward=2.0,
        next_inputs=np.zeros(209),
        next_mask=np.zeros(6, dtype=np.bool_),
        done=True,
        log_prob=0.0,
        value=0.0,
    )


class DoubleDQNTests(unittest.TestCase):
    def test_online_selects_target_evaluates_with_legal_mask(self) -> None:
        # Opposite rankings distinguish DDQN from target-max DQN. Illegal
        # action 2 wins both unmasked rankings and must never be selected.
        online = torch.tensor([[4.0, 1.0, 100.0], [1.0, 4.0, 100.0]], requires_grad=True)
        target = torch.tensor([[2.0, 9.0, 200.0], [9.0, 2.0, 200.0]], requires_grad=True)
        masks = torch.tensor([[True, True, False], [True, True, False]])
        actual = double_dqn_targets(
            online, target, masks, torch.tensor([0.5, -0.5]), torch.tensor([False, False])
        )
        torch.testing.assert_close(actual, torch.tensor([2.5, 1.5]))
        self.assertFalse(actual.requires_grad)

    def test_terminal_rewards_ignore_nan_q_values_and_empty_masks(self) -> None:
        actual = double_dqn_targets(
            torch.tensor([[float("nan")] * 6, [0.0, 3.0, 9.0, 9.0, 9.0, 9.0]]),
            torch.tensor([[float("nan")] * 6, [7.0, 2.0, 9.0, 9.0, 9.0, 9.0]]),
            torch.tensor([[False] * 6, [True, True, False, False, False, False]]),
            torch.tensor([-3.0, 1.0]),
            torch.tensor([True, False]),
        )
        torch.testing.assert_close(actual, torch.tensor([-3.0, 3.0]))
        with self.assertRaisesRegex(ValueError, "legal action"):
            double_dqn_targets(
                torch.zeros(1, 6),
                torch.zeros(1, 6),
                torch.zeros(1, 6, dtype=torch.bool),
                torch.ones(1),
                torch.tensor([False]),
            )

    def test_all_terminal_targets_are_exact_rewards(self) -> None:
        rewards = torch.tensor([3.0, -2.0])
        actual = double_dqn_targets(
            torch.full((2, 6), float("nan")),
            torch.full((2, 6), float("nan")),
            torch.zeros(2, 6, dtype=torch.bool),
            rewards,
            torch.ones(2, dtype=torch.bool),
        )
        torch.testing.assert_close(actual, rewards)

    def test_exploration_greedy_mask_and_evaluation_schedule(self) -> None:
        agent = DoubleDQN(1)
        with torch.no_grad():
            for parameter in agent.online.parameters():
                _ = parameter.zero_()
            output = agent.online[2]
            assert isinstance(output, nn.Linear)
            _ = output.bias.copy_(torch.tensor([2.0, 4.0, 100.0, 100.0, 100.0, 100.0]))
        inputs = np.zeros(209)
        mask = np.array([True, True, False, False, False, False])
        rng = np.random.default_rng(7)
        self.assertEqual(agent.epsilon, 1.0)
        sampled = {agent.decision(inputs, mask, rng, training=True).index for _ in range(100)}
        self.assertEqual(sampled, {0, 1})
        state_before = rng.bit_generator.state
        for _ in range(20):
            self.assertEqual(agent.decision(inputs, mask, rng, training=False).index, 1)
        self.assertEqual(agent.decisions, 100)
        self.assertEqual(rng.bit_generator.state, state_before)
        agent.decisions = 5_000
        self.assertAlmostEqual(agent.epsilon, 0.525)
        agent.decisions = 10_000
        self.assertAlmostEqual(agent.epsilon, 0.05)
        agent.decisions = 20_000
        self.assertAlmostEqual(agent.epsilon, 0.05)
        only = np.array([False, True, False, False, False, False])
        for training in (True, False):
            self.assertEqual(agent.decision(inputs, only, rng, training=training).index, 1)
        with self.assertRaisesRegex(ValueError, "legal action"):
            agent.decision(inputs, np.zeros(6, dtype=np.bool_), rng, training=False)

    def test_warmup_updates_frozen_target_and_scheduled_sync(self) -> None:
        agent = DoubleDQN(2)
        initial = [parameter.detach().clone() for parameter in agent.online.parameters()]
        transition = terminal_transition()
        agent.learn([transition] * 127)
        self.assertEqual(agent.updates, 0)
        agent.learn([transition])
        self.assertEqual(agent.updates, 1)
        self.assertEqual(len(agent.losses), 1)
        self.assertTrue(np.isfinite(agent.losses).all())
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(initial, agent.online.parameters(), strict=True)
            )
        )
        for _ in range(98):
            agent.learn([transition])
        self.assertEqual(agent.updates, 99)
        for before, target in zip(initial, agent.target.parameters(), strict=True):
            torch.testing.assert_close(before, target, rtol=0, atol=0)
            self.assertFalse(target.requires_grad)
            self.assertIsNone(target.grad)
        agent.learn([transition])
        self.assertEqual(agent.updates, 100)
        for online, target in zip(
            agent.online.parameters(), agent.target.parameters(), strict=True
        ):
            torch.testing.assert_close(online, target, rtol=0, atol=0)
        agent.finish()
        agent.learn([])
        self.assertEqual(agent.updates, 100)

    def test_replay_is_bounded_and_rejects_illegal_actions(self) -> None:
        agent = DoubleDQN(3)
        transition = terminal_transition()
        with self.assertRaisesRegex(ValueError, "legal"):
            agent.learn([replace(transition, action=2)])
        self.assertEqual(len(agent.replay), 0)
        with self.assertRaisesRegex(ValueError, "complete hand"):
            agent.learn([replace(transition, done=False)])
        agent.learn([transition] * 20_001)
        self.assertEqual(len(agent.replay), 20_000)
        self.assertEqual(agent.updates, 1)

    def test_replay_randomness_is_independent_of_action_randomness(self) -> None:
        left, right = DoubleDQN(6), DoubleDQN(6)
        trajectory = [
            replace(terminal_transition(), reward=index / 64, inputs=np.full(209, index / 128))
            for index in range(128)
        ]
        left.decision(np.zeros(209), trajectory[0].mask, np.random.default_rng(5), training=True)
        for agent in (left, right):
            agent.learn(trajectory)
        self.assertEqual(left.losses, right.losses)
        for a, b in zip(left.online.parameters(), right.online.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_policy_protocol_plays_legal_actions_and_collects_real_hands(self) -> None:
        agent: Agent = DoubleDQN(4)
        history = OpponentHistory()
        rng = np.random.default_rng(42)
        opponents: tuple[Policy, ...] = (ScriptedPolicy("loose"),) * 6
        for seed in range(6):
            env = HoldemEnv(seed)
            env.reset()
            trajectory, payoff = collect_hand(
                env, agent, opponents, history, rng, hero=seed, use_history=True
            )
            self.assertTrue(env.terminal)
            self.assertTrue(np.isfinite(payoff))
            for step in trajectory:
                self.assertTrue(step.mask[step.action])
            agent.learn(trajectory)
        env = HoldemEnv(9)
        env.reset()
        while not env.terminal:
            action = agent.act(env.observation(), history, rng)
            self.assertTrue(env.is_legal(action))
            env.step(action)

    def test_checkpoint_restores_networks_config_and_inference(self) -> None:
        agent = DoubleDQN(9, hidden=12, learning_rate=0.001, use_history=False)
        agent.learn([terminal_transition()] * 128)
        agent.decisions = 321
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ddqn.pt"
            agent.save(path)
            restored = DoubleDQN.load(path, device="cpu")
        self.assertEqual(restored.hidden, 12)
        self.assertEqual(restored.learning_rate, 0.001)
        self.assertFalse(restored.use_history)
        self.assertEqual(restored.updates, 1)
        self.assertEqual(restored.decisions, 321)
        self.assertEqual(restored.epsilon, agent.epsilon)
        self.assertEqual(len(restored.replay), 0)
        self.assertEqual(restored.losses, [])
        inputs = torch.ones(209)
        for original, loaded in ((agent.online, restored.online), (agent.target, restored.target)):
            torch.testing.assert_close(original(inputs), loaded(inputs), rtol=0, atol=0)
        env = HoldemEnv(0)
        env.reset()
        history = OpponentHistory()
        self.assertEqual(
            agent.act(env.observation(), history, np.random.default_rng(1)),
            restored.act(env.observation(), history, np.random.default_rng(2)),
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in restored.target.parameters())
        )

    def test_initialization_preserves_caller_torch_rng(self) -> None:
        before = torch.random.get_rng_state()
        _ = DoubleDQN(99)
        torch.testing.assert_close(torch.random.get_rng_state(), before)


if __name__ == "__main__":
    unittest.main()
