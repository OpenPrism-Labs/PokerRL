"""Exact-return neural self-play policy gradient, implemented entirely in NumPy."""

from pathlib import Path

import numpy as np

from pokerrl.evaluate import Evaluator
from pokerrl.game import Array, random_policy, realization_weights


class Network:
    """A private-observation, 7-input / 16-tanh / 2-softmax policy network."""

    def __init__(self, rng: np.random.Generator, player: int) -> None:
        self.inputs = np.zeros((6, 7))
        for row in range(6):
            self.inputs[row, row % 3] = 1
            self.inputs[row, 3 + player * 2 + row // 3] = 1
        self.parameters = [
            rng.normal(0, 0.3, (7, 16)),
            np.zeros(16),
            rng.normal(0, 0.1, (16, 2)),
            np.zeros(2),
        ]
        self.first = [np.zeros_like(parameter) for parameter in self.parameters]
        self.second = [np.zeros_like(parameter) for parameter in self.parameters]
        self.steps = 0

    def forward(self) -> tuple[Array, Array]:
        w1, b1, w2, b2 = self.parameters
        hidden = np.tanh(self.inputs @ w1 + b1)
        logits = hidden @ w2 + b2
        logits -= logits.max(axis=1, keepdims=True)
        exponentials = np.exp(logits)
        return exponentials / exponentials.sum(axis=1, keepdims=True), hidden

    def gradients(self, probability_gradient: Array) -> list[Array]:
        """Backpropagate the expected-return derivative through softmax and tanh."""
        probabilities, hidden = self.forward()
        logits_gradient = probabilities * (
            probability_gradient
            - np.sum(probabilities * probability_gradient, axis=1, keepdims=True)
        )
        hidden_gradient = (logits_gradient @ self.parameters[2].T) * (1 - hidden**2)
        return [
            self.inputs.T @ hidden_gradient,
            hidden_gradient.sum(axis=0),
            hidden.T @ logits_gradient,
            logits_gradient.sum(axis=0),
        ]

    def ascend(self, gradients: list[Array], learning_rate: float) -> None:
        """Adam ascent, with fresh moments local to each player's network."""
        self.steps += 1
        for parameter, first, second, gradient in zip(
            self.parameters, self.first, self.second, gradients, strict=True
        ):
            first *= 0.9
            first += 0.1 * gradient
            second *= 0.999
            second += 0.001 * gradient**2
            corrected_first = first / (1 - 0.9**self.steps)
            corrected_second = second / (1 - 0.999**self.steps)
            parameter += learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)


class NeuralPG:
    """Simultaneous zero-sum self-play ascent using exact expected-return gradients.

    This is full-tree, model-based policy gradient, not NFSP, sampled REINFORCE,
    or an equilibrium solver. The realization-weighted average is evaluated.
    """

    def __init__(self, seed: int, learning_rate: float = 0.01) -> None:
        rng = np.random.default_rng(seed)
        self.networks = (Network(rng, 0), Network(rng, 1))
        self.evaluator = Evaluator()
        self.learning_rate = learning_rate
        self.strategy_sum = np.zeros((12, 2))
        self.steps = 0

    def current_policy(self) -> Array:
        return np.concatenate([network.forward()[0] for network in self.networks])

    def average_policy(self) -> Array:
        totals = self.strategy_sum.sum(axis=1, keepdims=True)
        return np.divide(self.strategy_sum, totals, out=random_policy(), where=totals > 0)

    def step(self) -> None:
        """Accumulate the pre-update profile and update both players simultaneously."""
        policy = self.current_policy()
        self.strategy_sum += realization_weights(policy)[:, None] * policy
        gradient = self.evaluator.gradient(policy)
        gradients = (
            self.networks[0].gradients(gradient[:6]),
            self.networks[1].gradients(-gradient[6:]),
        )
        for network, derivatives in zip(self.networks, gradients, strict=True):
            network.ascend(derivatives, self.learning_rate)
        self.steps += 1

    def save(self, path: Path) -> None:
        """Save trainable weights, Adam state, average policy, and update budget."""
        arrays = {
            f"p{player}_{kind}{index}": array
            for player, network in enumerate(self.networks)
            for kind, values in (
                ("parameter", network.parameters),
                ("first", network.first),
                ("second", network.second),
            )
            for index, array in enumerate(values)
        }
        np.savez_compressed(
            path,
            allow_pickle=False,
            **arrays,
            steps=np.array(self.steps),
            learning_rate=np.array(self.learning_rate),
            strategy_sum=self.strategy_sum,
            current_policy=self.current_policy(),
            average_policy=self.average_policy(),
        )

    @classmethod
    def load(cls, path: Path) -> "NeuralPG":
        """Restore a checkpoint for inference or deterministic continued training."""
        with np.load(path, allow_pickle=False) as checkpoint:
            model = cls(0, float(checkpoint["learning_rate"]))
            model.steps = int(checkpoint["steps"])
            model.strategy_sum = checkpoint["strategy_sum"].copy()
            for player, network in enumerate(model.networks):
                network.steps = model.steps
                for kind, values in (
                    ("parameter", network.parameters),
                    ("first", network.first),
                    ("second", network.second),
                ):
                    for index in range(len(values)):
                        values[index] = checkpoint[f"p{player}_{kind}{index}"].copy()
        return model
