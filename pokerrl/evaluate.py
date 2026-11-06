"""Exact chance enumeration and information-set-consistent best responses."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from pokerrl.game import DEALS, Array, State, information_index


@dataclass(frozen=True, slots=True)
class Metrics:
    """Player-zero utility and half the sum of unilateral deviation gains."""

    value: float
    best_response_p0: float
    best_response_p1: float
    nash_conv: float
    exploitability: float


class Evaluator:
    """Thirty terminal paths evaluated together; no sampled evaluation noise."""

    def __init__(self) -> None:
        paths: list[list[tuple[int, int]]] = []
        payoffs: list[float] = []

        def visit(state: State, path: list[tuple[int, int]]) -> None:
            if state.terminal:
                paths.append(path)
                payoffs.append(state.rewards()[0] / len(DEALS))
                return
            index = information_index(state.observation())
            for action_index, action in enumerate(state.legal_actions):
                visit(state.step(action), [*path, (index, action_index)])

        for cards in DEALS:
            visit(State(cards), [])
        self.indices: NDArray[np.int64] = np.full((30, 3), 12, dtype=np.int64)
        self.actions: NDArray[np.int64] = np.zeros((30, 3), dtype=np.int64)
        for row, path in enumerate(paths):
            for column, (index, action) in enumerate(path):
                self.indices[row, column] = index
                self.actions[row, column] = action
        self.payoffs = np.array(payoffs)
        self.pure = np.array(tuple(product((0, 1), repeat=6)), dtype=np.int64)

    def values(self, profiles: Array) -> Array:
        """Accept a profile (12,2) or a batch (...,12,2); return player-zero values."""
        padding = np.ones((*profiles.shape[:-2], 1, 2))
        extended = np.concatenate((profiles, padding), axis=-2)
        probabilities = extended[..., self.indices, self.actions]
        return np.sum(np.prod(probabilities, axis=-1) * self.payoffs, axis=-1)

    def gradient(self, policy: Array) -> Array:
        """Exact derivative of player-zero expected return w.r.t. probabilities."""
        extended = np.concatenate((policy, np.ones((1, 2))))
        probabilities = extended[self.indices, self.actions]
        gradient = np.zeros((13, 2))
        for column in range(3):
            others = np.prod(np.delete(probabilities, column, axis=1), axis=1)
            np.add.at(
                gradient,
                (self.indices[:, column], self.actions[:, column]),
                self.payoffs * others,
            )
        return gradient[:12]

    def metrics(self, policy: Array) -> Metrics:
        """Each enumerated response fixes one action per private information set."""
        responses = np.eye(2)[self.pure]
        profiles0 = np.broadcast_to(policy, (64, 12, 2)).copy()
        profiles1 = profiles0.copy()
        profiles0[:, :6] = responses
        profiles1[:, 6:] = responses
        upper = float(np.max(self.values(profiles0)))
        lower = float(np.min(self.values(profiles1)))
        gap = upper - lower
        return Metrics(float(self.values(policy)), upper, -lower, gap, gap / 2)
