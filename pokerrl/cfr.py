"""Vanilla full-tree simultaneous counterfactual regret minimization."""

import numpy as np

from pokerrl.game import (
    DEALS,
    Array,
    State,
    information_index,
    random_policy,
    realization_weights,
)


class CFR:
    """Regret matching with chance-weighted counterfactual regret updates."""

    def __init__(self) -> None:
        self.regrets = np.zeros((12, 2))
        self.strategy_sum = np.zeros((12, 2))
        self.steps = 0

    def current_policy(self) -> Array:
        positive = np.maximum(self.regrets, 0)
        totals = positive.sum(axis=1, keepdims=True)
        return np.divide(positive, totals, out=random_policy(), where=totals > 0)

    def average_policy(self) -> Array:
        totals = self.strategy_sum.sum(axis=1, keepdims=True)
        return np.divide(self.strategy_sum, totals, out=random_policy(), where=totals > 0)

    def step(self) -> None:
        """One simultaneous update, enumerating all six ordered deals."""
        policy = self.current_policy()
        increments = np.zeros((12, 2))

        def visit(state: State, reach: tuple[float, float]) -> float:
            if state.terminal:
                return state.rewards()[0]
            player = state.player
            index = information_index(state.observation())
            values = np.empty(2)
            for action_index, action in enumerate(state.legal_actions):
                next_reach = list(reach)
                next_reach[player] *= float(policy[index, action_index])
                values[action_index] = visit(state.step(action), (next_reach[0], next_reach[1]))
            value = float(policy[index] @ values)
            sign = 1 if player == 0 else -1
            increments[index] += sign * reach[1 - player] * (values - value) / len(DEALS)
            return value

        for cards in DEALS:
            visit(State(cards), (1.0, 1.0))
        self.regrets += increments
        self.strategy_sum += realization_weights(policy)[:, None] * policy
        self.steps += 1
