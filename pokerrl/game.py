"""Heads-up Kuhn poker: one-card hands, one betting round, one-chip antes."""

from dataclasses import dataclass
from itertools import permutations
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Action = Literal["check", "bet", "fold", "call"]
HISTORIES = ("", "kb", "k", "b")
TERMINALS = ("kk", "bf", "bc", "kbf", "kbc")
DEALS = tuple(permutations(range(3), 2))
INFOSETS = tuple((card, history) for history in HISTORIES for card in range(3))


@dataclass(frozen=True, slots=True)
class InvalidState(ValueError):
    """Invalid deal, history, or action supplied at the game boundary."""

    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class Observation:
    """An acting player's information; the opponent's card is never included."""

    player: int
    card: int
    history: str
    legal_actions: tuple[Action, Action]


@dataclass(frozen=True, slots=True)
class State:
    """Immutable full simulator state; policies receive only observation()."""

    cards: tuple[int, int]
    history: str = ""

    def __post_init__(self) -> None:
        if self.cards not in DEALS or self.history not in HISTORIES + TERMINALS:
            raise InvalidState("Expected distinct cards in 0..2 and a legal Kuhn history")

    @property
    def terminal(self) -> bool:
        return self.history in TERMINALS

    @property
    def player(self) -> int:
        if self.terminal:
            raise InvalidState("Terminal states have no acting player")
        return len(self.history) % 2

    @property
    def legal_actions(self) -> tuple[Action, Action]:
        if self.terminal:
            raise InvalidState("Terminal states have no legal actions")
        return ("fold", "call") if self.history.endswith("b") else ("check", "bet")

    def observation(self) -> Observation:
        return Observation(self.player, self.cards[self.player], self.history, self.legal_actions)

    def step(self, action: Action) -> "State":
        if action not in self.legal_actions:
            raise InvalidState(f"Illegal action {action!r} at history {self.history!r}")
        symbols = {"check": "k", "bet": "b", "fold": "f", "call": "c"}
        return State(self.cards, self.history + symbols[action])

    def rewards(self) -> tuple[float, float]:
        if not self.terminal:
            raise InvalidState("Rewards are available only at terminal states")
        if self.history.endswith("f"):
            value = 1.0 if self.history == "bf" else -1.0
        else:
            stake = 1.0 if self.history == "kk" else 2.0
            value = stake if self.cards[0] > self.cards[1] else -stake
        return value, -value


def information_index(observation: Observation) -> int:
    """Map a private observation to one of twelve information sets."""
    return HISTORIES.index(observation.history) * 3 + observation.card


def random_policy() -> Array:
    """Uniform legal-action probabilities at each information set."""
    return np.full((12, 2), 0.5)


def equilibrium_policy() -> Array:
    """Kuhn equilibrium with first-player jack bluff probability alpha=1/3."""
    aggressive = np.array([1 / 3, 0, 1, 0, 2 / 3, 1, 1 / 3, 0, 1, 0, 1 / 3, 1])
    return np.column_stack((1 - aggressive, aggressive))


def realization_weights(policy: Array) -> Array:
    """Own reach probability, excluding opponent and chance reach."""
    weights = np.ones(12)
    weights[3:6] = policy[:3, 0]
    return weights
