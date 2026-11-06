"""Public opponent observations and reproducible training/test populations."""

from dataclasses import dataclass

import numpy as np

from pokerrl.game import DEALS, HISTORIES, TERMINALS, Array, InvalidState, State, information_index


class PublicHistory:
    """Mutable decayed counts of visible opponent actions, updated after each hand.

    No cards, rewards, opponent identifiers, or policy parameters enter this object.
    A hand's context is frozen until the hand ends. Decay is per hand, including
    hands where a particular decision is not reached.
    """

    def __init__(self, decay: float = 0.95) -> None:
        if not np.isfinite(decay) or not 0 < decay <= 1:
            raise InvalidState("History decay must be in (0, 1]")
        self.decay = decay
        self.counts = np.zeros((4, 2))

    def observe(self, terminal_history: str, opponent: int) -> None:
        """Consume only the public terminal action string and opponent seat."""
        if terminal_history not in TERMINALS or opponent not in (0, 1):
            raise InvalidState("Expected a terminal public history and opponent seat")
        self.counts *= self.decay
        for turn, symbol in enumerate(terminal_history):
            if turn % 2 == opponent:
                row = HISTORIES.index(terminal_history[:turn])
                self.counts[row, int(symbol in ("b", "c"))] += 1

    def features(self) -> Array:
        """Eight bounded counts encode action frequency and observation support."""
        return (self.counts / (2 + self.counts.sum(axis=1, keepdims=True))).ravel()


@dataclass(frozen=True, slots=True)
class Opponent:
    """A simulator-only policy table; its label is for reporting, never inference."""

    name: str
    policy: Array


def population(seed: int, held_out: bool = False) -> list[Opponent]:
    """Separate parameter draws; held-out pool also contains unseen mixed styles."""
    rng = np.random.default_rng(seed)
    opponents: list[Opponent] = []
    for style in ("overfold", "bluffer", "passive_call"):
        for variant in range(4):
            low = float(rng.uniform(0.02, 0.12))
            high = float(rng.uniform(0.78, 0.98))
            middle = float(rng.uniform(0.25, 0.55))
            match style:
                case "overfold":
                    bets = [middle, low, high]
                    calls = [low, low, middle]
                case "bluffer":
                    bets = [high, high, high]
                    calls = [middle, high, high]
                case "passive_call":
                    bets = [low, low, middle]
                    calls = [middle, high, high]
            aggressive = np.array(bets + calls + bets + calls)
            if held_out:
                # New parameterizations plus seat/history asymmetry not seen in training.
                aggressive = np.clip(aggressive + rng.uniform(-0.10, 0.10, size=12), 0.01, 0.99)
            opponents.append(
                Opponent(f"{style}_{variant}", np.column_stack((1 - aggressive, aggressive)))
            )
    if held_out:
        for index in range(3):
            blend = 0.3 + 0.2 * index
            policy = blend * opponents[index].policy + (1 - blend) * opponents[8 + index].policy
            opponents.append(Opponent(f"unseen_mix_{index}", policy))
    return opponents


def sample_hand(profile: Array, rng: np.random.Generator) -> str:
    """Simulator samples hidden deals internally and returns public actions only."""
    state = State(DEALS[int(rng.integers(len(DEALS)))])
    while not state.terminal:
        index = information_index(state.observation())
        action = int(rng.random() < profile[index, 1])
        state = state.step(state.legal_actions[action])
    return state.history
