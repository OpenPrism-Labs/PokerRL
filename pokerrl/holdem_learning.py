"""Public-observation Hold'em policies and sampled terminal-return REINFORCE.

Policy archives support inference. Trainer archives additionally preserve Adam,
the prior-return baseline and action RNG. Environment and opponent history belong
to the caller and must also be retained to resume an entire experiment.
"""

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, assert_never

import numpy as np
from numpy.typing import NDArray

from pokerrl.holdem import Action, HandResult, HoldemEnv, Observation

Array = NDArray[np.float64]


class OpponentHistory:
    """Accumulate public decisions from a bounded window of completed hands."""

    def __init__(self, window: int = 32) -> None:
        if window <= 0:
            raise ValueError("history window must be positive")
        self.window = window
        self._hands: deque[Array] = deque(maxlen=window)

    def update(self, result: HandResult) -> None:
        counts = np.zeros((6, 7))
        for action in result.actions:
            row = counts[action.seat]
            row[3] += 1
            row[6] += action.faced_bet
            match action.kind:
                case "raise_to":
                    row[2] += 1
                    if action.street == 0:
                        row[0:2] = 1
                case "check_call":
                    row[4] += action.faced_bet
                    if action.street == 0 and action.paid > 0:
                        row[0] = 1
                case "fold":
                    row[5] += action.faced_bet
                case _:
                    assert_never(action.kind)
        self._hands.append(counts)

    def clear(self) -> None:
        self._hands.clear()

    def features(self, observer: int) -> Array:
        if not 0 <= observer < 6:
            raise ValueError("observer must be a seat from zero to five")
        features = np.zeros((6, 6))
        if self._hands:
            totals = np.sum(np.stack(self._hands), axis=0)
            features[:, 0] = len(self._hands) / self.window
            features[:, 1:3] = totals[:, :2] / len(self._hands)
            features[:, 3] = np.divide(
                totals[:, 2], totals[:, 3], out=np.zeros(6), where=totals[:, 3] > 0
            )
            features[:, 4:6] = np.divide(
                totals[:, 4:6],
                totals[:, 6, None],
                out=np.zeros((6, 2)),
                where=totals[:, 6, None] > 0,
            )
        features = np.roll(features, -observer, axis=0)
        features[0] = 0
        return features


def abstract_actions(observation: Observation) -> tuple[Action | None, ...]:
    """Map six fixed slots to distinct legal actions, retaining earliest ties."""
    legal = observation.legal
    actions: list[Action | None] = [
        Action("fold") if legal.can_fold else None,
        Action("check_call") if legal.can_check_call else None,
        None,
        None,
        None,
        None,
    ]
    low, high = legal.min_raise_to, legal.max_raise_to
    if low is not None and high is not None:
        base = max(observation.bets)
        pot = observation.pot + legal.call_amount
        targets = (low, base + pot // 2, base + pot, high)
        seen: set[int] = set()
        for slot, target in enumerate(targets, start=2):
            amount = min(high, max(low, target))
            if amount not in seen:
                actions[slot] = Action("raise_to", amount)
                seen.add(amount)
    return tuple(actions)


def encode(observation: Observation, history: OpponentHistory, *, use_history: bool) -> Array:
    """Encode only the acting player's cards and the public information boundary."""
    values = np.zeros(209)
    values[list(observation.hole)] = 1
    values[[52 + card for card in observation.board]] = 1
    values[104 + observation.street] = 1
    values[108 + (observation.button - observation.seat) % 6] = 1
    latest = [0] * 6
    for action in observation.actions:
        latest[action.seat] = {"fold": 1, "check_call": 2, "raise_to": 3}[action.kind]
    for relative in range(6):
        seat = (relative + observation.seat) % 6
        start = 114 + 9 * relative
        values[start : start + 3] = np.log1p(
            np.array(
                [
                    observation.stacks[seat],
                    observation.bets[seat],
                    observation.contributions[seat],
                ]
            )
            / observation.big_blind
        )
        values[start + 3 : start + 5] = observation.active[seat], observation.all_in[seat]
        values[start + 5 + latest[seat]] = 1
    legal = observation.legal
    values[168:172] = np.log1p(
        np.array(
            [
                observation.pot,
                legal.call_amount,
                legal.min_raise_to or 0,
                legal.max_raise_to or 0,
            ]
        )
        / observation.big_blind
    )
    values[172] = legal.can_fold
    if use_history:
        values[173:] = history.features(observation.seat).ravel()
    return values


class Policy(Protocol):
    def act(
        self, observation: Observation, history: OpponentHistory, rng: np.random.Generator
    ) -> Action: ...


class NeuralPolicy:
    """Mutable 209-hidden-6 tanh network with an exactly masked softmax."""

    def __init__(self, seed: int, *, use_history: bool = True, hidden: int = 32) -> None:
        if hidden <= 0:
            raise ValueError("hidden width must be positive")
        self.use_history = use_history
        self.hidden = hidden
        rng = np.random.default_rng(seed)
        self.parameters: list[Array] = [
            rng.normal(0, 1 / np.sqrt(209), (209, hidden)),
            np.zeros(hidden),
            rng.normal(0, 1 / np.sqrt(hidden), (hidden, 6)),
            np.zeros(6),
        ]

    def forward(self, inputs: Array, mask: NDArray[np.bool_]) -> tuple[Array, Array]:
        w1, b1, w2, b2 = self.parameters
        hidden = np.tanh(inputs @ w1 + b1)
        logits = hidden @ w2 + b2
        probabilities = np.zeros(6)
        probabilities[mask] = np.exp(logits[mask] - logits[mask].max())
        probabilities /= probabilities.sum()
        return probabilities, hidden

    def probabilities(self, observation: Observation, history: OpponentHistory) -> Array:
        mask = np.array([action is not None for action in abstract_actions(observation)])
        return self.forward(encode(observation, history, use_history=self.use_history), mask)[0]

    def act(
        self, observation: Observation, history: OpponentHistory, rng: np.random.Generator
    ) -> Action:
        actions = abstract_actions(observation)
        action = actions[int(rng.choice(6, p=self.probabilities(observation, history)))]
        assert action is not None
        return action

    def log_gradient(
        self, inputs: Array, probabilities: Array, hidden: Array, selected: int
    ) -> list[Array]:
        logits_gradient = -probabilities.copy()
        logits_gradient[selected] += 1
        hidden_gradient = (self.parameters[2] @ logits_gradient) * (1 - hidden**2)
        return [
            np.outer(inputs, hidden_gradient),
            hidden_gradient,
            np.outer(hidden, logits_gradient),
            logits_gradient,
        ]

    def save(self, path: Path) -> None:
        """Save an inference archive; use Trainer.save for optimizer continuation."""
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                allow_pickle=False,
                kind=np.array("inference"),
                hidden=np.array(self.hidden),
                use_history=np.array(self.use_history),
                **{f"parameter{i}": value for i, value in enumerate(self.parameters)},
            )

    @classmethod
    def load(cls, path: Path) -> "NeuralPolicy":
        """Load policy weights from either an inference or trainer archive."""
        with np.load(path, allow_pickle=False) as data:
            policy = cls(0, hidden=int(data["hidden"]), use_history=bool(data["use_history"]))
            for i, parameter in enumerate(policy.parameters):
                parameter[:] = data[f"parameter{i}"]
        return policy


@dataclass(frozen=True, slots=True)
class ScriptedPolicy:
    """Public-card heuristics; randomness comes only from the supplied generator."""

    style: Literal["random", "loose", "tight", "aggressive"]
    aggression: float = 0.5

    def __post_init__(self) -> None:
        if not np.isfinite(self.aggression) or not 0 <= self.aggression <= 1:
            raise ValueError("aggression must be finite and between zero and one")
        if self.style not in ("random", "loose", "tight", "aggressive"):
            raise ValueError("unknown scripted style")

    def act(
        self, observation: Observation, history: OpponentHistory, rng: np.random.Generator
    ) -> Action:
        actions = abstract_actions(observation)
        available = [action for action in actions if action is not None]
        ranks = [card // 4 for card in observation.hole]
        strength = (sum(ranks) / 24 + (0.4 if ranks[0] == ranks[1] else 0)) / 1.4
        match self.style:
            case "random":
                return available[int(rng.integers(len(available)))]
            case "loose":
                threshold, raise_chance = 0.1, 0.25 * self.aggression
            case "tight":
                threshold, raise_chance = 0.5, self.aggression * strength
            case "aggressive":
                threshold, raise_chance = 0.2, self.aggression
            case _:
                assert_never(self.style)
        if actions[0] is not None and strength < threshold:
            return actions[0]
        raises = [action for action in actions[2:] if action is not None]
        if raises and rng.random() < raise_chance:
            return raises[int(rng.integers(len(raises)))]
        if actions[1] is not None:
            return actions[1]
        return available[0]


@dataclass(frozen=True, slots=True)
class HandSample:
    hero: int
    payoff_bb: float
    decisions: int


class Trainer:
    """Accumulate frozen-hand score gradients, then perform one Adam ascent step."""

    def __init__(self, seed: int, *, use_history: bool, learning_rate: float = 0.001) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        self.policy = NeuralPolicy(seed, use_history=use_history)
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        self.first = [np.zeros_like(value) for value in self.policy.parameters]
        self.second = [np.zeros_like(value) for value in self.policy.parameters]
        self.steps = 0
        self.hands = 0
        self.baseline = 0.0

    def train_hand(
        self,
        env: HoldemEnv,
        opponents: tuple[Policy, ...],
        history: OpponentHistory,
        *,
        hero: int,
    ) -> HandSample:
        if not 0 <= hero < 6 or len(opponents) != 6:
            raise ValueError("hero and opponents must describe six seats")
        if env.terminal or env.actor is None:
            raise RuntimeError("train_hand requires an already-reset live hand")
        gradients = [np.zeros_like(value) for value in self.policy.parameters]
        decisions = 0
        big_blind = env.observation().big_blind
        baseline = self.baseline
        while not env.terminal:
            observation = env.observation()
            if observation.seat == hero:
                actions = abstract_actions(observation)
                inputs = encode(observation, history, use_history=self.policy.use_history)
                mask = np.array([action is not None for action in actions])
                probabilities, hidden = self.policy.forward(inputs, mask)
                selected = int(self.rng.choice(6, p=probabilities))
                action = actions[selected]
                assert action is not None
                for total, gradient in zip(
                    gradients,
                    self.policy.log_gradient(inputs, probabilities, hidden, selected),
                    strict=True,
                ):
                    total += gradient
                decisions += 1
            else:
                action = opponents[observation.seat].act(observation, history, self.rng)
            env.step(action)
        result = env.result()
        payoff = result.payoffs[hero] / big_blind
        if decisions:
            self.steps += 1
            for parameter, first, second, gradient in zip(
                self.policy.parameters, self.first, self.second, gradients, strict=True
            ):
                gradient *= payoff - baseline
                first *= 0.9
                first += 0.1 * gradient
                second *= 0.999
                second += 0.001 * gradient**2
                parameter += (
                    self.learning_rate
                    * (first / (1 - 0.9**self.steps))
                    / (np.sqrt(second / (1 - 0.999**self.steps)) + 1e-8)
                )
        self.hands += 1
        self.baseline += (payoff - self.baseline) / self.hands
        history.update(result)
        return HandSample(hero, payoff, decisions)

    def save(self, path: Path) -> None:
        """Save trainer continuation; caller separately retains env and history."""
        arrays = {
            f"{kind}{i}": value
            for kind, values in (
                ("parameter", self.policy.parameters),
                ("first", self.first),
                ("second", self.second),
            )
            for i, value in enumerate(values)
        }
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                allow_pickle=False,
                **arrays,
                kind=np.array("trainer"),
                hidden=np.array(self.policy.hidden),
                use_history=np.array(self.policy.use_history),
                learning_rate=np.array(self.learning_rate),
                steps=np.array(self.steps),
                hands=np.array(self.hands),
                baseline=np.array(self.baseline),
                rng=np.array(json.dumps(self.rng.bit_generator.state)),
            )

    @classmethod
    def load(cls, path: Path) -> "Trainer":
        with np.load(path, allow_pickle=False) as data:
            if str(data["kind"]) != "trainer":
                raise ValueError("inference archive cannot resume trainer state")
            trainer = cls(
                0,
                use_history=bool(data["use_history"]),
                learning_rate=float(data["learning_rate"]),
            )
            trainer.steps = int(data["steps"])
            trainer.hands = int(data["hands"])
            trainer.baseline = float(data["baseline"])
            trainer.rng.bit_generator.state = json.loads(str(data["rng"]))
            for kind, values in (
                ("parameter", trainer.policy.parameters),
                ("first", trainer.first),
                ("second", trainer.second),
            ):
                for i, value in enumerate(values):
                    value[:] = data[f"{kind}{i}"]
        return trainer
