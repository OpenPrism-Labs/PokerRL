"""Public-observation, same-seat transitions shared by Hold'em RL algorithms."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from pokerrl.holdem import Action, HoldemEnv, Observation
from pokerrl.holdem_learning import OpponentHistory, Policy, abstract_actions, encode

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Decision:
    index: int
    log_prob: float = 0.0
    value: float = 0.0


@dataclass(frozen=True, slots=True)
class Transition:
    inputs: Array
    mask: Mask
    action: int
    reward: float
    next_inputs: Array
    next_mask: Mask
    done: bool
    log_prob: float
    value: float


class DecisionMaker(Protocol):
    def decision(
        self, inputs: Array, mask: Mask, rng: np.random.Generator, *, training: bool
    ) -> Decision: ...


class Agent(DecisionMaker, Protocol):
    updates: int
    losses: list[float]

    def act(
        self, observation: Observation, history: OpponentHistory, rng: np.random.Generator
    ) -> Action: ...

    def learn(self, trajectory: list[Transition]) -> None: ...

    def finish(self) -> None: ...

    def save(self, path: Path) -> None: ...


def collect_hand(
    env: HoldemEnv,
    agent: DecisionMaker,
    opponents: tuple[Policy, ...],
    history: OpponentHistory,
    rng: np.random.Generator,
    *,
    hero: int,
    reward_scale: float = 0.01,
    use_history: bool = True,
) -> tuple[list[Transition], float]:
    """Finish a hand before assigning its net reward to the final hero decision."""
    if env.terminal:
        raise RuntimeError("Collection requires an already-reset live hand")
    if len(opponents) != 6 or not 0 <= hero < 6:
        raise ValueError("Expected six opponents and a valid hero seat")
    if not np.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError("Reward scale must be finite and positive")
    big_blind = env.observation().big_blind
    records: list[tuple[Array, Mask, Decision]] = []
    while not env.terminal:
        observation = env.observation()
        if observation.seat == hero:
            actions = abstract_actions(observation)
            inputs = encode(observation, history, use_history=use_history)
            mask = np.array([action is not None for action in actions])
            decision = agent.decision(inputs.copy(), mask.copy(), rng, training=True)
            if not 0 <= decision.index < 6 or not mask[decision.index]:
                raise ValueError("Agent selected an illegal action slot")
            action = actions[decision.index]
            assert action is not None
            records.append((inputs, mask, decision))
        else:
            action = opponents[observation.seat].act(observation, history, rng)
        env.step(action)
    result = env.result()
    if sum(result.payoffs) != 0:
        raise RuntimeError("Settlement violated chip conservation")
    payoff = result.payoffs[hero] / big_blind
    trajectory: list[Transition] = []
    for index, (inputs, mask, decision) in enumerate(records):
        done = index == len(records) - 1
        next_inputs, next_mask = (
            (np.zeros(209), np.zeros(6, dtype=np.bool_))
            if done
            else (records[index + 1][0], records[index + 1][1])
        )
        trajectory.append(
            Transition(
                inputs,
                mask,
                decision.index,
                payoff * reward_scale if done else 0.0,
                next_inputs.copy(),
                next_mask.copy(),
                done,
                decision.log_prob,
                decision.value,
            )
        )
    history.update(result)
    return trajectory, payoff
