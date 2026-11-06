"""Masked Double DQN with uniform replay and complete-hand update boundaries.

The Q network is 209-hidden-tanh-6. Checkpoints retain both networks, constructor
settings, update count and exploration progress for inference or a warm restart.
They omit replay, replay RNG, optimizer moments and loss history, so loading is
not exact training continuation. ``finish`` does nothing: learning is online.
"""

from collections import deque
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pokerrl.holdem import Action, Observation
from pokerrl.holdem_learning import OpponentHistory, abstract_actions, encode
from pokerrl.rl_common import Array, Decision, Mask, Transition


class _Checkpoint(TypedDict):
    kind: str
    seed: int
    hidden: int
    learning_rate: float
    use_history: bool
    updates: int
    decisions: int
    online: dict[str, Tensor]
    target: dict[str, Tensor]


class _Optimizer(Protocol):
    """Only the optimizer operations this learner uses."""

    def zero_grad(self) -> None: ...

    def step(self) -> None: ...


def double_dqn_targets(
    online_next: Tensor,
    target_next: Tensor,
    next_mask: Tensor,
    rewards: Tensor,
    dones: Tensor,
) -> Tensor:
    """Return detached gamma-one targets, selecting online and evaluating target.

    Terminal rows are excluded before masked argmax, including the usual
    all-false terminal mask. Nonterminal rows must have at least one legal move.
    """
    with torch.no_grad():
        targets = rewards.clone()
        live = ~dones
        if bool(live.any()):
            masks = next_mask[live]
            if not bool(masks.any(dim=1).all()):
                raise ValueError("nonterminal next state must have a legal action")
            selected = online_next[live].masked_fill(~masks, -torch.inf).argmax(dim=1)
            targets[live] += target_next[live].gather(1, selected[:, None]).squeeze(1)
        return targets


class DoubleDQN:
    """209-hidden-6 tanh Q network; Adam, Huber loss and hard target updates.

    Replay holds 20,000 transitions, warms up at 128, and samples 64 without
    replacement once per nonempty ``learn(hand)``. Gamma is one, gradient norm
    is capped at 0.5, and the target synchronizes every 100 optimizer updates.
    Training epsilon decreases linearly from 1 to 0.05 over 10,000 decisions.
    """

    def __init__(
        self,
        seed: int,
        *,
        device: str = "cpu",
        hidden: int = 32,
        learning_rate: float = 0.0003,
        use_history: bool = True,
    ) -> None:
        if hidden <= 0:
            raise ValueError("hidden width must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        self.seed = seed
        self.device = torch.device(device)
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.use_history = use_history
        # Initialize on CPU without perturbing the caller's torch RNG.
        caller_rng = torch.random.get_rng_state()
        try:
            torch.random.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            self.online = nn.Sequential(nn.Linear(209, hidden), nn.Tanh(), nn.Linear(hidden, 6))
            self.target = nn.Sequential(nn.Linear(209, hidden), nn.Tanh(), nn.Linear(hidden, 6))
        finally:
            torch.random.set_rng_state(caller_rng)
        self.online = self.online.to(self.device)
        self.target = self.target.to(self.device)
        _ = self.target.load_state_dict(self.online.state_dict())
        _ = self.target.requires_grad_(False)
        _ = self.target.eval()
        self.optimizer: _Optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay: deque[Transition] = deque(maxlen=20_000)
        self.replay_rng = np.random.default_rng(seed)
        self.decisions = 0
        self.updates = 0
        self.losses: list[float] = []

    @property
    def epsilon(self) -> float:
        return 1.0 - 0.95 * min(self.decisions / 10_000, 1.0)

    def decision(
        self, inputs: Array, mask: Mask, rng: np.random.Generator, *, training: bool
    ) -> Decision:
        legal = np.flatnonzero(mask)
        if not len(legal):
            raise ValueError("decision requires a legal action")
        epsilon = self.epsilon if training else 0.0
        if training:
            self.decisions += 1
        if training and rng.random() < epsilon:
            return Decision(int(rng.choice(legal)))
        with torch.no_grad():
            values: Tensor = self.online(
                torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
            )
            valid = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            return Decision(int(values.masked_fill(~valid, -torch.inf).argmax().item()))

    def act(
        self, observation: Observation, history: OpponentHistory, rng: np.random.Generator
    ) -> Action:
        actions = abstract_actions(observation)
        mask = np.array([action is not None for action in actions])
        inputs = encode(observation, history, use_history=self.use_history)
        action = actions[self.decision(inputs, mask, rng, training=False).index]
        assert action is not None
        return action

    def learn(self, trajectory: list[Transition]) -> None:
        if not trajectory:
            return
        if not trajectory[-1].done:
            raise ValueError("learn requires a complete hand")
        for transition in trajectory:
            if not 0 <= transition.action < 6 or not transition.mask[transition.action]:
                raise ValueError("replay action must be legal")
        self.replay.extend(trajectory)
        if len(self.replay) < 128:
            return
        indices = self.replay_rng.choice(len(self.replay), size=64, replace=False)
        batch = [self.replay[int(index)] for index in indices]
        inputs = torch.as_tensor(
            np.stack([step.inputs for step in batch]), dtype=torch.float32, device=self.device
        )
        next_inputs = torch.as_tensor(
            np.stack([step.next_inputs for step in batch]), dtype=torch.float32, device=self.device
        )
        next_mask = torch.as_tensor(
            np.stack([step.next_mask for step in batch]), dtype=torch.bool, device=self.device
        )
        actions = torch.tensor([step.action for step in batch], device=self.device)
        rewards = torch.tensor(
            [step.reward for step in batch], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor([step.done for step in batch], device=self.device)
        with torch.no_grad():
            targets = double_dqn_targets(
                self.online(next_inputs),
                self.target(next_inputs),
                next_mask,
                rewards,
                dones,
            )
        values: Tensor = self.online(inputs)
        predicted = values.gather(1, actions[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad()
        torch.autograd.backward(loss)
        _ = nn.utils.clip_grad_norm_(self.online.parameters(), 0.5)
        self.optimizer.step()
        self.updates += 1
        self.losses.append(float(loss.detach().item()))
        if self.updates % 100 == 0:
            _ = self.target.load_state_dict(self.online.state_dict())

    def finish(self) -> None:
        """No pending rollout: replay updates already occur in ``learn``."""

    def save(self, path: Path) -> None:
        checkpoint: _Checkpoint = {
            "kind": "double_dqn_inference_v1",
            "seed": self.seed,
            "hidden": self.hidden,
            "learning_rate": self.learning_rate,
            "use_history": self.use_history,
            "updates": self.updates,
            "decisions": self.decisions,
            "online": dict(self.online.state_dict()),
            "target": dict(self.target.state_dict()),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> "DoubleDQN":
        """Restore inference and schedules, with empty replay and fresh Adam."""
        checkpoint: _Checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint["kind"] != "double_dqn_inference_v1":
            raise ValueError("not a DoubleDQN inference checkpoint")
        agent = cls(
            checkpoint["seed"],
            device=device,
            hidden=checkpoint["hidden"],
            learning_rate=checkpoint["learning_rate"],
            use_history=checkpoint["use_history"],
        )
        _ = agent.online.load_state_dict(checkpoint["online"])
        _ = agent.target.load_state_dict(checkpoint["target"])
        agent.updates = checkpoint["updates"]
        agent.decisions = checkpoint["decisions"]
        return agent
