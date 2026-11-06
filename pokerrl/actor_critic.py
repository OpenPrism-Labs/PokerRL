"""Complete-hand PPO and A2C with a shared 209-hidden-tanh trunk and two heads.

The actor head has six logits and the critic head one scalar. Returns use scaled
terminal rewards supplied by the collector. Checkpoints restore inference weights
and hyperparameters only: optimizer, pending rollout, RNG, counters, environment,
and opponent history are deliberately not restored for exact training resumption.
"""

from pathlib import Path
from typing import Literal, Protocol, TypedDict

import numpy as np
import torch
from torch import Tensor, nn

from pokerrl.holdem import Action, Observation
from pokerrl.holdem_learning import OpponentHistory, abstract_actions, encode
from pokerrl.rl_common import Array, Decision, Mask, Transition


def generalized_advantages(
    rewards: Array, values: Array, dones: Mask, *, gamma: float = 1.0, lam: float = 0.95
) -> tuple[Array, Array]:
    """GAE for a batch ending at a real terminal; never bootstrap across hands."""
    if len(rewards) == 0 or not dones[-1]:
        raise ValueError("GAE requires a nonempty batch of complete hands")
    advantages = np.zeros_like(rewards)
    carry = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = 0.0 if dones[index] else float(values[index + 1])
        continuation = 0.0 if dones[index] else 1.0
        delta = float(rewards[index] - values[index]) + gamma * next_value
        carry = delta + gamma * lam * continuation * carry
        advantages[index] = carry
    return advantages, advantages + values


def normalize_advantages(advantages: Tensor) -> Tensor:
    """Leave singleton/constant advantages intact rather than erase their signal."""
    deviation = advantages.std(correction=0) if advantages.numel() > 1 else advantages.new_zeros(())
    if float(deviation) <= 1e-8:
        return advantages
    return (advantages - advantages.mean()) / deviation


def masked_log_probs(logits: Tensor, masks: Tensor) -> Tensor:
    """Zero probability and zero actor gradient for illegal actions."""
    if not bool(masks.any(dim=-1).all()):
        raise ValueError("each decision must have at least one legal action")
    return torch.log_softmax(logits.masked_fill(~masks, -torch.inf), dim=-1)


def actor_critic_loss(
    logits: Tensor,
    values: Tensor,
    masks: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    returns: Tensor,
    *,
    algorithm: Literal["ppo", "a2c"],
    clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Tensor:
    """PPO clipped surrogate or A2C score loss, plus MSE minus legal entropy."""
    log_probs = masked_log_probs(logits, masks)
    selected = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    advantages = advantages.detach()
    if algorithm == "ppo":
        ratio = (selected - old_log_probs.detach()).exp()
        policy_loss = -torch.minimum(
            ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages
        ).mean()
    else:
        policy_loss = -(selected * advantages).mean()
    # Replace illegal log probabilities before multiplication: 0 * -inf is NaN.
    entropy = -(log_probs.exp() * log_probs.masked_fill(~masks, 0.0)).sum(-1).mean()
    value_loss = (values - returns.detach()).square().mean()
    return policy_loss + value_coef * value_loss - entropy_coef * entropy


class ActorCriticNetwork(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.trunk = nn.Linear(209, hidden)
        self.actor = nn.Linear(hidden, 6)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.trunk.forward(inputs).tanh()
        return self.actor.forward(hidden), self.critic.forward(hidden).squeeze(-1)


class _Checkpoint(TypedDict):
    kind: str
    seed: int
    algorithm: Literal["ppo", "a2c"]
    hidden: int
    learning_rate: float
    use_history: bool
    hyperparameters: dict[str, float | int]
    model: dict[str, Tensor]


class _Optimizer(Protocol):
    """The no-closure optimizer operations used here, typed at the Torch boundary."""

    def zero_grad(self) -> None: ...

    def step(self) -> object: ...


class ActorCritic:
    """On-policy updates after at least 256 transitions, retaining whole hands.

    ``updates`` counts optimizer steps, so PPO normally increments it 16 times
    per 256-transition rollout whereas A2C increments it once. ``finish`` trains
    any remaining complete hands. Collection and evaluation sample via caller RNG.
    """

    def __init__(
        self,
        seed: int,
        algorithm: Literal["ppo", "a2c"],
        *,
        device: str = "cpu",
        hidden: int = 32,
        learning_rate: float = 0.0003,
        use_history: bool = True,
    ) -> None:
        if algorithm not in ("ppo", "a2c"):
            raise ValueError("algorithm must be ppo or a2c")
        if hidden <= 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("hidden width and finite learning rate must be positive")
        self.seed = seed
        self.algorithm: Literal["ppo", "a2c"] = algorithm
        self.device = torch.device(device)
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.use_history = use_history
        self.gamma = 1.0
        self.gae_lambda = 0.95
        self.rollout_threshold = 256
        self.epochs = 4 if algorithm == "ppo" else 1
        self.minibatch_size = 64
        self.clip = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.gradient_norm = 0.5
        random_state = torch.get_rng_state()
        try:
            _ = torch.default_generator.manual_seed(seed)
            self.model = ActorCriticNetwork(hidden).to(self.device)
        finally:
            torch.set_rng_state(random_state)
        self.optimizer: _Optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.rng = np.random.default_rng(seed)
        self.rollout: list[Transition] = []
        self.updates = 0
        self.losses: list[float] = []

    def decision(
        self, inputs: Array, mask: Mask, rng: np.random.Generator, *, training: bool
    ) -> Decision:
        with torch.no_grad():
            logits, value = self.model.forward(
                torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
            )
            log_probs = masked_log_probs(
                logits, torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            )
            probabilities = np.asarray(log_probs.exp().cpu().numpy(), dtype=np.float64)
            probabilities /= probabilities.sum()
            index = int(rng.choice(6, p=probabilities))
            return Decision(index, float(log_probs[index]), float(value))

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
        if not trajectory[-1].done or any(step.done for step in trajectory[:-1]):
            raise ValueError("learn expects exactly one complete hand")
        self.rollout.extend(trajectory)
        if len(self.rollout) >= self.rollout_threshold:
            self._update()

    def finish(self) -> None:
        if self.rollout:
            self._update()

    def _update(self) -> None:
        rollout = self.rollout
        advantages_array, returns_array = generalized_advantages(
            np.array([step.reward for step in rollout]),
            np.array([step.value for step in rollout]),
            np.array([step.done for step in rollout]),
            gamma=self.gamma,
            lam=self.gae_lambda,
        )
        inputs = torch.tensor(
            np.stack([step.inputs for step in rollout]), dtype=torch.float32, device=self.device
        )
        masks = torch.tensor(
            np.stack([step.mask for step in rollout]), dtype=torch.bool, device=self.device
        )
        actions = torch.tensor([step.action for step in rollout], device=self.device)
        # Copies of the behavior statistics remain fixed for every PPO epoch.
        old_log_probs = torch.tensor(
            [step.log_prob for step in rollout], dtype=torch.float32, device=self.device
        )
        returns = torch.tensor(returns_array, dtype=torch.float32, device=self.device)
        advantages = normalize_advantages(
            torch.tensor(advantages_array, dtype=torch.float32, device=self.device)
        )
        size = len(rollout)
        batch_size = self.minibatch_size if self.algorithm == "ppo" else size
        for _ in range(self.epochs):
            order = self.rng.permutation(size) if self.algorithm == "ppo" else np.arange(size)
            for start in range(0, size, batch_size):
                indices = torch.tensor(order[start : start + batch_size], device=self.device)
                logits, values = self.model.forward(inputs[indices])
                loss = actor_critic_loss(
                    logits,
                    values,
                    masks[indices],
                    actions[indices],
                    old_log_probs[indices],
                    advantages[indices],
                    returns[indices],
                    algorithm=self.algorithm,
                    clip=self.clip,
                    value_coef=self.value_coef,
                    entropy_coef=self.entropy_coef,
                )
                self.optimizer.zero_grad()
                torch.autograd.backward(loss)
                _ = nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_norm)
                self.optimizer.step()
                self.losses.append(float(loss.detach()))
                self.updates += 1
        self.rollout.clear()

    def save(self, path: Path) -> None:
        """Save inference only; does not flush or serialize a pending rollout."""
        checkpoint: _Checkpoint = {
            "kind": "actor_critic_inference_v1",
            "seed": self.seed,
            "algorithm": self.algorithm,
            "hidden": self.hidden,
            "learning_rate": self.learning_rate,
            "use_history": self.use_history,
            "hyperparameters": {
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "rollout_threshold": self.rollout_threshold,
                "epochs": self.epochs,
                "minibatch_size": self.minibatch_size,
                "clip": self.clip,
                "entropy_coef": self.entropy_coef,
                "value_coef": self.value_coef,
                "gradient_norm": self.gradient_norm,
            },
            "model": dict(self.model.state_dict()),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> "ActorCritic":
        """Restore weights/hyperparameters with a new optimizer and empty rollout."""
        data: _Checkpoint = torch.load(path, map_location=device, weights_only=True)
        if data["kind"] != "actor_critic_inference_v1":
            raise ValueError("unsupported actor-critic checkpoint")
        agent = cls(
            data["seed"],
            data["algorithm"],
            device=device,
            hidden=data["hidden"],
            learning_rate=data["learning_rate"],
            use_history=data["use_history"],
        )
        _ = agent.model.load_state_dict(data["model"])
        hyperparameters = data["hyperparameters"]
        agent.gamma = float(hyperparameters["gamma"])
        agent.gae_lambda = float(hyperparameters["gae_lambda"])
        agent.rollout_threshold = int(hyperparameters["rollout_threshold"])
        agent.epochs = int(hyperparameters["epochs"])
        agent.minibatch_size = int(hyperparameters["minibatch_size"])
        agent.clip = float(hyperparameters["clip"])
        agent.entropy_coef = float(hyperparameters["entropy_coef"])
        agent.value_coef = float(hyperparameters["value_coef"])
        agent.gradient_norm = float(hyperparameters["gradient_norm"])
        return agent
