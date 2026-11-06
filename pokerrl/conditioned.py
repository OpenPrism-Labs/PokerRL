"""Reward-driven population learning conditioned solely on public action history."""

import json
from pathlib import Path

import numpy as np

from pokerrl.evaluate import Evaluator
from pokerrl.game import Array, random_policy
from pokerrl.neural import Network
from pokerrl.opponents import PublicHistory, population, sample_hand


class ContextNetwork(Network):
    """Reuse exact softmax/tanh backprop and Adam with eight extra public features."""

    def __init__(self, rng: np.random.Generator, player: int) -> None:
        super().__init__(rng, player)
        self.inputs = np.column_stack((self.inputs, np.zeros((6, 8))))
        self.parameters[0] = np.concatenate((self.parameters[0], rng.normal(0, 0.3, (8, 16))))
        self.first = [np.zeros_like(parameter) for parameter in self.parameters]
        self.second = [np.zeros_like(parameter) for parameter in self.parameters]

    def condition(self, context: Array) -> None:
        """Set the same public cross-hand context for every private information set."""
        self.inputs[:, 7:] = context


class OpponentLearner:
    """Two seat policies trained by exact next-hand reward at sampled contexts.

    Gradients stop at the history boundary: this optimizes immediate hand return,
    not the future information value of exploratory actions. Opponent tables are
    known to the training simulator, never given to the policy's forward pass.
    """

    def __init__(self, seed: int, learning_rate: float = 0.003, use_history: bool = True) -> None:
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.learning_rate = learning_rate
        self.use_history = use_history
        self.networks = (ContextNetwork(self.rng, 0), ContextNetwork(self.rng, 1))
        self.evaluator = Evaluator()
        self.opponents = population(1729)
        self.archive: list[Array] = [random_policy()]
        self.episodes = 0
        self.decay = 0.95

    def policy(self, player: int, history: PublicHistory) -> Array:
        """Return six private-observation policy rows using only public history."""
        context = history.features() if self.use_history else np.zeros(8)
        self.networks[player].condition(context)
        return self.networks[player].forward()[0]

    def update(self, opponent: Array, player: int, history: PublicHistory) -> float:
        """One exact reward-ascent update; the opponent table is a training target."""
        section = slice(player * 6, player * 6 + 6)
        profile = opponent.copy()
        profile[section] = self.policy(player, history)
        sign = 1 - 2 * player
        derivatives = self.networks[player].gradients(
            sign * self.evaluator.gradient(profile)[section]
        )
        self.networks[player].ascend(derivatives, self.learning_rate)
        return sign * float(self.evaluator.values(profile))

    def train_episode(self, hands: int = 24) -> None:
        """Sample one opponent per seat; observe successive hands before updating."""
        for player in (0, 1):
            if self.rng.random() < 0.25:
                opponent = self.archive[int(self.rng.integers(len(self.archive)))]
            else:
                opponent = self.opponents[int(self.rng.integers(len(self.opponents)))].policy
            history = PublicHistory(self.decay)
            for _ in range(hands):
                profile = opponent.copy()
                profile[player * 6 : player * 6 + 6] = self.policy(player, history)
                self.update(opponent, player, history)
                history.observe(sample_hand(profile, self.rng), 1 - player)
        self.episodes += 1
        if self.episodes % 20 == 0:
            # Historical policies are frozen zero-context tables, not live opponents.
            empty = PublicHistory(self.decay)
            self.archive.append(np.concatenate([self.policy(p, empty) for p in (0, 1)]))
            self.archive = self.archive[-16:]

    def save(self, path: Path) -> None:
        """Persist weights, Adam, population, historical pool, and RNG for continuation."""
        arrays = {
            f"p{player}_{kind}{index}": value
            for player, network in enumerate(self.networks)
            for kind, values in (
                ("parameter", network.parameters),
                ("first", network.first),
                ("second", network.second),
            )
            for index, value in enumerate(values)
        }
        np.savez_compressed(
            path,
            allow_pickle=False,
            **arrays,
            version=np.array(1),
            seed=np.array(self.seed),
            learning_rate=np.array(self.learning_rate),
            use_history=np.array(self.use_history),
            episodes=np.array(self.episodes),
            decay=np.array(self.decay),
            steps=np.array([network.steps for network in self.networks]),
            archive=np.array(self.archive),
            opponents=np.array([opponent.policy for opponent in self.opponents]),
            rng=np.array(json.dumps(self.rng.bit_generator.state)),
        )

    @classmethod
    def load(cls, path: Path) -> "OpponentLearner":
        """Load for inference or exact continuation at an episode boundary."""
        from pokerrl.game import InvalidState
        from pokerrl.opponents import Opponent

        with np.load(path, allow_pickle=False) as checkpoint:
            if int(checkpoint["version"]) != 1:
                raise InvalidState("Unsupported opponent learner checkpoint version")
            model = cls(
                int(checkpoint["seed"]),
                float(checkpoint["learning_rate"]),
                bool(checkpoint["use_history"]),
            )
            model.episodes = int(checkpoint["episodes"])
            model.decay = float(checkpoint["decay"])
            model.archive = [table.copy() for table in checkpoint["archive"]]
            model.opponents = [
                Opponent(f"training_{index}", table.copy())
                for index, table in enumerate(checkpoint["opponents"])
            ]
            model.rng.bit_generator.state = json.loads(str(checkpoint["rng"]))
            for player, network in enumerate(model.networks):
                network.steps = int(checkpoint["steps"][player])
                for kind, values in (
                    ("parameter", network.parameters),
                    ("first", network.first),
                    ("second", network.second),
                ):
                    for index in range(len(values)):
                        values[index] = checkpoint[f"p{player}_{kind}{index}"].copy()
        return model
