"""Public-history opponent learning: python -m pokerrl.opponent_experiment."""

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pokerrl.conditioned import OpponentLearner
from pokerrl.evaluate import Evaluator
from pokerrl.game import Array, equilibrium_policy
from pokerrl.opponents import Opponent, PublicHistory, population, sample_hand


@dataclass(frozen=True, slots=True)
class SeatMetrics:
    expected_return: float
    worst_case_return: float
    vulnerability: float


def seat_metrics(policy: Array, opponent: Array, player: int) -> SeatMetrics:
    """Exact next-hand reward and a best response to the frozen-context seat policy.

    Vulnerability = equilibrium seat value minus its worst-case return. The other
    learner seat is never paired with this seat to claim adaptive exploitability.
    """
    evaluator = Evaluator()
    profile = opponent.copy()
    profile[player * 6 : player * 6 + 6] = policy
    expected = (1 - 2 * player) * float(evaluator.values(profile))
    metrics = evaluator.metrics(profile)
    worst = -metrics.best_response_p1 if player == 0 else -metrics.best_response_p0
    equilibrium_value = (2 * player - 1) / 18
    return SeatMetrics(expected, worst, equilibrium_value - worst)


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    algorithm: str
    seed: int
    scenario: str
    player: int
    hand: int
    phase: str
    opponent: str
    expected_return: float
    worst_case_return: float
    vulnerability: float
    history_support: float


def evaluate(
    model: OpponentLearner | None,
    opponents: tuple[Opponent, ...],
    seed: int,
    *,
    hands: int,
    scenario: str,
    algorithm: str,
) -> list[EvaluationRow]:
    """Frozen weights; a fresh public history per match, retained across shifts.

    A tuple of opponents changes at equal hand intervals. Neither the model nor
    PublicHistory receives a change flag. Rows score before observing that hand.
    """
    rows: list[EvaluationRow] = []
    for player in (0, 1):
        rng = np.random.default_rng(np.random.SeedSequence([seed, player, 19001]))
        history = PublicHistory()
        for hand in range(hands * len(opponents)):
            phase = hand // hands
            opponent = opponents[phase]
            policy = (
                equilibrium_policy()[player * 6 : player * 6 + 6]
                if model is None
                else model.policy(player, history)
            )
            metrics = seat_metrics(policy, opponent.policy, player)
            rows.append(
                EvaluationRow(
                    algorithm,
                    seed,
                    scenario,
                    player,
                    hand,
                    f"phase_{phase}",
                    opponent.name,
                    metrics.expected_return,
                    metrics.worst_case_return,
                    metrics.vulnerability,
                    float(history.counts.sum()),
                )
            )
            profile = opponent.policy.copy()
            profile[player * 6 : player * 6 + 6] = policy
            history.observe(sample_hand(profile, rng), 1 - player)
    return rows


def main() -> None:
    """Train matched history/zero-history models, save checkpoints, test held-out play."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("results/opponent"))
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--hands-per-episode", type=int, default=24)
    parser.add_argument("--eval-hands", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    args = parser.parse_args()
    if min(args.episodes, args.hands_per_episode, args.eval_hands) < 1:
        parser.error("Training and evaluation budgets must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive")
    if any(seed < 0 for seed in args.seeds):
        parser.error("Seeds must be nonnegative")
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    test_pool = population(8675309, held_out=True)
    scenarios: list[tuple[str, tuple[Opponent, ...]]] = [
        (opponent.name, (opponent,)) for opponent in test_pool
    ]
    scenarios.append(("shifting", (test_pool[0], test_pool[4], test_pool[8])))
    configuration = {
        "seeds": args.seeds,
        "episodes": args.episodes,
        "hands_per_episode": args.hands_per_episode,
        "eval_hands": args.eval_hands,
        "learning_rate": args.learning_rate,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "training_population_seed": 1729,
        "test_population_seed": 8675309,
        "evaluation_rng": "SeedSequence([run_seed, player, 19001]); separate from training RNG",
        "architecture": "Separate seat MLPs: 15 inputs, 16 tanh, 2 softmax; Adam reward ascent",
        "features": "7 private/public observation features + 8 decayed public action counts",
        "context": "Per history/action count divided by 2 + history total; decay 0.95 per hand",
        "ablation": "Separately trained zero-history population learner; matched seed and budget",
        "fixed_baseline": "Analytic Kuhn equilibrium, no learning",
        "training": "Exact next-hand reward gradients at sampled public contexts; no labels",
        "history_pool": "25% historical frozen zero-context policies; every 20 episodes; cap 16",
        "shift": "Three equal phases; history retained without change notification",
        "metrics": {
            "expected_return": "Exact six-deal next-hand return at sampled pre-hand context",
            "worst_case_return": "Minimum frozen-seat return over 64 consistent pure responses",
            "vulnerability": "Equilibrium seat value (-1/18, +1/18) minus worst-case return",
            "limitation": "Frozen-context vulnerability is not full adaptive-agent exploitability",
        },
        "checkpoint_load": "OpponentLearner.load(Path(...)); policy(player, PublicHistory)",
        "limitations": [
            "History sampling noise remains; only next-hand integration is exact",
            "Immediate reward gradient ignores the future information value of actions",
            "Training simulator knows opponent tables; deployment policy does not",
            "Historical opponents are frozen tables, not independently adaptive agents",
            "No hyperparameter tuning on held-out evaluations",
        ],
        "train_opponents": [
            {"name": opponent.name, "policy": opponent.policy.tolist()}
            for opponent in population(1729)
        ],
        "test_opponents": [
            {"name": opponent.name, "policy": opponent.policy.tolist()} for opponent in test_pool
        ],
    }
    (output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    rows: list[EvaluationRow] = []
    timings: list[dict[str, str | int | float]] = []
    for seed in args.seeds:
        for algorithm in ("history", "zero_history", "equilibrium"):
            model = None
            start = time.perf_counter()
            if algorithm != "equilibrium":
                model = OpponentLearner(seed, args.learning_rate, algorithm == "history")
                for _ in range(args.episodes):
                    model.train_episode(args.hands_per_episode)
                model.save(output / f"{algorithm}_seed{seed}_checkpoint.npz")
            training_seconds = time.perf_counter() - start
            for scenario, opponents in scenarios:
                rows.extend(
                    evaluate(
                        model,
                        opponents,
                        seed,
                        hands=args.eval_hands,
                        scenario=scenario,
                        algorithm=algorithm,
                    )
                )
            timings.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "training_seconds": training_seconds,
                    "wall_seconds": time.perf_counter() - start,
                }
            )
            print(f"{algorithm} seed={seed} wall={time.perf_counter() - start:.2f}s", flush=True)
    with (output / "evaluation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    aggregates: list[dict[str, str | int | float]] = []
    keys = sorted({(row.algorithm, row.scenario, row.player, row.phase) for row in rows})
    for algorithm, scenario, player, phase in keys:
        selected = [
            row
            for row in rows
            if (row.algorithm, row.scenario, row.player, row.phase)
            == (algorithm, scenario, player, phase)
        ]
        entry: dict[str, str | int | float] = {
            "algorithm": algorithm,
            "scenario": scenario,
            "player": player,
            "phase": phase,
        }
        for metric in ("expected_return", "worst_case_return", "vulnerability"):
            means = [
                float(np.mean([getattr(row, metric) for row in selected if row.seed == seed]))
                for seed in args.seeds
            ]
            entry[f"{metric}_mean"] = float(np.mean(means))
            entry[f"{metric}_std"] = float(np.std(means, ddof=int(len(means) > 1)))
        aggregates.append(entry)
    summary = {
        "aggregate": aggregates,
        "timings": timings,
        "std_definition": "Sample standard deviation of per-seed hand means; zero for one seed",
        "rows": len(rows),
    }
    # allow_nan=False makes nonfinite output a failed experiment.
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
