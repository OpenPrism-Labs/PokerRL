"""Run the CPU experiment: python -m pokerrl.experiment --seeds 0 1 2."""

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pokerrl.cfr import CFR
from pokerrl.evaluate import Evaluator
from pokerrl.game import equilibrium_policy, random_policy
from pokerrl.neural import NeuralPG


@dataclass(frozen=True, slots=True)
class Row:
    """One exact evaluation of a training checkpoint."""

    algorithm: str
    seed: int
    iteration: int
    training_seconds: float
    wall_seconds: float
    value: float
    exploitability: float
    nash_conv: float
    current_value: float
    current_exploitability: float
    value_vs_random_p0: float
    value_vs_random_p1: float


def main() -> None:
    """Parse experiment budgets and write auditable numerical artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--cfr-iterations", type=int, default=5000)
    parser.add_argument("--neural-updates", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    arguments = parser.parse_args()
    if min(arguments.cfr_iterations, arguments.neural_updates, arguments.eval_every) < 1:
        parser.error("Training budgets and evaluation interval must be positive")
    if not np.isfinite(arguments.learning_rate) or arguments.learning_rate <= 0:
        parser.error("Learning rate must be positive and finite")
    output: Path = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator()
    equilibrium = evaluator.metrics(equilibrium_policy())
    if abs(equilibrium.value + 1 / 18) > 1e-12 or equilibrium.exploitability > 1e-12:
        raise RuntimeError("Exact Kuhn equilibrium sanity check failed")
    configuration = {
        "seeds": arguments.seeds,
        "cfr_iterations": arguments.cfr_iterations,
        "neural_updates": arguments.neural_updates,
        "eval_every": arguments.eval_every,
        "learning_rate": arguments.learning_rate,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "equilibrium": asdict(equilibrium),
        "budget_definitions": {
            "cfr": "One simultaneous regret update traverses six deals, 30 terminal leaves",
            "neural_pg": "One simultaneous Adam ascent update uses 30 exact terminal returns",
            "random": "Zero training; equal probability on each legal action",
        },
        "neural_architecture": "Two separate 7-input, 16-tanh, 2-softmax networks",
        "average": "Uniform-in-time behavioral policies weighted by each player's own reach",
        "evaluation": "Exact six-deal expectation; 64 consistent pure responses per player",
    }
    (output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    rows: list[Row] = []
    final_rows: list[Row] = []
    for seed in arguments.seeds:
        trainers = (
            ("random", None, 0),
            ("cfr", CFR(), arguments.cfr_iterations),
            ("neural_pg", NeuralPG(seed, arguments.learning_rate), arguments.neural_updates),
        )
        for algorithm, trainer, budget in trainers:
            start = time.perf_counter()
            training_seconds = 0.0
            policy = random_policy()
            current = random_policy()
            for iteration in range(budget + 1):
                if iteration > 0 and trainer is not None:
                    tick = time.perf_counter()
                    trainer.step()
                    training_seconds += time.perf_counter() - tick
                if iteration % arguments.eval_every and iteration != budget:
                    continue
                policy = random_policy() if trainer is None else trainer.average_policy()
                current = random_policy() if trainer is None else trainer.current_policy()
                metrics = evaluator.metrics(policy)
                current_metrics = evaluator.metrics(current)
                versus_random0 = random_policy()
                versus_random0[:6] = policy[:6]
                versus_random1 = random_policy()
                versus_random1[6:] = policy[6:]
                row = Row(
                    algorithm,
                    seed,
                    iteration,
                    training_seconds,
                    time.perf_counter() - start,
                    metrics.value,
                    metrics.exploitability,
                    metrics.nash_conv,
                    current_metrics.value,
                    current_metrics.exploitability,
                    float(evaluator.values(versus_random0)),
                    -float(evaluator.values(versus_random1)),
                )
                if not all(
                    np.isfinite(value) for value in asdict(row).values() if isinstance(value, float)
                ):
                    raise FloatingPointError(f"Nonfinite metrics for {algorithm}, seed {seed}")
                rows.append(row)
            final_rows.append(rows[-1])
            np.savez_compressed(
                output / f"{algorithm}_seed{seed}_policy.npz",
                average_policy=policy,
                current_policy=current,
            )
            if isinstance(trainer, NeuralPG):
                trainer.save(output / f"neural_pg_seed{seed}_checkpoint.npz")
            print(
                f"{algorithm} seed={seed} budget={budget} "
                f"exploitability={rows[-1].exploitability:.6f} "
                f"wall={rows[-1].wall_seconds:.2f}s",
                flush=True,
            )
    with (output / "learning_curves.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    aggregates: dict[str, dict[str, dict[str, float]]] = {}
    for algorithm in ("random", "cfr", "neural_pg"):
        selected = [row for row in final_rows if row.algorithm == algorithm]
        aggregates[algorithm] = {
            field: {
                "mean": float(np.mean([getattr(row, field) for row in selected])),
                "std": float(
                    np.std([getattr(row, field) for row in selected], ddof=int(len(selected) > 1))
                ),
            }
            for field in (
                "value",
                "exploitability",
                "current_exploitability",
                "value_vs_random_p0",
                "value_vs_random_p1",
                "training_seconds",
                "wall_seconds",
            )
        }
    summary = {
        "final": [asdict(row) for row in final_rows],
        "aggregate": aggregates,
        "std_definition": "Sample standard deviation across seeds; zero for one seed",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
