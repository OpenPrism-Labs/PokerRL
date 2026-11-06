"""Matched-hand six-player comparisons of PPO, A2C, Double DQN and REINFORCE."""

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from pokerrl.actor_critic import ActorCritic
from pokerrl.double_dqn import DoubleDQN
from pokerrl.holdem import HoldemEnv
from pokerrl.holdem_experiment import EvaluationRow, estimate, evaluate, opponents
from pokerrl.holdem_learning import NeuralPolicy, OpponentHistory, Policy, ScriptedPolicy, Trainer
from pokerrl.rl_common import Agent, collect_hand

Algorithm = Literal["ppo", "a2c", "ddqn", "reinforce", "ppo_no_history"]
ALGORITHMS: tuple[Algorithm, ...] = ("ppo", "a2c", "ddqn", "reinforce", "ppo_no_history")


def make_agent(algorithm: Algorithm, seed: int, device: str) -> Agent:
    match algorithm:
        case "ppo" | "a2c":
            return ActorCritic(seed, algorithm, device=device)
        case "ppo_no_history":
            return ActorCritic(seed, "ppo", device=device, use_history=False)
        case "ddqn":
            return DoubleDQN(seed, device=device)
        case "reinforce":
            raise ValueError("REINFORCE uses the preserved NumPy trainer")


def load_policy(algorithm: Algorithm, path: Path, device: str) -> Policy:
    match algorithm:
        case "ppo" | "a2c" | "ppo_no_history":
            return ActorCritic.load(path, device=device)
        case "ddqn":
            return DoubleDQN.load(path, device=device)
        case "reinforce":
            return NeuralPolicy.load(path)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    algorithm: str
    seed: int
    hands: int
    decisions: int
    updates: int
    training_seconds: float
    hands_per_second: float
    mean_training_bb: float
    checkpoint: str
    loss_last: float | None
    device: str


def train_one(
    algorithm: Algorithm,
    seed: int,
    hands: int,
    device: str,
    output: Path,
) -> TrainingResult:
    """Train on the same table/deck schedule, without using evaluation outcomes."""
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    env = HoldemEnv(seed)
    history = OpponentHistory()
    pool_rng = np.random.default_rng(np.random.SeedSequence([seed, 4701]))
    action_rng = np.random.default_rng(np.random.SeedSequence([seed, 4702]))
    table = opponents(pool_rng, held_out=False)
    archive_root = Path(__file__).resolve().parents[1] / "results" / "holdem"
    archive: list[Policy] = [
        NeuralPolicy.load(archive_root / f"history_seed0_snapshot{step}.npz")
        for step in (250, 500, 750)
    ]
    baseline = Trainer(seed, use_history=True) if algorithm == "reinforce" else None
    agent = None if baseline is not None else make_agent(algorithm, seed, device)
    rewards: list[float] = []
    decisions = 0
    curve: list[dict[str, int | float]] = []
    start = time.perf_counter()
    for hand in range(hands):
        if hand % 32 == 0:
            history.clear()
            table = opponents(pool_rng, held_out=False)
            if pool_rng.random() < 0.25:
                table = tuple(archive[int(pool_rng.integers(len(archive)))] for _ in range(6))
        env.reset()
        hero = (hand // 6) % 6
        if baseline is not None:
            sample = baseline.train_hand(env, table, history, hero=hero)
            payoff = sample.payoff_bb
            decisions += sample.decisions
        else:
            assert agent is not None
            trajectory, payoff = collect_hand(
                env,
                agent,
                table,
                history,
                action_rng,
                hero=hero,
                use_history=algorithm != "ppo_no_history",
            )
            decisions += len(trajectory)
            agent.learn(trajectory)
        rewards.append(payoff)
        if (hand + 1) % 500 == 0 or hand + 1 == hands:
            curve.append(
                {
                    "hands": hand + 1,
                    "mean_training_bb": float(np.mean(rewards[-500:])),
                    "elapsed_seconds": time.perf_counter() - start,
                    "decisions": decisions,
                }
            )
            print(f"TRAIN {algorithm} seed={seed} hands={hand + 1}", flush=True)
    suffix = "npz" if baseline is not None else "pt"
    checkpoint = output / f"{algorithm}_seed{seed}.{suffix}"
    if baseline is not None:
        updates = baseline.steps
        loss_last = None
        baseline.policy.save(checkpoint)
    else:
        assert agent is not None
        agent.finish()
        updates = agent.updates
        loss_last = agent.losses[-1] if agent.losses else None
        agent.save(checkpoint)
    if device == "cuda" and baseline is None:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    with (output / f"{algorithm}_seed{seed}_training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(curve)
    return TrainingResult(
        algorithm,
        seed,
        hands,
        decisions,
        updates,
        elapsed,
        hands / elapsed,
        float(np.mean(rewards)),
        checkpoint.name,
        loss_last,
        "cpu-numpy" if baseline is not None else device,
    )


def summarize(
    rows: list[EvaluationRow],
    algorithms: tuple[Algorithm, ...],
) -> list[dict[str, str | float | int]]:
    """Resample session means and paired differences, not individual hands."""
    groups: dict[tuple[str, str, str, int, int], list[float]] = defaultdict(list)
    for row in rows:
        for phase in ("all", row.phase):
            groups[(row.algorithm, row.scenario, phase, row.seed, row.session)].append(
                row.payoff_bb
            )
    rng = np.random.default_rng(761)
    output: list[dict[str, str | float | int]] = []
    keys = sorted({(row.seed, row.session) for row in rows})
    for scenario in ("fixed", "shifting"):
        for phase in ("all", "before", "after"):
            names = [*algorithms, "tight"]
            comparisons = [(name, "reinforce") for name in algorithms if name != "reinforce"]
            if "ppo" in algorithms and "ppo_no_history" in algorithms:
                comparisons.append(("ppo", "ppo_no_history"))
            for name, reference in [(name, "") for name in names] + comparisons:
                if reference and reference not in algorithms:
                    continue
                values: list[float] = []
                for seed, session in keys:
                    value = float(np.mean(groups[(name, scenario, phase, seed, session)]))
                    if reference:
                        value -= float(np.mean(groups[(reference, scenario, phase, seed, session)]))
                    values.append(value)
                stats = estimate(values, rng)
                output.append(
                    {
                        "algorithm": name if not reference else f"{name}_minus_{reference}",
                        "scenario": scenario,
                        "phase": phase,
                        "bb_per_hand": stats.mean,
                        "ci95_low": stats.low,
                        "ci95_high": stats.high,
                        "sessions": stats.sessions,
                    }
                )
    return output


def run(
    output: Path,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    algorithms: tuple[Algorithm, ...] = ALGORITHMS,
    train_hands: int = 3000,
    eval_sessions: int = 6,
    eval_hands: int = 120,
    switch_hand: int = 60,
    device: str = "cpu",
) -> None:
    """Train, reload and evaluate each method with predeclared matched budgets."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    config = {
        "seeds": seeds,
        "algorithms": algorithms,
        "train_hands": train_hands,
        "eval_sessions": eval_sessions,
        "eval_hands": eval_hands,
        "switch_hand": switch_hand,
        "device": device,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pokerkit": version("pokerkit"),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "features": 209,
        "hidden": 32,
        "actions": 6,
        "gamma": 1.0,
        "reward_scale_new_methods": 0.01,
        "reward_units": "terminal net BB / 100",
        "baseline": "Original NumPy REINFORCE uses unscaled BB, scalar running baseline, lr 0.001",
        "new_method_lr": 0.0003,
        "common": "Same action abstraction and visible-history encoder",
        "training_pool": "75% scripts; 25% tables of common frozen previous REINFORCE snapshots",
        "archive": [
            "history_seed0_snapshot250.npz",
            "history_seed0_snapshot500.npz",
            "history_seed0_snapshot750.npz",
        ],
        "comparison": "Equal training hands, not equal gradient updates or wall time",
        "evaluation": "Separate RNGs; fixed parameters and unannounced midpoint aggression shift",
        "uncertainty": "Session bootstrap conditional on fitted models; paired differences",
        "limitations": [
            "No equilibrium or exact exploitability guarantee",
            "Feed-forward abstraction may not be Markov",
            "Frozen off-policy replay population; no live opponent learning",
            "No evaluation-driven hyperparameter tuning",
        ],
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows: list[EvaluationRow] = []
    timings: list[TrainingResult] = []
    for seed in seeds:
        for algorithm in algorithms:
            result = train_one(algorithm, seed, train_hands, device, output)
            timings.append(result)
            policy = load_policy(algorithm, output / result.checkpoint, device)
            evaluations, _ = evaluate(
                policy, algorithm, seed, eval_sessions, eval_hands, switch_hand
            )
            rows.extend(evaluations)
            print(f"EVAL {algorithm} seed={seed} rows={len(evaluations)}", flush=True)
        evaluations, _ = evaluate(
            ScriptedPolicy("tight", 0.3),
            "tight",
            seed,
            eval_sessions,
            eval_hands,
            switch_hand,
        )
        rows.extend(evaluations)
    with (output / "evaluation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])), lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    summary = {
        "aggregate": summarize(rows, algorithms),
        "training": [asdict(row) for row in timings],
        "rows": len(rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"COMPLETE algorithms={len(algorithms)} rows={len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-hands", type=int, default=3000)
    parser.add_argument("--eval-sessions", type=int, default=6)
    parser.add_argument("--eval-hands", type=int, default=120)
    parser.add_argument("--switch-hand", type=int, default=60)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/algorithms"))
    args = parser.parse_args()
    if min(args.train_hands, args.eval_sessions, args.eval_hands) < 1:
        parser.error("Budgets must be positive")
    if not 0 < args.switch_hand < args.eval_hands:
        parser.error("Switch must occur within the evaluation session")
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct and nonnegative")
    run(
        args.output,
        seeds=tuple(args.seeds),
        train_hands=args.train_hands,
        eval_sessions=args.eval_sessions,
        eval_hands=args.eval_hands,
        switch_hand=args.switch_hand,
        device=args.device,
    )


if __name__ == "__main__":
    main()
