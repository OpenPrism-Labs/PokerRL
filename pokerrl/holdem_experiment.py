"""Train and evaluate small six-player no-limit Hold'em policies on CPU."""

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

from pokerrl.holdem import HandResult, HoldemEnv
from pokerrl.holdem_learning import (
    NeuralPolicy,
    OpponentHistory,
    Policy,
    ScriptedPolicy,
    Trainer,
)


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    algorithm: str
    seed: int
    session: int
    scenario: str
    hand: int
    hero: int
    button: int
    phase: str
    payoff_bb: float
    decisions: int


@dataclass(frozen=True, slots=True)
class Estimate:
    mean: float
    low: float
    high: float
    sessions: int


def estimate(values: list[float], rng: np.random.Generator) -> Estimate:
    """Bootstrap independent session means, conditional on the fitted models."""
    samples = np.asarray(values)
    bootstrap = rng.choice(samples, size=(2000, len(samples)), replace=True).mean(axis=1)
    return Estimate(
        float(samples.mean()),
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
        len(values),
    )


def opponents(
    rng: np.random.Generator, *, held_out: bool, shifted: bool = False
) -> tuple[Policy, ...]:
    """Public-observation scripts; evaluation parameters differ from training."""
    low, high = (0.15, 0.45) if held_out else (0.5, 0.8)
    if shifted:
        return tuple(ScriptedPolicy("aggressive", float(rng.uniform(0.8, 0.98))) for _ in range(6))
    pool: tuple[Policy, ...] = (
        ScriptedPolicy("tight", float(rng.uniform(low, high))),
        ScriptedPolicy("loose", float(rng.uniform(low, high))),
        ScriptedPolicy("aggressive", float(rng.uniform(low, high))),
        ScriptedPolicy("loose", float(rng.uniform(low, high))),
        ScriptedPolicy("tight", float(rng.uniform(low, high))),
        ScriptedPolicy("random", float(rng.uniform(low, high))),
    )
    order = rng.permutation(6)
    return tuple(pool[int(index)] for index in order)


def play_hand(
    env: HoldemEnv,
    hero_policy: Policy,
    table: tuple[Policy, ...],
    history: OpponentHistory,
    rng: np.random.Generator,
    hero: int,
) -> HandResult:
    """Complete a real hand with frozen policies and update only public history."""
    while not env.terminal:
        observation = env.observation()
        policy = hero_policy if observation.seat == hero else table[observation.seat]
        env.step(policy.act(observation, history, rng))
    result = env.result()
    if sum(result.payoffs) != 0:
        raise RuntimeError("Hold'em settlement violated chip conservation")
    history.update(result)
    return result


def evaluate(
    policy: Policy,
    algorithm: str,
    seed: int,
    sessions: int,
    hands: int,
    switch_hand: int,
) -> tuple[list[EvaluationRow], list[HandResult]]:
    """Use identical independently shuffled deals across comparison treatments."""
    rows: list[EvaluationRow] = []
    traces: list[HandResult] = []
    for scenario in ("fixed", "shifting"):
        for session in range(sessions):
            key = np.random.SeedSequence([seed, session, 90001])
            deck_seed, action_seed, opponent_seed = key.generate_state(3)
            env = HoldemEnv(int(deck_seed))
            rng = np.random.default_rng(int(action_seed))
            pool_rng = np.random.default_rng(int(opponent_seed))
            table = opponents(pool_rng, held_out=True)
            changed_table = opponents(pool_rng, held_out=True, shifted=True)
            history = OpponentHistory()
            for hand in range(hands):
                if scenario == "shifting" and hand == switch_hand:
                    table = changed_table
                env.reset()
                hero = (hand // 6) % 6
                result = play_hand(env, policy, table, history, rng, hero)
                if session == 0 and hand == 0:
                    traces.append(result)
                phase = "before" if hand < switch_hand else "after"
                rows.append(
                    EvaluationRow(
                        algorithm,
                        seed,
                        session,
                        scenario,
                        hand,
                        hero,
                        result.button,
                        phase,
                        result.payoffs[hero] / 2,
                        sum(action.seat == hero for action in result.actions),
                    )
                )
    return rows, traces


def summarize(rows: list[EvaluationRow]) -> list[dict[str, str | float | int]]:
    """Aggregate sessions, never pretend correlated adaptive hands are IID."""
    rng = np.random.default_rng(761)
    output: list[dict[str, str | float | int]] = []
    for scenario in ("fixed", "shifting"):
        for phase in ("all", "before", "after"):
            for algorithm in ("history", "zero_history", "tight", "history_minus_zero"):
                session_values: list[float] = []
                keys = sorted({(row.seed, row.session) for row in rows})
                for seed, session in keys:
                    selected = [
                        row
                        for row in rows
                        if row.scenario == scenario
                        and row.seed == seed
                        and row.session == session
                        and (phase == "all" or row.phase == phase)
                    ]
                    if algorithm == "history_minus_zero":
                        left = [row.payoff_bb for row in selected if row.algorithm == "history"]
                        right = [
                            row.payoff_bb for row in selected if row.algorithm == "zero_history"
                        ]
                        session_values.append(float(np.mean(left) - np.mean(right)))
                    else:
                        session_values.append(
                            float(
                                np.mean(
                                    [
                                        row.payoff_bb
                                        for row in selected
                                        if row.algorithm == algorithm
                                    ]
                                )
                            )
                        )
                result = estimate(session_values, rng)
                output.append(
                    {
                        "scenario": scenario,
                        "phase": phase,
                        "algorithm": algorithm,
                        "bb_per_hand": result.mean,
                        "ci95_low": result.low,
                        "ci95_high": result.high,
                        "sessions": result.sessions,
                    }
                )
    return output


def main() -> None:
    """Run reproducible training and held-out six-player cash-hand sessions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-hands", type=int, default=1000)
    parser.add_argument("--eval-sessions", type=int, default=6)
    parser.add_argument("--eval-hands", type=int, default=120)
    parser.add_argument("--switch-hand", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=Path("results/holdem"))
    args = parser.parse_args()
    if min(args.train_hands, args.eval_sessions, args.eval_hands) < 1:
        parser.error("Training hands, sessions and evaluation hands must be positive")
    if not 0 < args.switch_hand < args.eval_hands:
        parser.error("Switch hand must be strictly inside the evaluation session")
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct and nonnegative")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive")
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "seeds": args.seeds,
        "train_hands": args.train_hands,
        "eval_sessions": args.eval_sessions,
        "eval_hands": args.eval_hands,
        "switch_hand": args.switch_hand,
        "learning_rate": args.learning_rate,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pokerkit": version("pokerkit"),
        "players": 6,
        "stacks": [200] * 6,
        "blinds": [1, 2],
        "format": "No-limit cash hands, rebuy every hand, one runout, no rake",
        "action_space": "Env: arbitrary legal integer raise-to; learner: six slots",
        "architecture": "209 inputs, 32 tanh, six masked action probabilities",
        "training": "Terminal BB-return REINFORCE; prior running baseline; Adam",
        "opponents": "Scripted populations plus frozen learner snapshots on 25% of training tables",
        "training_schedule": "Reset table/history every 32; rotate hero every 6, button every hand",
        "evaluation": "Separate deck/action/population RNGs, frozen weights, held-out parameters",
        "shift": "All opponents turn aggressive mid-session; no history reset or notification",
        "ci": "2000 bootstrap draws of session means, conditional on fitted models",
        "limitations": [
            "Intervals do not capture population uncertainty over all possible trained models",
            "No exact exploitability, equilibrium, or competitive-strength guarantee",
            "Small training budget; scripted opponents are not representative human data",
            "Checkpoints restore policies; not complete environment/session RNG continuation",
        ],
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    all_rows: list[EvaluationRow] = []
    timings: list[dict[str, str | int | float]] = []
    for seed in args.seeds:
        for use_history in (True, False):
            algorithm = "history" if use_history else "zero_history"
            start = time.perf_counter()
            trainer = Trainer(seed, use_history=use_history, learning_rate=args.learning_rate)
            env = HoldemEnv(seed)
            history = OpponentHistory()
            pool_rng = np.random.default_rng(np.random.SeedSequence([seed, 4701]))
            table = opponents(pool_rng, held_out=False)
            archive: list[Policy] = []
            training_returns: list[float] = []
            training_decisions = 0
            for hand in range(args.train_hands):
                if hand % 32 == 0:
                    history.clear()
                    table = opponents(pool_rng, held_out=False)
                    if archive and pool_rng.random() < 0.25:
                        table = tuple(
                            archive[int(pool_rng.integers(len(archive)))] for _ in range(6)
                        )
                env.reset()
                sample = trainer.train_hand(env, table, history, hero=(hand // 6) % 6)
                training_returns.append(sample.payoff_bb)
                training_decisions += sample.decisions
                if (hand + 1) % 250 == 0:
                    path = output / f"{algorithm}_seed{seed}_snapshot{hand + 1}.npz"
                    trainer.policy.save(path)
                    archive.append(NeuralPolicy.load(path))
                if (hand + 1) % 250 == 0 or hand + 1 == args.train_hands:
                    print(f"{algorithm} seed={seed} trained={hand + 1}", flush=True)
            training_seconds = time.perf_counter() - start
            checkpoint = output / f"{algorithm}_seed{seed}_checkpoint.npz"
            trainer.policy.save(checkpoint)
            restored = NeuralPolicy.load(checkpoint)
            rows, traces = evaluate(
                restored,
                algorithm,
                seed,
                args.eval_sessions,
                args.eval_hands,
                args.switch_hand,
            )
            all_rows.extend(rows)
            (output / f"{algorithm}_seed{seed}_traces.json").write_text(
                json.dumps([asdict(trace) for trace in traces], indent=2) + "\n"
            )
            timings.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "training_seconds": training_seconds,
                    "training_hands_per_second": args.train_hands / training_seconds,
                    "hero_decisions": training_decisions,
                    "hero_decisions_per_second": training_decisions / training_seconds,
                    "training_bb_per_hand": float(np.mean(training_returns)),
                    "wall_seconds": time.perf_counter() - start,
                }
            )
            print(f"{algorithm} seed={seed} evaluated={len(rows)}", flush=True)
        rows, _ = evaluate(
            ScriptedPolicy("tight", 0.3),
            "tight",
            seed,
            args.eval_sessions,
            args.eval_hands,
            args.switch_hand,
        )
        all_rows.extend(rows)
    with (output / "evaluation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(all_rows[0])), lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in all_rows)
    summary = {"aggregate": summarize(all_rows), "timings": timings, "rows": len(all_rows)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"COMPLETE rows={len(all_rows)} output={output}", flush=True)


if __name__ == "__main__":
    main()
