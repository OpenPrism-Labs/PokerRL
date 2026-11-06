# PokerRL: a small, runnable imperfect-information experiment

This CPU experiment compares uniform random play, tabular CFR, and a small neural
self-play policy-gradient learner in heads-up Kuhn poker. NumPy is the only
runtime dependency. No GPU, downloaded data, poker service, or pretrained model
is used.

```sh
uv sync --extra dev
uv run python -m unittest discover -s tests
uv run python -m pokerrl.experiment --seeds 0 1 2 --output results
```

When NumPy is already installed, the direct commands also work:

```sh
python -m unittest discover -s tests
python -m pokerrl.experiment --seeds 0 1 2 --output results
```

For a quick smoke run:

```sh
python -m pokerrl.experiment --seeds 0 --cfr-iterations 20 \
  --neural-updates 20 --eval-every 10 --output smoke-results
```

The default budgets are 5,000 CFR iterations and 10,000 neural updates per seed,
with exact evaluation every 250 updates. Adjust `--cfr-iterations`,
`--neural-updates`, `--eval-every`, and `--learning-rate` (default 0.01). The
experiment is deliberately small enough for a CPU run; wall times are recorded
instead of assuming a particular machine's performance.

## Game and evaluation

Each player antes one chip and receives a distinct private card from J, Q, K
(represented by 0, 1, 2). Player zero acts first. Check/check shows down for a
net reward of one chip; a bet followed by a call shows down for two. Folding
loses one. Rewards are net profit and sum to zero. There is no raise.

An immutable simulator state holds both cards. Its `observation()` exposes only
the acting player's card, public history, seat, and legal actions. Networks use
one-hot private card and public history, with a separate network for each seat.
Histories use `k` for check, `b` for bet, `f` for fold, and `c` for call.
Policy rows follow `("", "kb", "k", "b")`, with J/Q/K within each history.
The first six rows belong to player zero. Action column zero is check or fold;
column one is bet or call.

Evaluation enumerates all six equally likely ordered deals and thirty terminal
paths. Best responses enumerate all 64 deterministic strategies per seat.
Each strategy fixes one action per private information set, so it cannot choose
different actions after seeing the opponent's hidden card.

`value` is the exact self-play payoff to player zero.
`nash_conv` is the sum of the two unilateral best-response gains.
`exploitability` is `nash_conv / 2`, in chips per hand. A supplied analytic
equilibrium is checked before every experiment: its value is -1/18 and its
exploitability is zero to floating-point tolerance. Payoff versus a random
opponent is also evaluated separately for each seat; this is not an equilibrium
metric.

## What is trained

**CFR:** vanilla simultaneous full-tree counterfactual regret minimization,
with regret matching, chance-weighted regret increments, and a uniformly
time-averaged policy weighted by the player's own realization reach.
One iteration enumerates all six deals. CFR is deterministic; repeating it
across seeds is intentionally a repeated reference result.

**Neural PG:** two independently initialized 7-input / 16-tanh / 2-softmax
networks, trained with simultaneous Adam ascent on each seat's expected return.
The game tree supplies exact returns and their derivatives. The implementation
backpropagates these reward gradients through softmax and tanh manually in
NumPy. Player one ascends the negative player-zero return. Both gradients use
the same pre-update profile. No CFR targets, equilibrium labels, supervised
distillation, or best-response oracle are used in training.

This is **full-tree, model-based neural self-play policy gradient**. It is not
NFSP, a sample-based model-free method, or a guaranteed equilibrium solver.
Exact enumeration removes sampling noise in this tiny game; it would not scale
to full poker. Neural approximation, shared features across information sets,
Adam, and simultaneous self-play can cause cycling or saturation. Report the
measured exploitability even if it is worse than CFR or random.

Both trainers report the own-reach-weighted average strategy and current
strategy. Averaging uses each pre-update policy. At iteration zero, the average
is the uniform fallback because no training strategy has been accumulated;
the neural current strategy already reflects seeded random initialization.
The averaged neural policy is a table representing a mixture over historical
networks; it is not generally representable by the final network weights.

Equal update counts do not imply equal work. A CFR iteration traverses 30
terminal leaves; a neural update computes exact return gradients over those
same 30 paths and performs two network updates. The CLI records explicit update
budgets, training-only CPU wall duration, and elapsed wall duration including
evaluation at each checkpoint. Per-algorithm final wall duration excludes final
artifact serialization; these are elapsed times, not process CPU-time counters.

## Artifacts

`results/config.json` records budgets, versions, architecture, seed list, and
the equilibrium check. `learning_curves.csv` stores exact metrics at iteration
zero, each evaluation interval, and the final update. `summary.json` contains
per-seed final results and means/sample standard deviations. Repeated seeds
should be avoided when interpreting statistical uncertainty.

Each `{algorithm}_seed{seed}_policy.npz` contains `average_policy` and
`current_policy`. Each `neural_pg_seed{seed}_checkpoint.npz` additionally stores
the network weights, Adam moments, update count, learning rate, and strategy
accumulator. Checkpoints can be loaded for inference or continued training:

```python
from pathlib import Path
from pokerrl.evaluate import Evaluator
from pokerrl.neural import NeuralPG

model = NeuralPG.load(Path("results/neural_pg_seed0_checkpoint.npz"))
print(Evaluator().metrics(model.average_policy()))
print(Evaluator().metrics(model.current_policy()))
model.step()  # Resume with restored Adam moments and averaging state.
```

The tests cover hidden observations, legal transitions, terminal payouts,
analytic equilibrium and random-policy evaluation, exact gradients, CFR
improvement, neural reward-driven parameter changes, and checkpoint continuation.
No claim of convergence follows from a smoke run.

## Measured run: November 6, 2026

The default command above completed on a local CPU with seeds 0, 1, and 2.
Lower exploitability is better; units are chips per hand. These are exact
evaluations, with variability from neural initialization rather than sampled
evaluation noise. Standard deviations below are sample deviations over three
seeds, not confidence intervals.

| Method | Updates per seed | Average-policy exploitability | Current-policy exploitability | Training seconds per seed |
|---|---:|---:|---:|---:|
| Random | 0 | 0.458333 | 0.458333 | 0 |
| Tabular CFR | 5,000 | 0.003302 | 0.166667 | 1.579 |
| Neural PG | 10,000 | 0.130239 +/- 0.035259 | 0.222222 +/- 0.048112 | 2.601 |

CFR's average strategy is much closer to equilibrium. Neural learning improves
over uniform random play, but remains exploitable after this budget. The final
network is weaker than its historical average; the two must not be conflated.
CFR is deterministic, so its repeated seeds are not independent stochastic
trials. Budgets are explicit and unequal; these results are not a tuned,
equal-compute comparison or a convergence guarantee.

The neural average earns 0.128824 chips/hand as player zero against random,
and 0.242723 as player one. CFR earns 0.126834 and 0.167515 respectively.
Thus better exploitation of a weak opponent can coexist with worse
exploitability.

Evidence is committed in `results/summary.json`, `results/learning_curves.csv`,
`results/config.json`, and `results/training.log`, with all three neural
checkpoints. The independent analytic equilibrium check returns
`-0.05555555555555558` and exploitability `4.16e-17`. All ten tests pass;
the verbose transcript is `results/tests.log`.

See `RESEARCH.md` for primary-source comparisons with NFSP, Deep CFR, ReBeL,
Student of Games, and recent 2026 work. This is a small Kuhn benchmark, not a
full Texas Hold'em environment or a reproduction of those systems. CPU runtime
was seconds, so no Modal GPU job was needed or launched.
