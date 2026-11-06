# PokerRL: six-player Hold'em and exact Kuhn benchmarks

## Six-player no-limit Texas Hold'em

The full-game environment is `pokerrl.holdem.HoldemEnv`, backed by PokerKit
0.7.7. It supports a 52-card deck, two private cards per player, flop/turn/river,
rotating button and blinds, arbitrary legal integer raise-to amounts, short
all-ins and reopening rules, multiple side pots, split pots, and odd-chip
settlement. Observations expose only the current player's private cards.

The experiment uses six-player cash hands with 100 big blinds per seat,
1/2-chip blinds, zero ante/rake and a single runout. Stacks are rebought after
each hand; this is not an elimination tournament. PokerKit's default tournament
rules mode supplies deterministic settlement; no multi-runout cash-game options
are enabled. The environment accepts unequal starting stacks. Local seeded
decks make experiments reproducible without sharing RNG state with policies.

```sh
uv sync --extra dev
uv run python -m unittest discover -s tests -v
uv run python -m pokerrl.holdem_experiment --seeds 0 1 2 --output results/holdem
```

Defaults train each of two small networks for 1,000 hands per seed and evaluate
six independent 120-hand sessions for each of two scenarios. A short run:

```sh
uv run python -m pokerrl.holdem_experiment --seeds 0 --train-hands 40 \
  --eval-sessions 2 --eval-hands 12 --switch-hand 6 --output smoke-results/holdem
```

### Opponent-aware learning

The NumPy policy has 209 inputs, 32 tanh hidden units, and six masked actions:
fold, check/call, minimum raise, half-pot raise, pot raise, and all-in. Duplicate
or illegal choices are masked. **The neural action abstraction is finite; the
environment's no-limit raise API is not restricted to those sizes.**

Inputs contain the player's cards, board, public stacks/bets/actions, position,
and opponent statistics from the previous 32 completed hands: VPIP, preflop
raising, aggression, calls and folds when facing bets, and observation support.
Neither opponent labels nor hidden cards enter the policy. The comparison
network has the same architecture and training budget with history inputs zero.

Training uses terminal-return REINFORCE with a prior running reward baseline
and Adam. Weights stay fixed within a hand. The training pool includes random,
tight, loose, aggressive, and frozen historical neural policies. Opponents are
not trained from human hand histories. Scripted opponents use simple hole-rank
heuristics and are deliberately weak, especially postflop.

Evaluation loads saved policies with frozen weights. Deck seeds, opponent
parameters and action RNGs are separate from training. The fixed scenario uses
held-out script parameters. In the shifting scenario all five opponents become
more aggressive after hand 60, without notifying the learner or resetting its
history. The hero changes seats every six hands while the button moves every
hand, so it experiences every relative position.

Results are sampled **big blinds per hand**, with bootstrap intervals over
independent session means, conditional on these fitted models. Paired differences
use the same session/deck seeds. The intervals do not capture uncertainty over
all possible trained models, and full six-player exploitability is not computed.
This small run is an implementation baseline, not a claim of strong poker play.

### Using the environment and saved model

```python
from pathlib import Path
import numpy as np
from pokerrl.holdem import HoldemEnv
from pokerrl.holdem_learning import NeuralPolicy, OpponentHistory

env = HoldemEnv(seed=42)
history = OpponentHistory()
policy = NeuralPolicy.load(Path("results/holdem/history_seed0_checkpoint.npz"))
rng = np.random.default_rng(42)
env.reset()
while not env.terminal:
    observation = env.observation()
    env.step(policy.act(observation, history, rng))
result = env.result()
history.update(result)
print(result.payoffs)  # Six net chip rewards, summing to zero.
```

The example uses one policy at every seat. `Action("raise_to", amount)` can be
passed directly to `env.step` for other legal integer sizing. `reset(deck=...)`
accepts a complete permutation of card IDs for deterministic scenarios.
`NeuralPolicy.save/load` stores inference weights; `Trainer.save/load` additionally
stores Adam, running baseline and RNG. Resuming a complete session also requires
caller-owned environment and history state.

`results/holdem/` holds config, per-hand evaluation CSV, summary with intervals,
checkpoints, historical snapshots, public hand traces and check/training logs.
The separate `pokerrl.opponent_experiment` preserves the smaller Kuhn adaptation
prototype; its full benchmark was superseded by this Hold'em implementation.

### Measured Hold'em run

Three seeds, 1,000 training hands per network, 6,000 total training hands and
12,960 evaluation hands completed on CPU. Each scenario/method contains 18
sessions of 120 hands. Returns below are BB/hand; higher is better.

| Policy | Fixed opponents | Shifting opponents |
|---|---:|---:|
| History-aware neural | 0.984 [-2.682, 4.783] | -0.788 [-3.997, 3.267] |
| No-history neural | -1.111 [-4.568, 2.125] | -1.326 [-5.944, 2.580] |
| Tight scripted baseline | 2.468 [-0.435, 5.318] | 3.960 [0.882, 7.188] |

Brackets show 95% session-bootstrap intervals. The paired history-minus-no-history
difference is 2.095 [-2.937, 7.564] against fixed opponents and
0.538 [-4.682, 6.407] across shifting sessions. **Neither difference establishes
an adaptation advantage.** The tight baseline has higher mean returns than both
networks in this run. Full results, including pre/post-switch breakdowns, are in
`results/holdem/summary.json`; do not interpret this small run as convergence.

Measured training throughput was 113-207 hands/second; each 1,000-hand network
run took 4.8-8.8 seconds excluding evaluation. The rules simulation is CPU work
and the network is tiny, so no Modal GPU was launched.

All 39 tests pass, including the original Kuhn tests. Ruff check/format and
strict Basedpyright pass (`results/holdem/checks-final.log`). A separate public-API
hand traversed all four betting streets and settled a five-card board with
payoffs `(-2, -2, 4, -2, -2, 4)`; see `results/holdem/manual-hand.log`.

## Original Kuhn benchmark

This CPU experiment compares uniform random play, tabular CFR, and a small neural
self-play policy-gradient learner in heads-up Kuhn poker. Its algorithms use only
NumPy. No GPU, downloaded data, poker service, or pretrained model
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
