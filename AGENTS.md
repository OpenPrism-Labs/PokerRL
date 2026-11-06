# PROJECT KNOWLEDGE BASE

**Generated:** 2026-11-06
**Commit:** c66375e
**Branch:** master

## OVERVIEW
CPU-only heads-up Kuhn poker experiment comparing random play, tabular CFR,
and exact-return neural self-play policy gradient. Python 3.11+, NumPy, setuptools.

## STRUCTURE
```text
PokerRL/
|-- pokerrl/          # Rules, exact evaluator, trainers, experiment CLI
|-- tests/            # unittest cases for rules, gradients, training, resumption
|-- results/          # Tracked numerical evidence and saved neural checkpoints
|-- README.md         # Run instructions and measured-result interpretation
|-- RESEARCH.md       # Primary-source context and comparison limits
`-- pyproject.toml    # Packaging and development-tool configuration
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Change game mechanics | `pokerrl/game.py` | See package guidance for numerical contracts |
| Change exact metrics | `pokerrl/evaluate.py` | Return, gradients, information-consistent responses |
| Change tabular training | `pokerrl/cfr.py` | Full-tree simultaneous CFR |
| Change neural training | `pokerrl/neural.py` | Manual NumPy derivatives and optimizer |
| Change CLI or output schema | `pokerrl/experiment.py` | `python -m pokerrl.experiment` |
| Verify hidden information | `tests/test_game.py` | Privacy, legal transitions, payouts |
| Verify mathematical evaluation | `tests/test_evaluate.py` | Equilibrium, random policy, finite differences |
| Verify learning and checkpoints | `tests/test_training.py` | CFR, neural gradients, exact continuation |
| Inspect recorded run | `results/config.json`, `results/summary.json` | Budgets, versions, per-seed metrics |

## CODE MAP
AST import and definition searches establish this map; LSP symbol/reference
requests are unsupported here. Reference centrality is unmeasured.

| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| `State` | class | `pokerrl/game.py:40` | unmeasured | Immutable simulator |
| `information_index` | function | `pokerrl/game.py:86` | unmeasured | Shared policy-row mapping |
| `Evaluator` | class | `pokerrl/evaluate.py:24` | unmeasured | Exact values, gradients, best responses |
| `CFR` | class | `pokerrl/cfr.py:16` | unmeasured | Tabular reference trainer |
| `Network` | class | `pokerrl/neural.py:12` | unmeasured | Per-seat policy and Adam state |
| `NeuralPG` | class | `pokerrl/neural.py:68` | unmeasured | Neural training and checkpoint persistence |
| `main` | function | `pokerrl/experiment.py:38` | unmeasured | Experiment orchestration |

## CONVENTIONS
- Tests use `unittest`; pytest is also configured for the `tests/` directory.
- Ruff uses 100-character lines and rules E, F, I, B, UP.
- Basedpyright strict checking targets `pokerrl/`.
- Runtime needs only NumPy; no GPU, downloads, or external poker service.

## ANTI-PATTERNS (THIS PROJECT)
- Do not claim convergence from a smoke run.
- Do not equate update counts with equal computational work.
- Do not count repeated deterministic CFR seeds as independent stochastic trials.
- Do not describe this learner as NFSP or a full Texas Hold'em implementation.
- Report measured exploitability even when it is worse than a baseline.

## UNIQUE STYLES
- Exact enumeration removes evaluation sampling noise in this tiny game.
- Analytic equilibrium is checked before each experiment: value -1/18.
- Exploitability is half NashConv, measured in chips per hand.
- Payoff against random opponents is a separate metric from exploitability.

## COMMANDS
Run from the repository root:
```bash
uv sync --extra dev
uv run python -m unittest discover -s tests
uv run python -m pokerrl.experiment --seeds 0 1 2 --output results
# Direct smoke run when NumPy is installed:
python -m pokerrl.experiment --seeds 0 --cfr-iterations 20 \
  --neural-updates 20 --eval-every 10 --output smoke-results
```

## NOTES
- `results/` is tracked evidence; experiment commands write into their output directory.
- Use `smoke-results/` for disposable runs; it is ignored by git.
- Default budgets: 5,000 CFR iterations, 10,000 neural updates, evaluation every 250.
- Default learning rate is 0.01; CLI budgets must be positive and rate finite/positive.
- `training_seconds` times training calls; `wall_seconds` includes evaluation.
- Final per-algorithm duration excludes final artifact serialization.
- Neural averages are historical mixtures; preserve separate average/current metrics.
- Existing run transcripts are in `results/tests.log` and `results/training.log`.
