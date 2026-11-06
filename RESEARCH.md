# Poker RL research notes

Literature checked November 6, 2026. This is a bounded review for choosing
experiments, not an exhaustive ranking of commercial poker systems.

## What this project measures

Start with Kuhn poker so a learned policy can be measured against an exact
best response. A high win rate against random play does not establish robust
play. Report exploitability, expected returns against fixed opponents, seed
variation, and compute budgets separately. CFR iterations and sampled RL
hands are different amounts of work; equal counts are not equal budgets.

This project's small neural baseline is an experiment, not a reproduction of
Deep CFR, ReBeL, Student of Games, or a full no-limit Hold'em agent. A neural
method losing to tabular CFR on a tiny game is a useful negative result.

## Established methods and evidence

| Method | Primary source | What the paper establishes | Relevance |
|---|---|---|---|
| Neural Fictitious Self-Play (NFSP) | Heinrich and Silver, 2016, arXiv:1603.01121 | Combines deep RL best responses with supervised average-strategy learning; approaches equilibrium in Leduc, unlike the common RL comparators in that study. | Strong next neural baseline; averaging is part of the algorithm, not merely smoothing a plotted curve. |
| Deep CFR | Brown et al., ICML 2019, arXiv:1811.00164; PMLR 97 | Neural approximation of CFR avoids a handcrafted abstract game and performs strongly in large poker games. | Suitable progression once tabular evaluation is trusted. A supervised fit to an already solved policy is not Deep CFR. |
| Pluribus | Brown and Sandholm, Science 2019, DOI:10.1126/science.aay2400 | Defeats elite professionals in six-player no-limit Hold'em. | Important multiplayer milestone; its result is not an exact exploitability certificate. |
| ReBeL | Brown et al., NeurIPS 2020, arXiv:2007.13544 | Combines RL with search over public belief states; gives a two-player zero-sum convergence result and reports superhuman heads-up no-limit Hold'em performance. | Hidden-information search needs beliefs, not perfect-information search with private cards exposed. |
| Student of Games | Schmid et al., Science Advances 2023, arXiv:2112.03178 | Combines guided search, learning and game-theoretic reasoning; beats the strongest openly available heads-up no-limit agent in its evaluation. | A strong general search-and-learning reference. “Openly available” does not establish superiority over every private solver. |

## Recent work worth checking

These are author-reported paper results, not independently reproduced here.
Different game formats, opponents, abstractions and metrics prevent a single
credible “current champion” declaration.

- **Hierarchical Deep CFR**, arXiv:2305.17327, revised September 14, 2026:
  hierarchical strategies and skill transfer, with tabular theory and neural
  objectives. Relevant for deeper games; unnecessary complexity for Kuhn.
- **Real-Time Parallel CFR**, arXiv:2605.19928, May 19, 2026:
  a heterogeneous CPU/GPU pipeline for depth-limited solving. The authors
  report 3.3–3.4 times postflop speedup on their desktop DGX Spark setup.
  This is a systems speedup under the paper's conditions, not a universal
  improvement in strategic strength.
- **Correlated Chance Sampling for MCCFR**, arXiv:2607.27035, July 29, 2026:
  correlated chance streams and paired experiments on Kuhn/Leduc, with
  reported exploitability improvements. A particularly relevant future
  experiment after adding sampled CFR and equal-node-touch accounting.
- **SafeAdapt-Poker**, arXiv:2610.20473, October 14, 2026:
  budgeted adaptation under shifting opponents, evaluated in exact Leduc
  and abstract Hold'em settings. Its safety and exploitation claims should
  not be generalized to unrestricted full Hold'em.
- **SpinGPT**, arXiv:2509.22387, revised February 23, 2026:
  supervised and reinforcement learning for three-player Spin & Go.
  Format-specific results do not rank it against heads-up equilibrium solvers.
- **How Far Are LLMs from Professional Poker Players?**,
  arXiv:2602.00528, January 31, 2026: evaluates play and reasoning;
  reports remaining game-theoretic weaknesses despite tool use and training.
- **PokerSkill**, arXiv:2605.30094, May 28, 2026:
  expert-authored action-grounding skills improve LLM play in the authors'
  benchmarks. The reported best result against GTOWizard remains negative
  (-57 ± 21 mbb/hand); “expert-level” in the title is not evidence of beating
  the solver.

## Recommended progression

1. Establish exact rules and evaluation on Kuhn; run a small network and CFR.
2. Add Leduc with a public card and two betting rounds; retain information-set
   correctness and compare NFSP, external-sampling MCCFR, and Deep CFR.
3. Report equal wall-time and node-touch comparisons, multiple seeds, held-out
   opponents, and exploitability or clearly identified approximate bounds.
4. Move batched neural training to a GPU only when measured workload warrants
   it. Modal is a possible execution platform, not itself an algorithm.
5. Full no-limit Hold'em needs action abstraction or action generation,
   substantially larger training, public-belief reasoning and careful
   evaluation. Small-game success does not establish that those problems are
   solved.

No Modal job was required for the initial CPU-sized benchmark. No claim here
is a reproduction of published large-scale poker performance.

## Implemented Hold'em extension

The repository now also provides a PokerKit-backed six-player no-limit Hold'em
environment, with a 209-32-6 NumPy policy trained by sampled terminal-return
REINFORCE. Public opponent-history features and a separately trained no-history
baseline operationalize the opponent-modeling idea in a small experiment.
The environment supports arbitrary legal integer raises; learned policies use
six discrete choices. This is neither Deep CFR nor a reproduction of Pluribus.

The committed 1,000-hand-per-model run does not demonstrate a statistically
resolved advantage for history conditioning. Its tight scripted baseline has
higher mean returns. Evaluation uses sampled BB/hand against scripted opponents,
not exact exploitability or measured human strength. Full rules are implemented;
strong full-game strategic play remains unestablished.
