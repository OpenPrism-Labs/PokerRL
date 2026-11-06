# Additional RL methods for six-player Hold'em

Research checked November 6, 2026. These are established general-purpose RL
baselines adapted to this environment, not reproductions of poker SOTA.

## Saved progress

Work was stopped at the user's request before the full benchmark.
PPO, A2C, Double DQN, the common rollout collector, comparison runner, and Modal
entrypoint are implemented. All 62 tests, Ruff lint/format, and strict
Basedpyright pass; see `results/algorithms/checks.log`.

The Modal CUDA image built successfully, but the pilot was stopped before
CPU/GPU timing results were returned. `results/algorithms/modal-pilot.log` is
partial setup evidence, not a completed GPU training result. No full multi-seed
algorithm benchmark was run, so there is no measured ranking of these methods.
The commands below describe how to run the unfinished comparison.

## Why these methods

| Method | Primary and maintained references | What this comparison tests |
|---|---|---|
| PPO | Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347; SB3-Contrib MaskablePPO | Critic-based advantages, multiple rollout epochs and clipped policy changes |
| A2C | Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning*, arXiv:1602.01783; Stable-Baselines3 A2C | Simpler synchronous on-policy actor-critic; A2C is the synchronous variant, not the original asynchronous A3C implementation |
| Double DQN | van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning*, arXiv:1509.06461; Tianshou Double DQN | Replay-based value learning with online action selection and separate target evaluation |
| REINFORCE | Existing committed NumPy implementation | Original terminal-return policy-gradient control |

Stable-Baselines3's maintained RL tips recommend PPO/A2C for parallel discrete
environments and DQN extensions for discrete sample reuse. Its standard DQN
implementation does **not** include Double DQN. The implementation here must
therefore be checked against the Double DQN target, rather than merely running
SB3 DQN and relabeling it.

Maintained references: Stable-Baselines3 `guide/rl_tips.html` and `modules/a2c.html`;
SB3-Contrib `modules/ppo_mask.html`; Tianshou's Double DQN implementation.
Original GAE reference: Schulman et al., arXiv:1506.02438.

## Adaptation to this game

All methods use the same private-observation encoder: 209 inputs, 32 hidden
units, and six legal-action slots. PPO/A2C also learn a critic. Illegal actions
are masked during collection, optimization and evaluation, including DQN's
next-action selection. A separately trained PPO with history inputs zero
isolates the contribution of cross-hand opponent statistics.

Transitions run between decisions of the same hero seat. Opponents may act
between them. Net terminal profit is credited once, to the final hero decision,
including a hero who folded before the hand finished. Intermediate rewards
are zero, gamma is one, and true terminal transitions never bootstrap.
New methods scale rewards to net BB divided by 100. The preserved REINFORCE
control uses unscaled BB, a scalar running baseline, and its original optimizer
settings; this is an explicit implementation difference.

PPO and A2C gather complete hands and use GAE (lambda 0.95). PPO keeps old log
probabilities fixed, clips ratios at 0.2, and uses four optimization epochs.
A2C uses one update per collected batch. Double DQN selects the best **legal**
next action using online Q values, evaluates it using the target network, and
uses replay with Huber loss and periodic hard target copies.

Training uses a common scripted population and the same frozen snapshots from
the previous REINFORCE run. Opponents do not learn during replay collection.
Held-out evaluation has separate RNG streams and fixed or unannounced shifting
opponents. Method budgets match training hands, not gradient steps or runtime.
The benchmark records both. No held-out result is used to tune hyperparameters.

## Poker-specific alternatives

- **NFSP**, Heinrich and Silver, arXiv:1603.01121, adds supervised average-policy
  learning, reservoir memory and anticipatory mixing to a best-response learner.
  Double DQN alone is not NFSP. The original paper found ordinary RL comparators
  could diverge in Leduc.
- **Deep CFR**, Brown et al., ICML 2019, arXiv:1811.00164, needs counterfactual
  traversals, regret learning and average-strategy machinery. It is not a loss
  replacement over ordinary played-hand trajectories. Its equilibrium theory
  concerns two-player zero-sum games.
- **ReBeL**, Brown et al., arXiv:2007.13544, adds public-belief representations
  and search; its main Hold'em result is heads-up. It is a separate architecture.
- **Pluribus**, Brown and Sandholm, Science 2019, DOI:10.1126/science.aay2400,
  is an established six-player reference combining specialized self-play and
  search. This bounded review does not establish a universal latest champion.

Our feed-forward observation abstraction is not guaranteed to be Markov.
Sampled performance against scripts is neither full-game exploitability nor
proof of equilibrium, robustness against humans, or competitive poker strength.

## Reproduction

```sh
uv sync --extra dev
uv run python -m unittest discover -s tests -v
uv run python -m pokerrl.algorithm_experiment --seeds 0 1 2 --output results/algorithms
```

Defaults: 3,000 training hands per seed/method, six 120-hand evaluation sessions
per scenario and seed, shift at hand 60. There are three new methods, the original
REINFORCE control, a PPO no-history ablation, and a tight scripted evaluation
baseline. Checkpoints are reloaded before evaluation.

Local uv installs CPU PyTorch wheels. Modal installs CUDA-enabled PyTorch in
an ephemeral L4 container:

```sh
modal run modal_algorithms.py --pilot
# Optional full GPU benchmark, after comparing pilot throughput:
modal run modal_algorithms.py --train-hands 3000
```

The pilot measures real poker simulation plus learning for PPO and DDQN on CPU
and CUDA in the same container. Tiny networks may be limited by Python
simulation and device transfers; a GPU is not assumed faster. Jobs have bounded
timeouts and stop with their ephemeral apps. Results are retrieved locally.
