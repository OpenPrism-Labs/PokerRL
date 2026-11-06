# POKERRL PACKAGE

## OVERVIEW
Kuhn rules, exact evaluation, two trainers, and the experiment CLI live in six modules.
Score: 8 (code ratio, package boundary, symbol density, exports); distinct numerical domain.

## WHERE TO LOOK
| Change | Location | Contract |
|--------|----------|----------|
| Deals, actions, payouts | `game.py:State` | Immutable full state; observations hide the other card |
| Policy indexing | `game.py:information_index` | History-major rows, then J/Q/K |
| Strategy averaging | `game.py:realization_weights` | Own reach only; excludes chance and opponent |
| Regret updates | `cfr.py:CFR.step` | Accumulate all deals before changing regrets |
| Return derivatives | `evaluate.py:Evaluator.gradient` | Derivative of player-zero return |
| Best responses | `evaluate.py:Evaluator.metrics` | One action per private information set |
| Network derivatives | `neural.py:Network.gradients` | Manual softmax/tanh backpropagation |
| Optimizer state | `neural.py:Network.ascend` | Per-seat Adam ascent and update counter |
| Continued training | `neural.py:NeuralPG.save/load` | Restore weights, moments, counters, accumulator |
| Run artifacts | `experiment.py:main` | Config, policies, checkpoints, CSV, summary |

## CONVENTIONS
- A policy has shape `(12, 2)`; histories are `("", "kb", "k", "b")`.
- Rows 0-5 belong to player zero; rows 6-11 belong to player one.
- Column zero means check/fold; column one means bet/call.
- `Evaluator.values` also accepts batches shaped `(..., 12, 2)`.
- Evaluation pads with sentinel information index 12 to vectorize 30 terminal paths.
- Best responses enumerate 64 deterministic strategies per seat.
- Neural inputs are six rows per seat with seven private/public features.
- Both trainers accumulate the pre-update policy with own-realization weights.
- Neural gradients for both seats use the same pre-update profile.
- Player one ascends the negative of the player-zero return gradient.
- With no accumulated strategy, `average_policy()` returns uniform play.
- Neural checkpoints carry trainable state; policy archives carry two policy tables.
- `__init__.py` has no re-exports; import symbols from their defining modules.

## ANTI-PATTERNS
- Never expose `State.cards` as policy input; use `observation()`.
- Never let best responses condition their action on the opponent's hidden card.
- Do not replace simultaneous updates with sequential seat updates.
- Do not treat the average policy table as the final neural network.
- Do not introduce CFR targets or equilibrium labels into the reward-driven learner.
