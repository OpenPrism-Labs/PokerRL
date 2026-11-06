# Stopped at user request

- Implemented: masked PPO, A2C, Double DQN, shared complete-hand rollout
  collection, comparison CLI, and ephemeral Modal L4 entrypoint.
- Verified: 62 tests, Ruff lint/format, strict Basedpyright.
- Evidence: `checks.log`.
- Modal image build completed; no CPU/GPU throughput measurements returned.
  `modal-pilot.log` is an interrupted run log.
- Full three-seed training and held-out method comparison: not run.
- No new algorithm performance or GPU speedup claim is supported yet.
- Owned Modal app `ap-IpfSqUS0XiKeGHWGokmckE` was explicitly stopped and
  confirmed stopped with no running tasks. The saved log has one trailing
  space removed for Git whitespace checks; substantive output is unchanged.

Resume command:

```sh
uv sync --extra dev
uv run python -m pokerrl.algorithm_experiment --seeds 0 1 2 --output results/algorithms
```
