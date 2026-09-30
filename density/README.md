# Decomposition variants

This directory reproduces the decomposition studies of the paper: the density sweep (Sec. 4.5, Fig. 4,
App. C.1 and C.2), the decomposer comparison (App. B.3), and the controls of App. C.3 and C.4. All
variants are trained on the 16 RoboCasa365 Composite-Seen tasks with GR00T-N1.5 and the StructRL PPO
recipe of `launch/train_gr00t_structrl_ppo.sbatch`; only the decomposition, its predicates, or the
reference durations change.

```bash
VARIANT=n14 sbatch density/train_density.sbatch
```

The launcher calls `make_overlay.py`, which copies `structrl/` to `density/overlays/<variant>/structrl/`
with the variant's `stage_config.py` (and predicate checker) swapped in, and points
`STRUCTRL_PROJECT_ROOT` and `STRUCTRL_TIMING_JSON` at it. Training stops at iteration 100; convert and
evaluate the checkpoint as in `docs/REPRODUCE.md`.

## Variants

Overall SR (%) is the paper result; N̄ is the average number of subtask signals per task.

| Variant | Decomposition | Signals (N̄) | Overall SR | Paper |
|---|---|---|---|---|
| terminal reward only | `launch/train_gr00t_sparse_rl_ppo.sbatch` | 0 (0) | 40.2 | Fig. 4, Table 10 |
| `n1` | one signal per task | 16 (1.0) | 45.9 | Fig. 4, Table 10 |
| default | `structrl/stage_config.py` | 38 (2.375) | 49.1 | Fig. 4, Table 10 |
| `n5` | adds grasp milestones and timer steps | 80 (5.0) | 50.4 | Fig. 4, Table 10 |
| `n10` | subset of `n29`, keeps all `n5` milestones | 160 (10.0) | 46.6 | Fig. 4, Table 10 |
| `n14` | subset of `n29`, keeps all `n5` milestones | 224 (14.0) | 44.1 | Fig. 4, Table 10 |
| `n29` | each `n5` milestone split into motion checkpoints | 470 (29.4) | 43.6 | Fig. 4, Table 10 |
| `n52` | finer motion checkpoints | 840 (52.5) | 37.6 | Fig. 4, Table 10 |
| `qwen` | stages proposed by Qwen3.5-9B | 63 (3.9) | 45.7 | Table 8 |
| `n5_earlier` | `n5` with each predicate moved to an earlier motion checkpoint | 80 (5.0) | 47.4 ± 0.8 (vs 50.4 ± 1.0) | Table 11 |
| `n14_budget` | `n14` with the total intermediate reward capped | 224 (14.0) | 47.7 ± 0.4 (vs 44.1 ± 1.8) | Table 11 |
| `uniform_pacing` | default decomposition, `T_d = H / N` | 38 (2.375) | 43.1 ± 0.4 (vs 49.2 ± 1.1) | Table 12 |

The paper rounds the N̄ of `n29` and `n52` to 29 and 52. Results with ± are the mean and sample
standard deviation over three evaluation runs.

## Layout

```
density/
├── stage_configs/    stage_config.py of each variant (same format as structrl/stage_config.py)
├── checkers/         predicate checkers of the denser variants
│                     (subtask_checker_n5.py: n5, qwen; subtask_checker_n29.py: n10, n14, n29, n5_earlier,
│                      n14_budget; subtask_checker_n52.py: n52); n1 and uniform_pacing use structrl/
├── timing/           reference durations T_d of each variant
├── decomposer/       Qwen3.5-9B decomposition: script, task commands, raw model output
├── make_overlay.py   builds the structrl/ package of a variant
└── train_density.sbatch
```

## Notes

- **Denser variants.** For the sweep the verifiability and progress filters of the default
  decomposition are relaxed, so fine motion checkpoints are allowed; each still has an executable
  predicate. Checkpoint signals are named `<milestone>__s<k>` and computed by the `n29` and `n52`
  checkers. `n10` and `n14` are deterministic subsets of the `n29` signals that keep every `n5`
  milestone and the `n5` stage order.
- **Reward cap (`n14_budget`).** `INTERMEDIATE_BUDGET=B` (here 1.4) sets the per-subtask base reward
  to `B / (1.5 N)` for a task with `N` signals, so the total intermediate reward no longer grows with
  `N`. With the StructRL pacing curve, whose event reward is at most twice the base, the total is at
  most `(4/3) B` (1.87 for `B = 1.4`) and `(2/3) B` at demonstration pace, below `lambda_c = 2.0`.
  The cap lives in `structrl/timing_reward.py` and is inactive unless the variable is set.
- **Uniform pacing.** `timing/uniform_pacing.json` replaces every `T_d` of the default decomposition
  with `H / N`, where `H` is the task horizon and `N` the number of subtasks of the task.
- **Qwen decomposer.** `python density/decomposer/llm_decompose.py` queries Qwen3.5-9B with only the
  task command and the decomposition requirements and writes `decomposer/qwen3.5-9b_output.json`.
  `stage_configs/qwen.py` grounds that output with the rules of App. B.2; the model's stages for each
  task are recorded in a comment next to its entry.
- **Reference durations.** Except for `uniform_pacing.json`, the `timing/` files were computed from the
  same SFT demonstrations as the default `structrl/demo_timing_composite_seen.json`.
