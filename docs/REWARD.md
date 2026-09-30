# The reward in code

## How a step is paid

- **Stages.** `structrl/stage_config.py` lists, for each task, the ordered stages of subtask names;
  subtasks inside a stage may complete in any order. `structrl/subtask_checker.py` computes the yes/no
  predicate behind each name from the simulator state.
- **Gating.** During training, only subtasks of the currently open stage are checked. A stage opens once
  all subtasks of the previous stages are complete, and each subtask is credited once.
- **Pacing.** A credited completion pays `beta / (1 + T / T_d)` with `beta = 0.6`. `T` is the time since
  the previous credited completion (since the episode start for the first one), and `T_d` is the
  reference duration of the subtask computed from the SFT demonstrations
  (`structrl/extract_demo_timing.py`).
- **Success.** The first success of the whole task adds `lambda_c = 2.0`, paid at the first successful
  chunk.

Rewards are summed per action chunk inside the environment workers.

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `STRUCTRL_REWARD_VARIANT` | `structrl` (the paper's reward), `fixed` (0.3 per subtask), `smooth_band_cap10` (beta = 1.0) | `timing` (earlier variant, not used in the paper) |
| `STRUCTRL_TIMING_JSON` | reference durations `T_d` | `structrl/demo_timing_composite_seen.json` |
| `STRUCTRL_PROJECT_ROOT` | directory containing the `structrl/` package to use | this repository |
| `OUTCOME_REWARD_OVERRIDE` | terminal reward `lambda_c` | `2.0` |
| `INTERMEDIATE_BUDGET` | reward cap of App. C.3 (see `density/README.md`) | unset |
| `STRUCTRL_BAND_CAP` | beta override (needs `ablations/beta_sweep` on `PYTHONPATH`) | unset |
| `STRUCTRL_NO_PRECEDENCE`, `LIBERO_NO_PRECEDENCE` | disable gating (needs `ablations/ungated` on `PYTHONPATH`) | unset |
| `LIBERO_DENSE_REWARD`, `LIBERO_REWARD_VARIANT`, `LIBERO_TIMING_JSON` | LIBERO reward switch, variant (`structrl` / `fixed`), `T_d` | off / `fixed` / none |

The launchers set these for each experiment; see [REPRODUCE.md](REPRODUCE.md).

## LIBERO

The LIBERO-Long reward (stages, predicates, pacing) is `rlinf/envs/libero/dense_reward.py`, added by
`patches/libero_addon.patch`; the LIBERO Spatial / Object / Goal stages are in
`libero/spatial_object_goal/pyhook/libero3_stages.py`.

## Decomposition variants

`density/` holds the decompositions of the density study (Sec. 4.5, Fig. 4, App. C), the Qwen3.5-9B
decomposer (App. B.3) and the controls of App. C.3 and C.4. One launcher trains any of them:

```bash
VARIANT=n5 sbatch density/train_density.sbatch    # n1 n5 n10 n14 n29 n52 qwen n5_earlier n14_budget uniform_pacing
```

See [density/README.md](../density/README.md) for what each variant is and the paper result it reproduces.
