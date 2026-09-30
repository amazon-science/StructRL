"""Extract per-subtask completion timing from LIBERO-Long human demos.

LIBERO counterpart of structrl/extract_demo_timing.py (robocasa365). For each
of the 50 demos per libero_10 task we replay the saved flattened sim state at
every frame, evaluate each BDDL goal conjunct via the leaf env's own
`_eval_predicate` (same code path as `_check_success`), and record the first
frame at which each conjunct flips False -> True.

From those per-demo timelines we derive, under D3 semantics (timer resets at
EVERY subtask completion), the per-stage "rank budgets":

    rank_budgets[stage_idx][rank] = median over demos of the GAP (in sim
    frames) between the (rank)-th and (rank-1)-th completion within that stage
    (rank 0 = gap from the previous stage's last completion / episode start).

These budgets are the human-demo timing used by the smooth_band dense reward
(LiberoSmoothBandReward). NO scaling to robot horizon — the budget is the raw
human-demo median sim-frame gap (verified against robocasa365: Sum(budget) ~=
human horizon, NOT robot max_episode_steps; the robot's slowness is absorbed by
a generous max_episode_steps + the smooth 0.6/(1+ratio) curve, not by scaling).

Pre-satisfied conjuncts (true at frame 0, e.g. KITCHEN8 stove already on) are
dropped per-demo BEFORE the D3 walk, exactly mirroring
LiberoSmoothBandReward.reset(already_true=...), so the offline budget table is
indexed identically to the live reward's post-drop stage structure.

Stage definitions are imported from the single source of truth
rlinf.envs.libero.dense_reward.TASK_STAGES (get_stage_def / goal_signature),
so the extractor and the live reward can never drift.

Usage (with the LIBERO RLinf checkout and the LIBERO training venv):
    PYTHONPATH=${LIBERO_RL_ROOT}/RLinf:\
${LIBERO_RL_ROOT}/LIBERO \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    .venv-rlinf-libero/bin/python extract_libero_demo_timing.py \
        --output_json ${LIBERO_RL_ROOT}/libero_long_demo_timing.json
"""
import argparse
import glob
import json
import os
from collections import OrderedDict

import h5py
import numpy as np

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from rlinf.envs.libero.dense_reward import get_stage_def, goal_signature


_DATASET_ROOT = os.path.join(get_libero_path("datasets"), "libero_10")
_BDDL_ROOT = os.path.join(get_libero_path("bddl_files"), "libero_10")


def _resolve_bddl(h5_path: str) -> str:
    """Map a demo hdf5 file to its bddl in THIS libero checkout (the path
    stored in the hdf5 attrs points at the original collection machine)."""
    base = os.path.basename(h5_path).replace("_demo.hdf5", ".bddl")
    full = os.path.join(_BDDL_ROOT, base)
    if not os.path.exists(full):
        raise FileNotFoundError(f"bddl not found for {h5_path}: {full}")
    return full


def _conjuncts(goal_state) -> list[tuple]:
    return [tuple(str(x) for x in c) for c in goal_state]


def _replay_demo_timeline(inner, sim_states) -> tuple[dict, int, int | None, set]:
    """Replay one demo, return (first_true, T, success_frame, frame0_true).

    first_true[conjunct_tuple] = first frame index it held.
    frame0_true = conjuncts already True at frame 0 (pre-satisfied; dropped to
    mirror the live reward's reset(already_true)).
    """
    goal_state = inner.parsed_problem["goal_state"]
    conjs = _conjuncts(goal_state)
    T = len(sim_states)

    inner.sim.set_state_from_flattened(sim_states[0])
    inner.sim.forward()

    def _eval_all():
        out = set()
        for c in conjs:
            try:
                if bool(inner._eval_predicate(list(c))):
                    out.add(c)
            except Exception:
                pass
        return out

    frame0_true = _eval_all()
    first_true: dict[tuple, int] = {c: 0 for c in frame0_true}
    success_frame = None

    for t in range(T):
        try:
            inner.sim.set_state_from_flattened(sim_states[t])
            inner.sim.forward()
        except Exception:
            continue
        true_now = _eval_all()
        for c in true_now:
            if c not in first_true:
                first_true[c] = t
        try:
            if success_frame is None and bool(inner._check_success()):
                success_frame = t
        except Exception:
            pass

    return first_true, T, success_frame, frame0_true


def _aggregate_rank_budgets(goal_state, first_true_list, frame0_list) -> list[list[float]]:
    """D3 rank budgets on the POST-DROP stage structure.

    Drops pre-satisfied (frame-0-true) conjuncts per demo, rebuilds effective
    stages (empty stages removed, preserving order) exactly like
    LiberoSmoothBandReward.reset, then walks D3 to collect gaps. Returns a
    fixed list-of-lists median table indexed by post-drop stage / rank.
    """
    template = get_stage_def(goal_state)
    if template is None:
        return []

    # collected[stage_idx][rank] -> list of gaps across demos
    collected: dict[int, dict[int, list]] = {}

    for ft, f0 in zip(first_true_list, frame0_list):
        # Effective (post-drop) stages for THIS demo.
        eff = []
        for s in template:
            kept = [c for c in s if c not in f0]
            if kept:
                eff.append(kept)
        prev_completion = 0
        for stage_idx, stage_subs in enumerate(eff):
            times = sorted(
                ft[c] for c in stage_subs if c in ft and ft[c] >= prev_completion
            )
            for rank, t_complete in enumerate(times):
                gap = t_complete - prev_completion
                collected.setdefault(stage_idx, {}).setdefault(rank, []).append(gap)
                prev_completion = t_complete
            if len(times) < len(stage_subs):
                break  # stage not fully completed in this demo

    if not collected:
        return []
    n_stages = max(collected.keys()) + 1
    table: list[list[float]] = []
    for stage_idx in range(n_stages):
        ranks = collected.get(stage_idx, {})
        if not ranks:
            table.append([])
            continue
        n_ranks = max(ranks.keys()) + 1
        row = []
        for rank in range(n_ranks):
            arr = np.array(ranks.get(rank, [1.0]))
            row.append(float(np.median(arr)))
        table.append(row)
    return table


def _subtask_stats(first_true_list, frame0_list):
    all_c = set()
    for d in first_true_list:
        all_c.update(d.keys())
    stats = OrderedDict()
    for c in sorted(all_c):
        steps = [d[c] for d in first_true_list if c in d]
        pre = sum(1 for f0 in frame0_list if c in f0)
        stats[" ".join(c)] = {
            "median": float(np.median(steps)),
            "mean": float(np.mean(steps)),
            "std": float(np.std(steps)),
            "min": int(np.min(steps)),
            "max": int(np.max(steps)),
            "coverage": len(steps) / len(first_true_list),
            "n_pre_satisfied": pre,
        }
    return stats


def _process_task(h5_path: str, max_episodes: int, verbose: bool):
    bddl = _resolve_bddl(h5_path)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.reset()
    inner = env.env
    goal_state = inner.parsed_problem["goal_state"]
    template = get_stage_def(goal_state)
    if template is None:
        print(f"  [{os.path.basename(h5_path)}] goal not in TASK_STAGES, skipping")
        env.close()
        return None

    h = h5py.File(h5_path, "r")
    demo_keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[1]))
    demo_keys = demo_keys[:max_episodes]

    first_true_list, T_list, success_list, frame0_list = [], [], [], []
    for dk in demo_keys:
        sim_states = h["data"][dk]["states"][:]
        try:
            ft, T, succ, f0 = _replay_demo_timeline(inner, sim_states)
        except Exception as e:
            print(f"    {dk} ERROR {type(e).__name__}: {e}")
            continue
        first_true_list.append(ft)
        T_list.append(T)
        success_list.append(succ)
        frame0_list.append(f0)
    h.close()
    env.close()

    if not first_true_list:
        return None

    rank_budgets = _aggregate_rank_budgets(goal_state, first_true_list, frame0_list)
    subtask_stats = _subtask_stats(first_true_list, frame0_list)
    succ_frames = [s for s in success_list if s is not None]

    print(f"\n===== {os.path.basename(h5_path)} ({len(first_true_list)} demos) =====")
    print(f"  goal_state: {[list(c) for c in _conjuncts(goal_state)]}")
    print(f"  template stages: {[[ ' '.join(c) for c in s] for s in template]}")
    print(f"  horizon median={np.median(T_list):.0f} range=[{min(T_list)},{max(T_list)}]")
    if succ_frames:
        print(f"  success {len(succ_frames)}/{len(success_list)} median@{np.median(succ_frames):.0f}")
    for name, s in subtask_stats.items():
        print(f"    {name:<48s} med@{s['median']:>5.0f} cov={s['coverage']:>4.0%} pre={s['n_pre_satisfied']}")
    print(f"  rank_budgets (D3, post-drop, sim-frames): {rank_budgets}")
    print(f"  Sum(budget)={sum(b for row in rank_budgets for b in row):.0f}  "
          f"(compare to human horizon median {np.median(T_list):.0f})")

    return {
        "task": os.path.basename(h5_path).replace("_demo.hdf5", ""),
        "n_episodes": len(first_true_list),
        "goal_state": [list(c) for c in _conjuncts(goal_state)],
        "horizon": {"median": float(np.median(T_list)),
                    "min": int(min(T_list)), "max": int(max(T_list))},
        "success": {"n_success": len(succ_frames), "n_total": len(success_list),
                    "median": float(np.median(succ_frames)) if succ_frames else None},
        "subtask_timing": subtask_stats,
        "rank_budgets": rank_budgets,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_episodes", type=int, default=50)
    ap.add_argument("--output_json", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    h5_files = sorted(glob.glob(os.path.join(_DATASET_ROOT, "*_demo.hdf5")))
    if not h5_files:
        raise SystemExit(f"No demos found in {_DATASET_ROOT}")
    print(f"Found {len(h5_files)} libero_10 demo files in {_DATASET_ROOT}")

    results = {}
    for f in h5_files:
        r = _process_task(f, args.max_episodes, args.verbose)
        if r:
            results[r["task"]] = r

    if args.output_json:
        with open(args.output_json, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"\nWrote {args.output_json}  ({len(results)} tasks)")


if __name__ == "__main__":
    main()
