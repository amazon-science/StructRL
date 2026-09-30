"""Extract per-subtask completion timing from RoboCasa demonstration episodes.

For each demo episode, replay the saved sim state at every step, run
`get_subtask_states(env)` from structrl.subtask_checker, and record the
first step at which each subtask flips from False -> True.

Output: per-task statistics (median / mean / std / min / max) of completion
step for each subtask. Used to design timing-based dense rewards.

Usage:
    python structrl/extract_demo_timing.py \
        --task PrepareCoffee --max_episodes 20

    python structrl/extract_demo_timing.py \
        --task_list PrepareCoffee,PackIdenticalLunches --max_episodes 20
"""
import argparse
import gzip
import json
from collections import OrderedDict
import os
from pathlib import Path

import numpy as np
import robocasa  # noqa: F401
import robocasa.environments  # noqa: F401
from robocasa.utils.robomimic.robomimic_env_wrapper import EnvRobocasa

from structrl.subtask_checker import get_subtask_states
from structrl.stage_config import get_stage_def


_DATASET_ROOT = Path(os.environ.get("RC365_DATASET_ROOT", "datasets/v1.0")) / "target/composite"


_ENV_CACHE = {}  # cache one inner robosuite env per task name


def _get_env(env_name: str):
    """Lazily build a raw robosuite Kitchen env (skip robomimic wrapper).

    We don't need observations or robomimic obs-modality bookkeeping —
    we only need sim state replay + subtask state queries. Skipping the
    wrapper avoids OBS_KEYS_TO_MODALITIES init issues.
    """
    if env_name not in _ENV_CACHE:
        import robosuite
        from robosuite.controllers import load_composite_controller_config
        ctrl = load_composite_controller_config(controller=None, robot="PandaOmron")
        env = robosuite.make(
            env_name=env_name,
            robots="PandaOmron",
            controller_configs=ctrl,
            has_renderer=False,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=False,
            obj_instance_split="pretrain",
            layout_and_style_ids=[[1, 1], [2, 2], [4, 4], [6, 9], [7, 10]],
        )
        _ENV_CACHE[env_name] = env
    return _ENV_CACHE[env_name]


def _replay_episode_to_subtask_timeline(env_name: str, extras_dir: Path):
    """Replay one demo episode, return {subtask_name: first_True_step}.

    Uses EnvRobocasa.reset_to(state) which takes the full demo state:
    model XML + ep_meta + flattened sim state. This is the official
    robomimic-style demo replay path — it actually reconstructs the
    scene exactly as it was during recording.
    """
    states = np.load(extras_dir / "states.npz", allow_pickle=True)
    sim_states = states["states"]  # (T, state_dim)
    T = len(sim_states)

    ep_meta = json.loads((extras_dir / "ep_meta.json").read_text())
    with gzip.open(extras_dir / "model.xml.gz", "rt") as f:
        model_xml = f.read()

    inner = _get_env(env_name)

    # Mirror robomimic_env_wrapper.reset_to() but skip the obs path.
    # 1) ep_meta gives layout/fixtures/object_cfgs
    if hasattr(inner, "set_ep_meta"):
        inner.set_ep_meta(ep_meta)
    elif hasattr(inner, "set_attrs_from_ep_meta"):
        inner.set_attrs_from_ep_meta(ep_meta)
    # 2) full reset is needed before reset_from_xml_string
    inner.reset()
    # 3) feed the recorded model XML — this rebuilds fixtures EXACTLY where
    #    they were during demo recording (which is the bit my old code
    #    was missing)
    xml = inner.edit_model_xml(model_xml)
    inner.reset_from_xml_string(xml)
    inner.sim.reset()
    # 4) load the very first sim state
    inner.sim.set_state_from_flattened(sim_states[0])
    inner.sim.forward()
    if hasattr(inner, "update_state"):
        inner.update_state()

    first_true: dict[str, int] = {}
    success_step = None

    for t in range(T):
        try:
            inner.sim.set_state_from_flattened(sim_states[t])
            inner.sim.forward()
        except Exception:
            continue

        # CRITICAL: latched fixture flags (coffee_machine._turned_on,
        # toaster.toaster_on, etc.) live as Python attributes outside the
        # sim state, and are updated only when env.step calls _post_action.
        # We are NOT stepping, so we manually invoke update_state() each
        # frame to sync those flags from sim contact.
        try:
            inner.update_state()
        except Exception:
            pass

        try:
            grouped = get_subtask_states(inner)
        except Exception:
            continue

        for group_name in ("progress", "quiescence"):
            for sub_name, value in grouped.get(group_name, {}).items():
                if bool(value) and sub_name not in first_true:
                    first_true[sub_name] = t

        try:
            if inner._check_success() and success_step is None:
                success_step = t
        except Exception:
            pass

    return first_true, T, success_step


def _aggregate_rank_budgets(task_name: str, first_true_list: list[dict],
                             success_list: list[int | None]) -> dict:
    """Per-subtask budget under D3 semantics: timer resets every subtask.

    For each demo, walk through stages in order. WITHIN a stage, sort the
    completion times of the subtasks. The "budget" for the k-th finisher
    in stage i is the GAP between its completion time and the previous
    subtask completion (or the stage's previous-stage-last-completion
    if k=0). I.e. budget = (current sub completed at) - (last sub completed at).

    Across stages: the start of stage i+1 is the last completion in stage i
    (the same boundary timestamp).

    Aggregated as median/mean/std across demos. The final "budget table"
    used by the RL reward computer is rank_budgets[stage_idx][rank].

    Returns: dict[stage_idx -> dict[rank_idx -> {median,mean,std,n,subtasks}]]
    """
    stages = get_stage_def(task_name)
    n_stages = len(stages)

    # collected[stage][rank] = list of (gap-to-previous-subtask) per demo
    collected = {i: {} for i in range(n_stages)}

    for ft, succ in zip(first_true_list, success_list):
        prev_completion = 0  # time of last subtask completion (or 0 at start)
        for stage_idx, stage_subs in enumerate(stages):
            # Filter subtasks in this stage that actually fired AFTER
            # the previous stage's end (silent-skip-aware).
            times = sorted(
                ft[s] for s in stage_subs
                if s in ft and ft[s] >= prev_completion
            )
            for rank, t_complete in enumerate(times):
                # Gap from PREVIOUS completion to this one
                gap = t_complete - prev_completion
                collected[stage_idx].setdefault(rank, []).append(gap)
                prev_completion = t_complete  # advance reference
            # If stage didn't fully complete, stop walking deeper stages
            if len(times) < len(stage_subs):
                break

    # Aggregate
    out = {}
    for stage_idx in range(n_stages):
        out[stage_idx] = {}
        for rank, elapseds in collected[stage_idx].items():
            arr = np.array(elapseds)
            out[stage_idx][rank] = {
                "median": float(np.median(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": int(np.min(arr)),
                "max": int(np.max(arr)),
                "n": len(arr),
                "subtasks_in_stage": stages[stage_idx],
            }
    return out


def _aggregate_finalize_budget(task_name: str, first_true_list: list[dict],
                                success_list: list[int | None]) -> dict:
    """Median gap (sim steps) from LAST progress subtask completion to success.

    This is the "finalize / retreat" segment: in human demos, once the last
    progress flag is True the demonstrator just retreats and success fires
    shortly after. The RL reward uses this as the budget after which a policy
    that has latched all progress flags but not yet succeeded starts bleeding
    the stuck penalty again (closes the "latch then idle" exploit).

    Only demos that actually succeeded AND fired every progress subtask are
    counted (otherwise there is no well-defined last-progress -> success gap).
    """
    stages = get_stage_def(task_name)  # progress stages only (retreat removed)
    all_progress_subs = [s for stage in stages for s in stage]

    gaps = []
    for ft, succ in zip(first_true_list, success_list):
        if succ is None:
            continue
        prog_times = [ft[s] for s in all_progress_subs if s in ft]
        if len(prog_times) < len(all_progress_subs):
            continue  # not all progress flags fired in this demo
        last_prog = max(prog_times)
        gap = succ - last_prog
        if gap >= 0:
            gaps.append(gap)

    if not gaps:
        return {"median": None, "mean": None, "std": None, "n": 0}
    arr = np.array(gaps)
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "n": len(arr),
    }


def _aggregate(first_true_list: list[dict], T_list: list[int],
               success_list: list[int | None]):
    """Aggregate per-episode timing into median/mean/std stats."""
    # All subtasks ever observed
    all_subtasks = set()
    for d in first_true_list:
        all_subtasks.update(d.keys())

    stats = OrderedDict()
    for sub in sorted(all_subtasks):
        steps = [d[sub] for d in first_true_list if sub in d]
        coverage = len(steps) / len(first_true_list)
        if not steps:
            continue
        stats[sub] = {
            "median": float(np.median(steps)),
            "mean": float(np.mean(steps)),
            "std": float(np.std(steps)),
            "min": int(np.min(steps)),
            "max": int(np.max(steps)),
            "coverage": coverage,  # fraction of episodes where this fired
            "n": len(steps),
        }

    success_steps = [s for s in success_list if s is not None]
    success_stats = {
        "median": float(np.median(success_steps)) if success_steps else None,
        "mean": float(np.mean(success_steps)) if success_steps else None,
        "std": float(np.std(success_steps)) if success_steps else None,
        "n_success": len(success_steps),
        "n_total": len(success_list),
    }
    horizon_stats = {
        "median": float(np.median(T_list)),
        "mean": float(np.mean(T_list)),
        "min": int(np.min(T_list)),
        "max": int(np.max(T_list)),
    }
    return stats, success_stats, horizon_stats


def _process_task(task_name: str, max_episodes: int, verbose: bool):
    task_dir = _DATASET_ROOT / task_name
    if not task_dir.exists():
        print(f"  [{task_name}] dataset not found at {task_dir}")
        return None
    # Find the lerobot extras dir
    cand = list(task_dir.glob("*/lerobot/extras"))
    if not cand:
        print(f"  [{task_name}] no lerobot/extras directory found under {task_dir}")
        return None
    extras_root = cand[0]
    eps = sorted([p for p in extras_root.glob("episode_*") if p.is_dir()])
    if not eps:
        print(f"  [{task_name}] no episodes")
        return None
    eps = eps[:max_episodes]

    print(f"\n========== {task_name} ({len(eps)} episodes) ==========")
    first_true_list = []
    T_list = []
    success_list = []
    for i, ep_dir in enumerate(eps):
        try:
            ft, T, succ = _replay_episode_to_subtask_timeline(task_name, ep_dir)
            first_true_list.append(ft)
            T_list.append(T)
            success_list.append(succ)
            if verbose:
                ft_str = ", ".join(f"{k}@{v}" for k, v in sorted(ft.items(), key=lambda x: x[1]))
                print(f"  ep {i:>2} (T={T}): success@{succ}  subtasks: {ft_str}")
        except Exception as e:
            print(f"  ep {i:>2} ERROR: {type(e).__name__}: {e}")
            continue

    if not first_true_list:
        return None

    stats, succ_stats, horiz_stats = _aggregate(
        first_true_list, T_list, success_list
    )
    rank_budgets = _aggregate_rank_budgets(task_name, first_true_list, success_list)
    finalize_budget = _aggregate_finalize_budget(task_name, first_true_list, success_list)

    print(f"\n  horizon: median={horiz_stats['median']:.0f}  range=[{horiz_stats['min']}, {horiz_stats['max']}]")
    print(f"  success: {succ_stats['n_success']}/{succ_stats['n_total']}  "
          f"median={succ_stats['median']}  std={succ_stats['std']:.1f}"
          if succ_stats['median'] else f"  success: 0/{succ_stats['n_total']} (no successes!)")
    print(f"\n  per-subtask first-True step (raw):")
    print(f"  {'subtask':<32s} {'median':>8s} {'mean':>8s} {'std':>8s} {'min':>6s} {'max':>6s} {'cov':>6s}")
    for sub, s in stats.items():
        print(f"  {sub:<32s} {s['median']:>8.0f} {s['mean']:>8.0f} {s['std']:>8.1f} "
              f"{s['min']:>6d} {s['max']:>6d} {s['coverage']:>6.0%}")

    print(f"\n  rank-based stage budgets (elapsed in stage, by completion rank):")
    print(f"  {'stage':>6} {'rank':>5} {'median':>8s} {'mean':>8s} {'std':>8s} {'n':>4} subtasks_in_stage")
    for stage_idx, ranks in rank_budgets.items():
        for rank, st in ranks.items():
            print(f"  {stage_idx:>6} {rank:>5} {st['median']:>8.0f} {st['mean']:>8.0f} "
                  f"{st['std']:>8.1f} {st['n']:>4} {st['subtasks_in_stage']}")

    if finalize_budget["median"] is not None:
        print(f"\n  finalize budget (last progress -> success): "
              f"median={finalize_budget['median']:.0f} mean={finalize_budget['mean']:.0f} "
              f"std={finalize_budget['std']:.1f} n={finalize_budget['n']}")
    else:
        print(f"\n  finalize budget: N/A (no demo fired all progress flags + succeeded)")

    return {
        "task": task_name,
        "n_episodes": len(first_true_list),
        "horizon": horiz_stats,
        "success": succ_stats,
        "subtask_timing": stats,
        "rank_budgets": rank_budgets,
        "finalize_budget": finalize_budget,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None)
    ap.add_argument("--task_list", default=None,
                    help="comma-separated list of tasks to process")
    ap.add_argument("--max_episodes", type=int, default=20)
    ap.add_argument("--output_json", default=None,
                    help="if set, write aggregated stats to JSON")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.task and args.task_list:
        ap.error("--task xor --task_list")
    if args.task:
        tasks = [args.task]
    elif args.task_list:
        tasks = [t.strip() for t in args.task_list.split(",") if t.strip()]
    else:
        tasks = ["PrepareCoffee"]

    all_results = {}
    for t in tasks:
        r = _process_task(t, args.max_episodes, args.verbose)
        if r:
            all_results[t] = r

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
