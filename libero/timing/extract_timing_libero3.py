"""Per-suite demo-timing extraction for LIBERO-Spatial/Object/Goal.

Wraps extract_libero_demo_timing.py (single source of truth for the D3
rank-budget aggregation) with three deltas:
  1. suite-parameterized demo/bddl roots (our downloaded hdf5s);
  2. conjunct evaluation = union(goal conjuncts, stage-template conjuncts)
     via dense_reward.eval_conjuncts_true — with LIBERO3_HOOK=1 this includes
     the new ("grasp", obj) predicate;
  3. tasks sharing a goal signature (all 10 libero_spatial tasks) are POOLED
     before the median, instead of last-task-wins.

Run (needs GPU node for EGL env construction):
  LIBERO3_HOOK=1 LIBERO_CONFIG_PATH=$HOME/.libero \
  PYTHONPATH=${PROJECT_ROOT}/libero/spatial_object_goal/pyhook:${LIBERO_RL_ROOT}:\
${LIBERO_RL_ROOT}/RLinf:${LIBERO_RL_ROOT}/LIBERO \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  <venv-rlinf-libero>/python extract_timing_libero3.py --suite libero_spatial \
      --output_json rl/timing/libero_spatial_demo_timing.json
"""
import argparse
import glob
import json
import os

import h5py

from libero.libero.envs import OffScreenRenderEnv

import extract_libero_demo_timing as base  # LIBERO-Long extractor (libero/timing on PYTHONPATH)
import rlinf.envs.libero.dense_reward as dr

DATA_ROOT = os.environ.get("LIBERO_DATA_ROOT", "data")  # <DATA_ROOT>/<suite>/*.hdf5 demonstrations
BDDL_ROOT = os.path.join(os.environ.get("LIBERO_RL_ROOT", "."), "LIBERO/libero/libero/bddl_files")


def replay_with_stage_conjuncts(inner, sim_states):
    """Like base._replay_demo_timeline but times ALL stage conjuncts (incl.
    auxiliary grasp/open) through dense_reward.eval_conjuncts_true (grasp-aware
    when the libero3 hook is installed)."""
    goal_state = inner.parsed_problem["goal_state"]
    template = dr.get_stage_def(goal_state) or []
    conjs = {tuple(str(x) for x in c) for c in goal_state}
    for s in template:
        conjs.update(s)
    conjs = sorted(conjs)
    T = len(sim_states)

    def eval_all():
        return dr.eval_conjuncts_true(inner, conjs)

    inner.sim.set_state_from_flattened(sim_states[0])
    inner.sim.forward()
    frame0_true = eval_all()
    first_true = {c: 0 for c in frame0_true}
    success_frame = None
    for t in range(T):
        try:
            inner.sim.set_state_from_flattened(sim_states[t])
            inner.sim.forward()
        except Exception:
            continue
        for c in eval_all():
            if c not in first_true:
                first_true[c] = t
        try:
            if success_frame is None and bool(inner._check_success()):
                success_frame = t
        except Exception:
            pass
    return first_true, T, success_frame, frame0_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True,
                    choices=["libero_spatial", "libero_object", "libero_goal"])
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--max_episodes", type=int, default=50)
    args = ap.parse_args()

    demo_root = os.path.join(DATA_ROOT, args.suite)
    bddl_root = os.path.join(BDDL_ROOT, args.suite)
    h5s = sorted(glob.glob(os.path.join(demo_root, "*_demo.hdf5")))
    assert h5s, f"no demos under {demo_root}"

    # pooled[signature] = (goal_state, [first_true...], [frame0...], [task names])
    pooled = {}
    for h5_path in h5s:
        task = os.path.basename(h5_path).replace("_demo.hdf5", "")
        bddl = os.path.join(bddl_root, task + ".bddl")
        env = OffScreenRenderEnv(bddl_file_name=bddl,
                                 camera_heights=128, camera_widths=128)
        env.reset()
        inner = base.unwrap_inner(env) if hasattr(base, "unwrap_inner") else env.env
        if not hasattr(inner, "parsed_problem"):
            inner = env.env
        goal_state = inner.parsed_problem["goal_state"]
        if dr.get_stage_def(goal_state) is None:
            print(f"[skip] {task}: goal not in TASK_STAGES (hook missing?)")
            env.close()
            continue
        sig = dr.goal_signature(goal_state)
        entry = pooled.setdefault(sig, (goal_state, [], [], []))
        with h5py.File(h5_path, "r") as h:
            keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[1]))
            for dk in keys[: args.max_episodes]:
                states = h["data"][dk]["states"][:]
                try:
                    ft, T, succ, f0 = replay_with_stage_conjuncts(inner, states)
                except Exception as e:
                    print(f"  {task}/{dk} ERROR {type(e).__name__}: {e}")
                    continue
                entry[1].append(ft)
                entry[2].append(f0)
        entry[3].append(task)
        env.close()
        print(f"[done] {task}: pooled demos so far for its signature = {len(entry[1])}")

    out = {}
    for sig, (goal_state, fts, f0s, tasks) in pooled.items():
        budgets = base._aggregate_rank_budgets(goal_state, fts, f0s)
        template = dr.get_stage_def(goal_state)
        for task in tasks:
            out[task] = {
                "goal_state": [list(c) for c in goal_state],
                "stages": [[" ".join(c) for c in s] for s in template],
                "n_demos_pooled": len(fts),
                "rank_budgets": budgets,
            }
        print(f"signature({tasks[0]}...) stages={ [[ ' '.join(c) for c in s] for s in template] } budgets={budgets}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {args.output_json} ({len(out)} tasks)")


if __name__ == "__main__":
    main()
