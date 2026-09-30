"""Unit check: out-of-order subtask fires immediately under noprec, not under baseline."""
import os, sys
MODE = sys.argv[1]  # rc365 | libero

if MODE == "rc365":
    sys.path.insert(0, __import__("os").environ.get("PROJECT_ROOT", "."))
    from structrl.smooth_band_reward import SmoothBandReward
    rw = SmoothBandReward(stages=[["a", "b"], ["c"]], rank_budgets=[[10.0, 10.0], [10.0]])
    # step 1: only "c" (stage-1 subtask) true -> baseline: 0 reward; noprec: fires
    out1 = rw.step({"progress": {"a": False, "b": False, "c": True}}, False, sim_t=5)
    # step 2: a,b true too
    out2 = rw.step({"progress": {"a": True, "b": True, "c": True}}, False, sim_t=8)
    # step 3: success
    out3 = rw.step({"progress": {"a": True, "b": True, "c": True}}, True, sim_t=9)
    print(f"r1={out1['r_total']:.4f} events1={[e['subtask'] for e in out1['events']]}")
    print(f"r2={out2['r_total']:.4f} events2={[e['subtask'] for e in out2['events']]}")
    print(f"r3={out3['r_total']:.4f} outcome={out3['r_outcome']:.1f} stage_after={out3['stage_idx_after']} cum={out3['cumulative']:.4f}")
    for k in ("t","stage_idx_after","subtask_start_step","elapsed_in_stage","r_event","r_persistent","r_outcome","r_total","cumulative","cumulative_progress","cumulative_persistent","cumulative_outcome","events","stage_just_completed","latched"):
        assert k in out3, f"missing contract key {k}"
    print("CONTRACT-OK")
else:
    sys.path.insert(0, __import__("os").path.join(__import__("os").environ.get("LIBERO_RL_ROOT", "."), "RLinf"))
    import rlinf.envs.libero.dense_reward as dr
    rw = dr.LiberoSmoothBandReward(stages=[[("x",)], [("y",)]])
    rw.reset(already_true=set(), rank_budgets=[[10.0], [10.0]])
    out1 = rw.step({("y",)}, False, sim_t=5)   # stage-1 conjunct first
    out2 = rw.step({("x",), ("y",)}, False, sim_t=8)
    out3 = rw.step({("x",), ("y",)}, True, sim_t=9)
    print(f"r1={out1['r_total']:.4f} n1={out1['n_newly_latched']}")
    print(f"r2={out2['r_total']:.4f} n2={out2['n_newly_latched']}")
    print(f"r3={out3['r_total']:.4f} outcome={out3['r_outcome']:.1f} stage_after={out3['stage_idx_after']} cum={out3['cumulative']:.4f}")
    for k in ("r_total","r_event","r_outcome","n_newly_latched","stage_idx_after","n_total","n_latched","cumulative","cumulative_progress","cumulative_outcome"):
        assert k in out3, f"missing contract key {k}"
    print("CONTRACT-OK")

# ---- flat-variant checks (appended) ----
if MODE == "rc365_flat":
    sys.path.insert(0, __import__("os").environ.get("PROJECT_ROOT", "."))
    from structrl.flat_reward import FlatEventReward
    rw = FlatEventReward(stages=[["a", "b"], ["c"]], rank_budgets=[[10.0, 10.0], [10.0]])
    out1 = rw.step({"progress": {"a": False, "b": False, "c": True}}, False, sim_t=5)
    out2 = rw.step({"progress": {"a": True, "b": True, "c": True}}, True, sim_t=8)
    print(f"r1={out1['r_total']:.4f} events1={[e['subtask'] for e in out1['events']]}" if 'events' in out1 else f"r1={out1['r_total']:.4f} (gated)")
    print(f"r2={out2['r_total']:.4f} outcome={out2['r_outcome']:.1f} cum={out2['cumulative']:.4f}")
if MODE == "libero_flat":
    sys.path.insert(0, __import__("os").path.join(__import__("os").environ.get("LIBERO_RL_ROOT", "."), "RLinf"))
    import rlinf.envs.libero.dense_reward as dr
    rw = dr.LiberoDenseReward(stages=[[("x",)], [("y",)]])
    rw.reset(already_true=set())
    out1 = rw.step({("y",)}, False)
    out2 = rw.step({("x",), ("y",)}, True)
    print(f"r1={out1['r_total']:.4f} n1={out1['n_newly_latched']}")
    print(f"r2={out2['r_total']:.4f} outcome={out2['r_outcome']:.1f} stage_after={out2['stage_idx_after']} cum={out2['cumulative']:.4f}")
    for k in ("r_total","r_event","r_outcome","n_newly_latched","stage_idx_after","n_total","n_latched","cumulative","cumulative_progress","cumulative_outcome"):
        assert k in out2, k
    print("CONTRACT-OK")
