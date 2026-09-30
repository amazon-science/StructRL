"""Unit-style sanity tests for structrl.timing_reward.

Hand-craft a few synthetic trajectories and check the cumulative reward
matches our spec.
"""
from collections import OrderedDict
from structrl.timing_reward import TimingReward, OUTCOME_REWARD


def _run(traj_events: list[tuple[int, dict]], traj_success_step: int | None,
         tr: TimingReward, T: int) -> tuple[float, list[dict]]:
    """Run a trajectory through tr, return (cumulative, per-step debug list).

    traj_events: list of (step, {"progress": {...}, "quiescence": {...}})
                 Steps not in this list use the previous step's flags.
                 Subtask flags are LATCHING in the env we simulate (once
                 True, stays True).
    traj_success_step: int or None — step at which env._check_success() turns True.
    """
    # Build per-step flat flag maps. Once True, stays True (env latches).
    cur_progress: dict[str, bool] = {}
    cur_quiescence: dict[str, bool] = {}
    schedule: dict[int, dict] = {s: g for s, g in traj_events}

    debug = []
    for t in range(T):
        if t in schedule:
            for k, v in schedule[t].get("progress", {}).items():
                if v:
                    cur_progress[k] = True
            for k, v in schedule[t].get("quiescence", {}).items():
                if v:
                    cur_quiescence[k] = True
        success = (traj_success_step is not None and t >= traj_success_step)
        info = tr.step(
            {"progress": OrderedDict(cur_progress),
             "quiescence": OrderedDict(cur_quiescence)},
            success,
        )
        debug.append(info)
    return tr.cumulative, debug


# Helper for printing
def _print_summary(name: str, total: float, debug: list[dict]):
    n = len(debug)
    last = debug[-1]
    n_events = sum(len(d["events"]) for d in debug)
    print(f"\n[{name}] T={n} cum={total:.4f}  "
          f"prog={last['cumulative_progress']:.4f} "
          f"persist={last['cumulative_persistent']:.4f} "
          f"outcome={last['cumulative_outcome']:.4f}  "
          f"#events={n_events}")
    for d in debug:
        if d["events"] or d["r_persistent"] != 0 or d["r_outcome"] != 0:
            tag = "★" if d["r_outcome"] > 0 else ("E" if d["events"] else "p")
            print(f"  step {d['t']:>4} {tag}  r={d['r_total']:+.4f}  "
                  f"stage={d['stage_idx_after']}  events={d['events']}")


def test_prepare_coffee_perfect():
    """PrepareCoffee: stage 0 (mug), stage 1 (machine), retreat empty.
    Demo budgets: stage 0 [293], stage 1 [177].
    Trajectory: mug@200 (early), machine@370 (early); success@400.
    """
    tr = TimingReward(
        stages=[["mug_at_machine"], ["machine_turned_on"], []],
        rank_budgets=[[293.0], [177.0], []],
    )
    events = [
        (200, {"progress": {"mug_at_machine": True}}),
        (370, {"progress": {"machine_turned_on": True}}),
    ]
    total, dbg = _run(events, traj_success_step=400, tr=tr, T=420)
    _print_summary("PrepareCoffee perfect", total, dbg)
    # Check: 2 paced events + outcome (2.0) ≈ 0.35 + 0.31 + 2.0 ≈ 2.65 (no persistent penalty)
    assert total > 2.4, f"Expected > 2.4, got {total}"
    assert total < 2.9, f"Expected < 2.9, got {total}"
    assert tr.cumulative_persistent == 0.0
    print("  ✓ perfect trajectory total in expected range")


def test_prepare_coffee_slow():
    """Same task but slow: mug@500 (ratio 1.7, decayed), machine@900 (ratio
    400/177=2.26, decayed), success@920.
    """
    tr = TimingReward(
        stages=[["mug_at_machine"], ["machine_turned_on"], []],
        rank_budgets=[[293.0], [177.0], []],
    )
    events = [
        (500, {"progress": {"mug_at_machine": True}}),
        (900, {"progress": {"machine_turned_on": True}}),
    ]
    total, dbg = _run(events, traj_success_step=920, tr=tr, T=940)
    _print_summary("PrepareCoffee slow", total, dbg)
    # Should still get outcome (1.0). Events both decayed but >= 0.05*sub_base.
    # Persistent penalty accumulates while elapsed > 1.5 * budget without latch.
    # Stage 0: 1.5*293=440; from step 440 to 500, 60 steps × -0.0005 = -0.030
    # Stage 1: 1.5*177=265; from step 500+265=765 to 900, 135 steps × -0.0005 = -0.068
    # ≈ -0.10 persistent penalty, +1.0 outcome, +small event reward
    assert total > 0.8, f"Expected > 0.8, got {total}"
    assert tr.cumulative_persistent < 0
    print("  ✓ slow trajectory has persistent penalty + outcome")


def test_prepare_coffee_fail_stage1():
    """mug@200 OK, but never finish machine. T=2900."""
    tr = TimingReward(
        stages=[["mug_at_machine"], ["machine_turned_on"], []],
        rank_budgets=[[293.0], [177.0], []],
    )
    events = [
        (200, {"progress": {"mug_at_machine": True}}),
    ]
    total, dbg = _run(events, traj_success_step=None, tr=tr, T=2900)
    _print_summary("PrepareCoffee fail stage1", total, dbg)
    # Stage 0 event at 200: ratio=200/293=0.68 → 0.5 * 1.136 = 0.568
    # Stage 1 starts at step 200, never completes.
    # Persistent starts when elapsed > 1.5*177=265 → from step 200+265=465 to 2899
    # = 2434 steps × -0.0005 = -1.22
    # Total ≈ 0.57 - 1.22 = -0.65
    assert total < 0, f"Expected negative, got {total}"
    assert tr.cumulative_outcome == 0.0
    print("  ✓ fail-stage-1 trajectory has large negative cumulative")


def test_packlunch_parallel():
    """PackIdenticalLunches: stage 0 = {tupper0, tupper1} parallel.
    D3 budgets (relative gaps): rank 0 = 686, rank 1 = 808 (= 1494-686).
    Trajectory: tupper0@600 (rank 0 elapsed=600, ratio 600/686=0.87 bonus),
    tupper1@1300 (rank 1 elapsed=1300-600=700 from last reset, ratio 700/808=0.87 bonus).
    success@1320.
    """
    tr = TimingReward(
        stages=[["tupper0_complete", "tupper1_complete"], []],
        rank_budgets=[[686.0, 808.0], []],
    )
    events = [
        (600, {"progress": {"tupper0_complete": True}}),
        (1300, {"progress": {"tupper1_complete": True}}),
    ]
    total, dbg = _run(events, traj_success_step=1320, tr=tr, T=1340)
    _print_summary("PackLunch parallel", total, dbg)
    assert total > 1.4, f"Expected > 1.4 (perfect ≈ 2.0+), got {total}"
    print("  ✓ parallel set works, both ranks rewarded")


def test_silent_skip():
    """Try to fire stage-1 subtask before stage 0 completes.
    PrepareCoffee: machine_turned_on @ step 100 (before mug). Should be IGNORED.
    Then mug @ step 300. Then machine should fire on the same step at most.
    """
    tr = TimingReward(
        stages=[["mug_at_machine"], ["machine_turned_on"], []],
        rank_budgets=[[293.0], [177.0], []],
    )
    events = [
        # silent skip — stage 1 subtask True before stage 0 done
        (100, {"progress": {"machine_turned_on": True}}),
        (300, {"progress": {"mug_at_machine": True}}),
    ]
    total, dbg = _run(events, traj_success_step=305, tr=tr, T=320)
    _print_summary("Silent-skip then recover", total, dbg)
    # Step 100: machine flag True but stage_idx still 0 → IGNORE, latched=False
    # Step 300: mug fires (stage 0 done), stage advances to 1. Same step the
    # machine flag is already True (latched in env) so machine should latch
    # immediately.
    n_latched_total = sum(1 for v in tr.latched.values() if v)
    assert n_latched_total == 2, (
        f"Expected both latched after recovery, got {n_latched_total}: {tr.latched}"
    )
    print("  ✓ silent-skip ignored, recovery latches both")


if __name__ == "__main__":
    test_prepare_coffee_perfect()
    test_prepare_coffee_slow()
    test_prepare_coffee_fail_stage1()
    test_packlunch_parallel()
    test_silent_skip()
    print("\nAll tests passed!")
