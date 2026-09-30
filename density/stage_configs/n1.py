"""Density N̄ = 1 (Sec. 4.5, Fig. 4): one intermediate signal per task, 16 in total.

For each task the signal is the default-decomposition subtask closest to the middle of the
demonstrations. Same format as structrl/stage_config.py; uses the default checker.
"""
from collections import OrderedDict


# Per-task stage definition. Order of the list IS the stage execution order.
# Each stage is a list of subtask names (from subtask_checker output).
# 'progress' subtasks come from grouped_states['progress'][...] keys;
# 'quiescence' from grouped_states['quiescence'][...] keys.
TASK_STAGES = OrderedDict([
    # Single-stage progress + retreat
    ("DeliverStraw", {
        "progress_stages": [["straw_in_glass_cup"]],
        "retreat_stage": ["gripper_far"],
    }),
    # Sequential 2 progress + retreat
    ("GetToastedBread", {
        "progress_stages": [["toaster_on"]],
        "retreat_stage": ["gripper_far"],
    }),
    # Sequential 2 progress + retreat (kept as 2 stages even though
    # physically related; could be merged)
    ("KettleBoiling", {
        "progress_stages": [["kettle_on_stove"]],
        "retreat_stage": ["gripper_far"],
    }),
    ("LoadDishwasher", {
        "progress_stages": [["dish1_on_rack"]],
        "retreat_stage": [],
    }),
    # Parallel 2 progress + retreat
    ("PackIdenticalLunches", {
        "progress_stages": [["tupper1_complete"]],
        "retreat_stage": ["gripper_far"],
    }),
    # Parallel 3 progress + retreat
    ("PreSoakPan", {
        "progress_stages": [["sponge_in_sink"]],
        "retreat_stage": ["gripper_far_pan", "gripper_far_sponge"],
    }),
    # Sequential 2 progress + retreat
    ("PrepareCoffee", {
        "progress_stages": [["mug_at_machine"]],
        "retreat_stage": ["gripper_obj_far", "gripper_button_far"],
    }),
    # Parallel 3 progress, no retreat
    ("RinseSinkBasin", {
        "progress_stages": [["washed_center"]],
        "retreat_stage": [],
    }),
    # Parallel 2 progress + retreat
    # (2 metrics of the same continuous "scrub" action — merge to 1 stage)
    ("ScrubCuttingBoard", {
        "progress_stages": [["sweep_range_0p1m"]],
        "retreat_stage": ["gripper_sponge_far"],
    }),
    # Demo order: meat in pan first (median 642), then pan moved onto
    # active knob (median 890). Reflect actual demo sequence.
    ("SearingMeat", {
        "progress_stages": [["meat_in_pan"]],
        "retreat_stage": ["gripper_meat_far"],
    }),
    ("SetUpCuttingStation", {
        "progress_stages": [["knife_on_board"]],
        "retreat_stage": ["gripper_far"],
    }),
    # 2 stages by demo behavior: stack bowls on counter, then put the
    # stacked pair into the cabinet. Both well-separated in time.
    ("StackBowlsCabinet", {
        "progress_stages": [["bowls_stacked"]],
        "retreat_stage": ["gripper_far"],
    }),
    # Real success = veg_in_bowl AND bowl_in_micro AND door_closed AND microwave_on.
    # Note: microwave_on is a Python-attribute latch only updated when env.step
    # calls fixture.update_state with live gripper contact data; demo replay
    # (set_state_from_flattened only) cannot reproduce it, so we have no demo
    # timing for it. Drop microwave_on from intermediate stages — outcome
    # reward will still incentivize it.
    ("SteamInMicrowave", {
        "progress_stages": [["bowl_in_micro"]],
        "retreat_stage": [],
    }),
    # Cumulative-timer task — _check_success = (success_time >= 5).
    # Pot spawns already on the active knob, so that flag is useless.
    # Real milestones: (s0) both veg in pot (parallel), (s1) grasp
    # spatula, (s2) stirring completes the cumulative timer.
    ("StirVegetables", {
        "progress_stages": [["veg2_in_pot"]],
        "retreat_stage": [],
    }),
    # Parallel 2 progress, then 1 sequential progress + retreat
    ("StoreLeftoversInBowl", {
        "progress_stages": [["chicken_in_bowl"]],
        "retreat_stage": ["gripper_far"],
    }),
    # Cumulative-timer task — _check_success = (washed_time >= 25).
    # Intermediate milestone: water turned on at the sink.
    ("WashLettuce", {
        "progress_stages": [["water_on"]],
        "retreat_stage": [],
    }),
])


def get_stage_def(task_name: str):
    """Return list of PROGRESS stage sets for a task.

    Output: list of lists of subtask names — progress stages ONLY. The
    retreat stage (gripper_*_far) is intentionally NOT included: retreating
    earns no dense progress reward. Its only incentive is the one-shot
    +OUTCOME_REWARD, which is unreachable unless the gripper retreats (the
    simulator requires retreat before declaring success).
    """
    cfg = TASK_STAGES.get(task_name)
    if cfg is None:
        raise KeyError(f"No stage def for task: {task_name}")
    return list(cfg["progress_stages"])


def get_n_stages(task_name: str) -> int:
    return len(get_stage_def(task_name))
