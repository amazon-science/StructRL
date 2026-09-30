"""Density N̄ = 5 (Sec. 4.5): 80 signals in 41 ordered stages.

Adds grasp milestones and activity-timer steps to the default decomposition. A grasp shares the
stage of the placement it enables; timer steps form trailing stages. Same format as
structrl/stage_config.py; uses density/checkers/subtask_checker_n5.py.
"""
from collections import OrderedDict


TASK_STAGES = OrderedDict([
    ("DeliverStraw", {
        "progress_stages": [
            ["grasp_straw"],
            ["straw_in_glass_cup"],
        ],
        "retreat_stage": ["gripper_far"],
    }),
    ("GetToastedBread", {
        "progress_stages": [
            ["toaster_on"],
            ["grasp_toast"],
            ["toast_on_plate"],
        ],
        "retreat_stage": ["gripper_far"],
    }),
    ("KettleBoiling", {
        "progress_stages": [
            ["grasp_kettle"],
            ["kettle_on_stove"],
            ["kettle_on_active_burner"],
        ],
        "retreat_stage": ["gripper_far"],
    }),
    ("LoadDishwasher", {
        "progress_stages": [
            # both dishes: grasp+place parallel pool (either dish first)
            ["grasp_dish0", "dish0_on_rack", "grasp_dish1", "dish1_on_rack"],
            ["door_half_closed"],
            ["dishwasher_closed"],
        ],
        "retreat_stage": [],
    }),
    ("PackIdenticalLunches", {
        "progress_stages": [
            # all four objects: grasp + packed, parallel pool
            ["grasp_vegetable0", "vegetable0_packed",
             "grasp_vegetable1", "vegetable1_packed",
             "grasp_meat0", "meat0_packed",
             "grasp_meat1", "meat1_packed"],
            # tupper*_complete dropped: they fire at the SAME step as the 4th
            # placement (derivative signals, gap-to-prev budget = 0).
        ],
        "retreat_stage": ["gripper_far"],
    }),
    ("PreSoakPan", {
        "progress_stages": [
            ["grasp_pan", "pan_in_sink", "grasp_sponge", "sponge_in_sink",
             "water_on"],
        ],
        "retreat_stage": ["gripper_far_pan", "gripper_far_sponge"],
    }),
    ("PrepareCoffee", {
        "progress_stages": [
            ["grasp_mug"],
            ["mug_at_machine"],
            ["machine_turned_on"],
        ],
        "retreat_stage": ["gripper_obj_far", "gripper_button_far"],
    }),
    ("RinseSinkBasin", {
        "progress_stages": [
            ["water_on"],
            ["washed_left", "washed_center", "washed_right"],
        ],
        "retreat_stage": [],
    }),
    ("ScrubCuttingBoard", {
        "progress_stages": [
            ["grasp_sponge"],
            ["contact_1step"],
            ["contact_3steps", "sweep_range_0p05m"],
            ["contact_5steps", "sweep_range_0p1m"],
        ],
        "retreat_stage": ["gripper_sponge_far"],
    }),
    ("SearingMeat", {
        "progress_stages": [
            # pan leg then meat leg (demo order: place pan, then meat)
            ["grasp_pan", "pan_on_stove", "pan_on_active_knob"],
            ["grasp_meat", "meat_in_pan"],
        ],
        "retreat_stage": ["gripper_meat_far"],
    }),
    ("SetUpCuttingStation", {
        "progress_stages": [
            ["grasp_meat", "meat_on_board", "grasp_knife", "knife_on_board"],
        ],
        "retreat_stage": ["gripper_far"],
    }),
    ("StackBowlsCabinet", {
        "progress_stages": [
            ["grasp_any_bowl"],
            ["bowls_stacked"],
            ["any_bowl_in_cabinet", "both_bowls_in_cabinet"],
        ],
        "retreat_stage": ["gripper_far"],
    }),
    ("SteamInMicrowave", {
        "progress_stages": [
            ["grasp_vegetable", "veg_in_bowl"],
            ["grasp_bowl", "bowl_in_micro"],
            ["door_half_closed", "door_closed"],
            # microwave_on kept OUT of stages (latched attr not reproducible
            # on demo replay — same caveat as v1); outcome reward covers it.
        ],
        "retreat_stage": [],
    }),
    ("StirVegetables", {
        "progress_stages": [
            ["grasp_veg1", "veg1_in_pot", "grasp_veg2", "veg2_in_pot"],
            ["spatula_grasped"],
            ["stir_t1", "stir_t3"],
            ["task_complete"],
        ],
        "retreat_stage": [],
    }),
    ("StoreLeftoversInBowl", {
        "progress_stages": [
            ["grasp_chicken", "chicken_in_bowl",
             "grasp_vegetable", "vegetable_in_bowl"],
            ["grasp_bowl", "bowl_in_fridge"],
        ],
        "retreat_stage": ["gripper_far"],
    }),
    ("WashLettuce", {
        "progress_stages": [
            ["water_on"],
            ["lettuce_under_water"],
            ["wash_t5", "wash_t10", "wash_t15", "wash_t20"],
            ["task_complete"],
        ],
        "retreat_stage": [],
    }),
])


def get_stage_def(task_name: str):
    cfg = TASK_STAGES.get(task_name)
    if cfg is None:
        raise KeyError(f"No stage def for task: {task_name}")
    return list(cfg["progress_stages"])


def get_n_stages(task_name: str) -> int:
    return len(get_stage_def(task_name))
