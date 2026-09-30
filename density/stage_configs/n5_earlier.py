"""Earlier-checkpoint control (App. C.3, Table 11): the N̄ = 5 stages with each predicate replaced by an
earlier motion checkpoint of the same sub-motion, taken from the N̄ = 29 chains.

Only candidates with full demonstration coverage and a non-zero demonstration firing time are
used; 78 of the 80 signals move earlier, and the 2 without a valid earlier candidate keep the
N̄ = 5 predicate. Uses density/checkers/subtask_checker_n29.py.
"""
from collections import OrderedDict


TASK_STAGES = OrderedDict([
    ("DeliverStraw", {
        "progress_stages": [
            ['grasp_straw__s2'],
            ['straw_in_glass_cup__s2'],
        ],
        "retreat_stage": ['gripper_far'],
    }),
    ("GetToastedBread", {
        "progress_stages": [
            ['toaster_on__s3'],
            ['grasp_toast__s1'],
            ['toast_on_plate__s3'],
        ],
        "retreat_stage": ['gripper_far'],
    }),
    ("KettleBoiling", {
        "progress_stages": [
            ['grasp_kettle__s1'],
            ['kettle_on_stove__s2'],
            ['kettle_on_active_burner__s4'],
        ],
        "retreat_stage": ['gripper_far'],
    }),
    ("LoadDishwasher", {
        "progress_stages": [
            ['grasp_dish0__s2', 'dish0_on_rack__s2', 'grasp_dish1__s2', 'dish1_on_rack__s2'],
            ['door_half_closed__s2'],
            ['dishwasher_closed__s2'],
        ],
        "retreat_stage": [],
    }),
    ("PackIdenticalLunches", {
        "progress_stages": [
            ['grasp_vegetable0__s3', 'vegetable0_packed__s2', 'grasp_vegetable1__s3', 'vegetable1_packed__s2', 'grasp_meat0__s3', 'meat0_packed__s3', 'grasp_meat1__s3', 'meat1_packed__s2'],
        ],
        "retreat_stage": ['gripper_far'],
    }),
    ("PreSoakPan", {
        "progress_stages": [
            ['grasp_pan__s4', 'pan_in_sink__s4', 'grasp_sponge__s3', 'sponge_in_sink__s4', 'water_on__s2'],
        ],
        "retreat_stage": ['gripper_far_pan', 'gripper_far_sponge'],
    }),
    ("PrepareCoffee", {
        "progress_stages": [
            ['grasp_mug__s3'],
            ['mug_at_machine__s4'],
            ['machine_turned_on__s4'],
        ],
        "retreat_stage": ['gripper_obj_far', 'gripper_button_far'],
    }),
    ("RinseSinkBasin", {
        "progress_stages": [
            ['water_on__s2'],
            ['washed_left__s4', 'washed_center__s6', 'washed_right__s4'],
        ],
        "retreat_stage": [],
    }),
    ("ScrubCuttingBoard", {
        "progress_stages": [
            ['grasp_sponge__s2'],
            ['contact_1step__s3'],
            ['contact_3steps__s2', 'sweep_range_0p05m__s2'],
            ['contact_5steps__s3', 'sweep_range_0p1m__s2'],
        ],
        "retreat_stage": ['gripper_sponge_far'],
    }),
    ("SearingMeat", {
        "progress_stages": [
            ['grasp_pan__s2', 'pan_on_stove__s3', 'pan_on_active_knob__s3'],
            ['grasp_meat__s2', 'meat_in_pan__s2'],
        ],
        "retreat_stage": ['gripper_meat_far'],
    }),
    ("SetUpCuttingStation", {
        "progress_stages": [
            ['grasp_meat__s2', 'meat_on_board__s1', 'grasp_knife__s1', 'knife_on_board__s2'],
        ],
        "retreat_stage": ['gripper_far'],
    }),
    ("StackBowlsCabinet", {
        "progress_stages": [
            ['grasp_any_bowl__s2'],
            ['bowls_stacked__s3'],
            ['any_bowl_in_cabinet__s1', 'both_bowls_in_cabinet__s4'],
        ],
        "retreat_stage": ['gripper_far'],
    }),
    ("SteamInMicrowave", {
        "progress_stages": [
            ['grasp_vegetable__s3', 'veg_in_bowl__s3'],
            ['grasp_bowl__s2', 'bowl_in_micro__s2'],
            ['door_half_closed__s3', 'door_closed__s3'],
        ],
        "retreat_stage": [],
    }),
    ("StirVegetables", {
        "progress_stages": [
            ['grasp_veg1__s3', 'veg1_in_pot__s3', 'grasp_veg2__s2', 'veg2_in_pot__s3'],
            ['spatula_grasped__s2'],
            ['stir_t1__s3', 'stir_t3__s1'],
            ['task_complete__s5'],
        ],
        "retreat_stage": [],
    }),
    ("StoreLeftoversInBowl", {
        "progress_stages": [
            ['grasp_chicken__s2', 'chicken_in_bowl__s3', 'grasp_vegetable__s3', 'vegetable_in_bowl__s3'],
            ['grasp_bowl__s3', 'bowl_in_fridge__s3'],
        ],
        "retreat_stage": ['gripper_far'],
    }),
    ("WashLettuce", {
        "progress_stages": [
            ['water_on__s3'],
            ['lettuce_under_water__s3'],
            ['wash_t5__s2', 'wash_t10__s2', 'wash_t15__s2', 'wash_t20__s2'],
            ['task_complete__s2'],
        ],
        "retreat_stage": [],
    }),
])


def get_stage_def(task_name: str):
    """Progress stages only. The retreat stage is deliberately excluded."""
    cfg = TASK_STAGES.get(task_name)
    if cfg is None:
        raise KeyError(f"No stage def for task: {task_name}")
    return list(cfg["progress_stages"])


def get_n_stages(task_name: str) -> int:
    return len(get_stage_def(task_name))
