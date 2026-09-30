"""Density N̄ = 29 (Sec. 4.5): 470 signals.

Each N̄ = 5 milestone is refined into a chain of motion checkpoints named <milestone>__s<k>, with
the N̄ = 5 stage order. Uses density/checkers/subtask_checker_n29.py.
"""
from collections import OrderedDict


TASK_STAGES = OrderedDict([
    ("DeliverStraw", {
        "progress_stages": [
            ['grasp_straw__s1', 'grasp_straw__s2', 'grasp_straw__s3', 'grasp_straw__s4', 'grasp_straw__s5', 'grasp_straw__s6'],
            ['straw_in_glass_cup__s1', 'straw_in_glass_cup__s2', 'straw_in_glass_cup__s3', 'straw_in_glass_cup__s4', 'straw_in_glass_cup__s5', 'straw_in_glass_cup__s6'],
        ],
        "retreat_stage": ['gripper_far'],
    }),
    ("GetToastedBread", {
        "progress_stages": [
            ['toaster_on__s1', 'toaster_on__s2', 'toaster_on__s3', 'toaster_on__s4', 'toaster_on__s5', 'toaster_on__s6', 'toaster_on__s7'],
            ['grasp_toast__s1', 'grasp_toast__s2', 'grasp_toast__s3', 'grasp_toast__s4', 'grasp_toast__s5', 'grasp_toast__s6'],
            ['toast_on_plate__s1', 'toast_on_plate__s2', 'toast_on_plate__s3', 'toast_on_plate__s4', 'toast_on_plate__s5', 'toast_on_plate__s6'],
        ],
        "retreat_stage": ['gripper_far'],
    }),
    ("KettleBoiling", {
        "progress_stages": [
            ['grasp_kettle__s1', 'grasp_kettle__s2', 'grasp_kettle__s3', 'grasp_kettle__s4', 'grasp_kettle__s5', 'grasp_kettle__s6'],
            ['kettle_on_stove__s1', 'kettle_on_stove__s2', 'kettle_on_stove__s3', 'kettle_on_stove__s4', 'kettle_on_stove__s5', 'kettle_on_stove__s6'],
            ['kettle_on_active_burner__s1', 'kettle_on_active_burner__s2', 'kettle_on_active_burner__s3', 'kettle_on_active_burner__s4', 'kettle_on_active_burner__s5', 'kettle_on_active_burner__s6'],
        ],
        "retreat_stage": ['gripper_far'],
    }),
    ("LoadDishwasher", {
        "progress_stages": [
            ['grasp_dish0__s1', 'grasp_dish0__s2', 'grasp_dish0__s3', 'grasp_dish0__s4', 'grasp_dish0__s5', 'grasp_dish0__s6', 'dish0_on_rack__s1', 'dish0_on_rack__s2', 'dish0_on_rack__s3', 'dish0_on_rack__s4', 'dish0_on_rack__s5', 'dish0_on_rack__s6', 'grasp_dish1__s1', 'grasp_dish1__s2', 'grasp_dish1__s3', 'grasp_dish1__s4', 'grasp_dish1__s5', 'grasp_dish1__s6', 'dish1_on_rack__s1', 'dish1_on_rack__s2', 'dish1_on_rack__s3', 'dish1_on_rack__s4', 'dish1_on_rack__s5', 'dish1_on_rack__s6'],
            ['door_half_closed__s1', 'door_half_closed__s2', 'door_half_closed__s3', 'door_half_closed__s4', 'door_half_closed__s5', 'door_half_closed__s6'],
            ['dishwasher_closed__s1', 'dishwasher_closed__s2', 'dishwasher_closed__s3', 'dishwasher_closed__s4', 'dishwasher_closed__s5'],
        ],
        "retreat_stage": [],
    }),
    ("PackIdenticalLunches", {
        "progress_stages": [
            ['grasp_vegetable0__s1', 'grasp_vegetable0__s2', 'grasp_vegetable0__s3', 'grasp_vegetable0__s4', 'grasp_vegetable0__s5', 'grasp_vegetable0__s6', 'vegetable0_packed__s1', 'vegetable0_packed__s2', 'vegetable0_packed__s3', 'vegetable0_packed__s4', 'vegetable0_packed__s5', 'vegetable0_packed__s6', 'grasp_vegetable1__s1', 'grasp_vegetable1__s2', 'grasp_vegetable1__s3', 'grasp_vegetable1__s4', 'grasp_vegetable1__s5', 'grasp_vegetable1__s6', 'vegetable1_packed__s1', 'vegetable1_packed__s2', 'vegetable1_packed__s3', 'vegetable1_packed__s4', 'vegetable1_packed__s5', 'vegetable1_packed__s6', 'grasp_meat0__s1', 'grasp_meat0__s2', 'grasp_meat0__s3', 'grasp_meat0__s4', 'grasp_meat0__s5', 'grasp_meat0__s6', 'meat0_packed__s1', 'meat0_packed__s2', 'meat0_packed__s3', 'meat0_packed__s4', 'meat0_packed__s5', 'meat0_packed__s6', 'grasp_meat1__s1', 'grasp_meat1__s2', 'grasp_meat1__s3', 'grasp_meat1__s4', 'grasp_meat1__s5', 'grasp_meat1__s6', 'meat1_packed__s1', 'meat1_packed__s2', 'meat1_packed__s3', 'meat1_packed__s4', 'meat1_packed__s5', 'meat1_packed__s6'],
        ],
        "retreat_stage": ['gripper_far'],
    }),
    ("PreSoakPan", {
        "progress_stages": [
            ['grasp_pan__s1', 'grasp_pan__s2', 'grasp_pan__s3', 'grasp_pan__s4', 'grasp_pan__s5', 'grasp_pan__s6', 'pan_in_sink__s1', 'pan_in_sink__s2', 'pan_in_sink__s3', 'pan_in_sink__s4', 'pan_in_sink__s5', 'pan_in_sink__s6', 'grasp_sponge__s1', 'grasp_sponge__s2', 'grasp_sponge__s3', 'grasp_sponge__s4', 'grasp_sponge__s5', 'grasp_sponge__s6', 'sponge_in_sink__s1', 'sponge_in_sink__s2', 'sponge_in_sink__s3', 'sponge_in_sink__s4', 'sponge_in_sink__s5', 'sponge_in_sink__s6', 'water_on__s1', 'water_on__s2', 'water_on__s3', 'water_on__s4', 'water_on__s5', 'water_on__s6'],
        ],
        "retreat_stage": ['gripper_far_pan', 'gripper_far_sponge'],
    }),
    ("PrepareCoffee", {
        "progress_stages": [
            ['grasp_mug__s1', 'grasp_mug__s2', 'grasp_mug__s3', 'grasp_mug__s4', 'grasp_mug__s5', 'grasp_mug__s6'],
            ['mug_at_machine__s1', 'mug_at_machine__s2', 'mug_at_machine__s3', 'mug_at_machine__s4', 'mug_at_machine__s5', 'mug_at_machine__s6'],
            ['machine_turned_on__s1', 'machine_turned_on__s2', 'machine_turned_on__s3', 'machine_turned_on__s4', 'machine_turned_on__s5', 'machine_turned_on__s6', 'machine_turned_on__s7'],
        ],
        "retreat_stage": ['gripper_obj_far', 'gripper_button_far'],
    }),
    ("RinseSinkBasin", {
        "progress_stages": [
            ['water_on__s1', 'water_on__s2', 'water_on__s3', 'water_on__s4', 'water_on__s5', 'water_on__s6'],
            ['washed_left__s1', 'washed_left__s2', 'washed_left__s3', 'washed_left__s4', 'washed_left__s5', 'washed_left__s6', 'washed_center__s1', 'washed_center__s2', 'washed_center__s3', 'washed_center__s4', 'washed_center__s5', 'washed_center__s6', 'washed_right__s1', 'washed_right__s2', 'washed_right__s3', 'washed_right__s4', 'washed_right__s5', 'washed_right__s6'],
        ],
        "retreat_stage": [],
    }),
    ("ScrubCuttingBoard", {
        "progress_stages": [
            ['grasp_sponge__s1', 'grasp_sponge__s2', 'grasp_sponge__s3', 'grasp_sponge__s4', 'grasp_sponge__s5', 'grasp_sponge__s6'],
            ['contact_1step__s1', 'contact_1step__s2', 'contact_1step__s3', 'contact_1step__s4', 'contact_1step__s5', 'contact_1step__s6'],
            ['contact_3steps__s1', 'contact_3steps__s2', 'contact_3steps__s3', 'contact_3steps__s4', 'contact_3steps__s5', 'sweep_range_0p05m__s1', 'sweep_range_0p05m__s2', 'sweep_range_0p05m__s3', 'sweep_range_0p05m__s4', 'sweep_range_0p05m__s5'],
            ['contact_5steps__s1', 'contact_5steps__s2', 'contact_5steps__s3', 'contact_5steps__s4', 'contact_5steps__s5', 'sweep_range_0p1m__s1', 'sweep_range_0p1m__s2', 'sweep_range_0p1m__s3', 'sweep_range_0p1m__s4', 'sweep_range_0p1m__s5'],
        ],
        "retreat_stage": ['gripper_sponge_far'],
    }),
    ("SearingMeat", {
        "progress_stages": [
            ['grasp_pan__s1', 'grasp_pan__s2', 'grasp_pan__s3', 'grasp_pan__s4', 'grasp_pan__s5', 'grasp_pan__s6', 'pan_on_stove__s1', 'pan_on_stove__s2', 'pan_on_stove__s3', 'pan_on_stove__s4', 'pan_on_stove__s5', 'pan_on_stove__s6', 'pan_on_active_knob__s1', 'pan_on_active_knob__s2', 'pan_on_active_knob__s3', 'pan_on_active_knob__s4', 'pan_on_active_knob__s5', 'pan_on_active_knob__s6'],
            ['grasp_meat__s1', 'grasp_meat__s2', 'grasp_meat__s3', 'grasp_meat__s4', 'grasp_meat__s5', 'grasp_meat__s6', 'meat_in_pan__s1', 'meat_in_pan__s2', 'meat_in_pan__s3', 'meat_in_pan__s4', 'meat_in_pan__s5', 'meat_in_pan__s6'],
        ],
        "retreat_stage": ['gripper_meat_far'],
    }),
    ("SetUpCuttingStation", {
        "progress_stages": [
            ['grasp_meat__s1', 'grasp_meat__s2', 'grasp_meat__s3', 'grasp_meat__s4', 'grasp_meat__s5', 'grasp_meat__s6', 'meat_on_board__s1', 'meat_on_board__s2', 'meat_on_board__s3', 'meat_on_board__s4', 'meat_on_board__s5', 'meat_on_board__s6', 'grasp_knife__s1', 'grasp_knife__s2', 'grasp_knife__s3', 'grasp_knife__s4', 'grasp_knife__s5', 'grasp_knife__s6', 'knife_on_board__s1', 'knife_on_board__s2', 'knife_on_board__s3', 'knife_on_board__s4', 'knife_on_board__s5', 'knife_on_board__s6'],
        ],
        "retreat_stage": ['gripper_far'],
    }),
    ("StackBowlsCabinet", {
        "progress_stages": [
            ['grasp_any_bowl__s1', 'grasp_any_bowl__s2', 'grasp_any_bowl__s3', 'grasp_any_bowl__s4', 'grasp_any_bowl__s5', 'grasp_any_bowl__s6'],
            ['bowls_stacked__s1', 'bowls_stacked__s2', 'bowls_stacked__s3', 'bowls_stacked__s4', 'bowls_stacked__s5', 'bowls_stacked__s6'],
            ['any_bowl_in_cabinet__s1', 'any_bowl_in_cabinet__s2', 'any_bowl_in_cabinet__s3', 'any_bowl_in_cabinet__s4', 'any_bowl_in_cabinet__s5', 'any_bowl_in_cabinet__s6', 'both_bowls_in_cabinet__s1', 'both_bowls_in_cabinet__s2', 'both_bowls_in_cabinet__s3', 'both_bowls_in_cabinet__s4', 'both_bowls_in_cabinet__s5'],
        ],
        "retreat_stage": ['gripper_far'],
    }),
    ("SteamInMicrowave", {
        "progress_stages": [
            ['grasp_vegetable__s1', 'grasp_vegetable__s2', 'grasp_vegetable__s3', 'grasp_vegetable__s4', 'grasp_vegetable__s5', 'grasp_vegetable__s6', 'veg_in_bowl__s1', 'veg_in_bowl__s2', 'veg_in_bowl__s3', 'veg_in_bowl__s4', 'veg_in_bowl__s5', 'veg_in_bowl__s6'],
            ['grasp_bowl__s1', 'grasp_bowl__s2', 'grasp_bowl__s3', 'grasp_bowl__s4', 'grasp_bowl__s5', 'grasp_bowl__s6', 'bowl_in_micro__s1', 'bowl_in_micro__s2', 'bowl_in_micro__s3', 'bowl_in_micro__s4', 'bowl_in_micro__s5', 'bowl_in_micro__s6'],
            ['door_half_closed__s1', 'door_half_closed__s2', 'door_half_closed__s3', 'door_half_closed__s4', 'door_half_closed__s5', 'door_half_closed__s6', 'door_closed__s1', 'door_closed__s2', 'door_closed__s3', 'door_closed__s4', 'door_closed__s5', 'door_closed__s6'],
        ],
        "retreat_stage": [],
    }),
    ("StirVegetables", {
        "progress_stages": [
            ['grasp_veg1__s1', 'grasp_veg1__s2', 'grasp_veg1__s3', 'grasp_veg1__s4', 'grasp_veg1__s5', 'grasp_veg1__s6', 'veg1_in_pot__s1', 'veg1_in_pot__s2', 'veg1_in_pot__s3', 'veg1_in_pot__s4', 'veg1_in_pot__s5', 'veg1_in_pot__s6', 'grasp_veg2__s1', 'grasp_veg2__s2', 'grasp_veg2__s3', 'grasp_veg2__s4', 'grasp_veg2__s5', 'grasp_veg2__s6', 'veg2_in_pot__s1', 'veg2_in_pot__s2', 'veg2_in_pot__s3', 'veg2_in_pot__s4', 'veg2_in_pot__s5', 'veg2_in_pot__s6'],
            ['spatula_grasped__s1', 'spatula_grasped__s2', 'spatula_grasped__s3', 'spatula_grasped__s4', 'spatula_grasped__s5', 'spatula_grasped__s6'],
            ['stir_t1__s1', 'stir_t1__s2', 'stir_t1__s3', 'stir_t1__s4', 'stir_t1__s5', 'stir_t1__s6', 'stir_t3__s1', 'stir_t3__s2', 'stir_t3__s3', 'stir_t3__s4', 'stir_t3__s5', 'stir_t3__s6'],
            ['task_complete__s1', 'task_complete__s2', 'task_complete__s3', 'task_complete__s4', 'task_complete__s5'],
        ],
        "retreat_stage": [],
    }),
    ("StoreLeftoversInBowl", {
        "progress_stages": [
            ['grasp_chicken__s1', 'grasp_chicken__s2', 'grasp_chicken__s3', 'grasp_chicken__s4', 'grasp_chicken__s5', 'grasp_chicken__s6', 'chicken_in_bowl__s1', 'chicken_in_bowl__s2', 'chicken_in_bowl__s3', 'chicken_in_bowl__s4', 'chicken_in_bowl__s5', 'chicken_in_bowl__s6', 'grasp_vegetable__s1', 'grasp_vegetable__s2', 'grasp_vegetable__s3', 'grasp_vegetable__s4', 'grasp_vegetable__s5', 'grasp_vegetable__s6', 'vegetable_in_bowl__s1', 'vegetable_in_bowl__s2', 'vegetable_in_bowl__s3', 'vegetable_in_bowl__s4', 'vegetable_in_bowl__s5', 'vegetable_in_bowl__s6'],
            ['grasp_bowl__s1', 'grasp_bowl__s2', 'grasp_bowl__s3', 'grasp_bowl__s4', 'grasp_bowl__s5', 'grasp_bowl__s6', 'bowl_in_fridge__s1', 'bowl_in_fridge__s2', 'bowl_in_fridge__s3', 'bowl_in_fridge__s4', 'bowl_in_fridge__s5', 'bowl_in_fridge__s6'],
        ],
        "retreat_stage": ['gripper_far'],
    }),
    ("WashLettuce", {
        "progress_stages": [
            ['water_on__s1', 'water_on__s2', 'water_on__s3', 'water_on__s4', 'water_on__s5', 'water_on__s6'],
            ['lettuce_under_water__s1', 'lettuce_under_water__s2', 'lettuce_under_water__s3', 'lettuce_under_water__s4', 'lettuce_under_water__s5', 'lettuce_under_water__s6'],
            ['wash_t5__s1', 'wash_t5__s2', 'wash_t5__s3', 'wash_t5__s4', 'wash_t5__s5', 'wash_t10__s1', 'wash_t10__s2', 'wash_t10__s3', 'wash_t10__s4', 'wash_t10__s5', 'wash_t15__s1', 'wash_t15__s2', 'wash_t15__s3', 'wash_t15__s4', 'wash_t15__s5', 'wash_t20__s1', 'wash_t20__s2', 'wash_t20__s3', 'wash_t20__s4', 'wash_t20__s5'],
            ['task_complete__s1', 'task_complete__s2', 'task_complete__s3', 'task_complete__s4', 'task_complete__s5'],
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
