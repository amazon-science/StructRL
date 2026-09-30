"""Decomposer-sensitivity study (App. B.3, Table 8): stages proposed by Qwen3.5-9B.

Stage/subtask structure proposed by **Qwen3.5-9B** (thinking disabled) from
ONLY the task language command + decomposition requirements. Mechanical
post-processing (App. B.2): ground to checkable signal vocabulary,
DROP unsignaled/no-timing items (open/close cabinet-drawer-faucet, lift/move,
waits, retreat, duplicates, grasp_lettuce 35%-cov, microwave_on no-replay),
keep LLM stage order; emptied stages collapse.
Raw output: density/decomposer/qwen3.5-9b_output.json. Uses density/checkers/subtask_checker_n5.py.
"""
from collections import OrderedDict


TASK_STAGES = OrderedDict([
    # LLM: [[open drawer*, grasp straw], [close drawer*, place straw in/on glass cup]]
    ("DeliverStraw", {
        "progress_stages": [["grasp_straw"], ["straw_in_glass_cup"]],
        "retreat_stage": ["gripper_far"],
    }),
    # LLM: [[turn on toaster, wait for lever*], [grasp bread, place bread on plate]]
    ("GetToastedBread", {
        "progress_stages": [["toaster_on"], ["grasp_toast", "toast_on_plate"]],
        "retreat_stage": ["gripper_far"],
    }),
    # LLM: [[grasp kettle, lift*, move*], [place kettle on burner], [turn on burner]]
    ("KettleBoiling", {
        "progress_stages": [["grasp_kettle"], ["kettle_on_stove"], ["kettle_on_active_burner"]],
        "retreat_stage": ["gripper_far"],
    }),
    # LLM: [[grasp cup, grasp bowl], [place cup in dw, place bowl in dw], [close door]]
    ("LoadDishwasher", {
        "progress_stages": [
            ["grasp_dish0", "grasp_dish1"],
            ["dish0_on_rack", "dish1_on_rack"],
            ["dishwasher_closed"],
        ],
        "retreat_stage": [],
    }),
    # LLM: [[grasp eggplant, grasp chicken], [place e in tupper, place c in tupper],
    #       [place e in tupper, place c in tupper]]  (2nd occurrence -> object index 1)
    ("PackIdenticalLunches", {
        "progress_stages": [
            ["grasp_vegetable0", "grasp_meat0"],
            ["vegetable0_packed", "meat0_packed"],
            ["vegetable1_packed", "meat1_packed"],
        ],
        "retreat_stage": ["gripper_far"],
    }),
    # LLM: [[grasp pan, grasp sponge], [place pan in sink, place sponge in sink], [turn on water]]
    ("PreSoakPan", {
        "progress_stages": [
            ["grasp_pan", "grasp_sponge"],
            ["pan_in_sink", "sponge_in_sink"],
            ["water_on"],
        ],
        "retreat_stage": ["gripper_far_pan", "gripper_far_sponge"],
    }),
    # LLM: [[open cabinet*, grasp mug], [place mug on counter*, close cabinet*],
    #       [grasp mug(dup*), place mug under dispenser], [press start]]
    ("PrepareCoffee", {
        "progress_stages": [["grasp_mug"], ["mug_at_machine"], ["machine_turned_on"]],
        "retreat_stage": ["gripper_obj_far", "gripper_button_far"],
    }),
    # LLM single stage: [[turn on sink, maneuver+wash left, center, right]]
    ("RinseSinkBasin", {
        "progress_stages": [
            ["water_on", "washed_left", "washed_center", "washed_right"],
        ],
        "retreat_stage": [],
    }),
    # LLM: [[grasp sponge, move to board*, scrub board], [release*]]
    ("ScrubCuttingBoard", {
        "progress_stages": [["grasp_sponge", "contact_5steps"]],
        "retreat_stage": ["gripper_sponge_far"],
    }),
    # LLM: [[open cab*, grasp pan, close cab*], [place pan on rear right burner],
    #       [grasp chicken, place chicken], [turn on burner*]]
    ("SearingMeat", {
        "progress_stages": [
            ["grasp_pan"],
            ["pan_on_active_knob"],
            ["grasp_meat", "meat_in_pan"],
        ],
        "retreat_stage": ["gripper_meat_far"],
    }),
    # LLM: [[open drawer*, grasp knife, close drawer*], [place knife on board], [grasp meat, place meat on board]]
    ("SetUpCuttingStation", {
        "progress_stages": [
            ["grasp_knife"],
            ["knife_on_board"],
            ["grasp_meat", "meat_on_board"],
        ],
        "retreat_stage": ["gripper_far"],
    }),
    # LLM: [[grasp larger bowl, place larger bowl in cabinet], [grasp smaller(dup*), place smaller in/on larger]]
    ("StackBowlsCabinet", {
        "progress_stages": [
            ["grasp_any_bowl", "any_bowl_in_cabinet"],
            ["bowls_stacked"],
        ],
        "retreat_stage": ["gripper_far"],
    }),
    # LLM: [[grasp onion, place onion in bowl], [grasp bowl, place bowl in micro], [close door, press start*]]
    ("SteamInMicrowave", {
        "progress_stages": [
            ["grasp_vegetable", "veg_in_bowl"],
            ["grasp_bowl", "bowl_in_micro"],
            ["door_closed"],
        ],
        "retreat_stage": [],
    }),
    # LLM: [[grasp corn, grasp tomato], [place corn in pot, place tomato in pot], [grasp spatula], [stir for a while]]
    ("StirVegetables", {
        "progress_stages": [
            ["grasp_veg1", "grasp_veg2"],
            ["veg1_in_pot", "veg2_in_pot"],
            ["spatula_grasped"],
            ["task_complete"],
        ],
        "retreat_stage": [],
    }),
    # LLM: [[grasp chicken, grasp eggplant], [place chicken in bowl, place eggplant in bowl], [grasp bowl], [place bowl in fridge]]
    ("StoreLeftoversInBowl", {
        "progress_stages": [
            ["grasp_chicken", "grasp_vegetable"],
            ["chicken_in_bowl", "vegetable_in_bowl"],
            ["grasp_bowl"],
            ["bowl_in_fridge"],
        ],
        "retreat_stage": ["gripper_far"],
    }),
    # LLM: [[grasp lettuce*, open faucet(dup), turn on faucet], [place lettuce in sink*, run water over lettuce], [turn off*, place in bowl*]]
    ("WashLettuce", {
        "progress_stages": [["water_on"], ["lettuce_under_water"]],
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
