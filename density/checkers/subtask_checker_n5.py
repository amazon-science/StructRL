"""DENSER (v2) per-task subtask state extractor — ABLATION ONLY.

Completely isolated from the production `structrl/subtask_checker.py`:
nothing in structrl/ or RLinf imports this file. It exists so we can
measure (a) whether the denser milestones fire reliably on demo replay
and (b) their human-demo timing, WITHOUT touching the running setup.

Density additions vs v1:
  1. grasp milestones  — OU.check_obj_grasped() for every carried object
     (raw signal is non-monotone: True while held, False after release.
      Timing extraction latches first-True; an RL consumer must latch too.)
  2. timer ladders     — cumulative env counters exposed at intermediate
     thresholds (StirVegetables success_time, WashLettuce washed_time,
     ScrubCuttingBoard board_contact_timer / sweep range).
  3. state intermediates — door half-closed, water_on setup, under-water
     contact, both-bowls-in-cabinet, per-object packed flags.

Same output contract as v1:
    {"progress": OrderedDict[name->bool], "quiescence": OrderedDict[name->bool]}
"""
from collections import OrderedDict
import numpy as np
import robocasa.utils.object_utils as OU


def _grasped(env, name):
    try:
        return bool(OU.check_obj_grasped(env, name))
    except Exception:
        return False


def _in_recep(env, o, r, **kw):
    try:
        return bool(OU.check_obj_in_receptacle(env, o, r, **kw))
    except Exception:
        return False


def get_subtask_states(env) -> dict:
    cls_name = type(env).__name__

    # ----- DeliverStraw -----
    if cls_name == "DeliverStraw":
        progress = OrderedDict([
            ("grasp_straw",       _grasped(env, "straw") or _in_recep(env, "straw", "glass_cup", th=0.5)),
            ("straw_in_glass_cup", _in_recep(env, "straw", "glass_cup", th=0.5)),
        ])
        quiescence = OrderedDict([
            ("gripper_far", OU.gripper_obj_far(env, obj_name="straw")),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- GetToastedBread -----
    if cls_name == "GetToastedBread":
        progress = OrderedDict([
            ("toaster_on",     bool(getattr(env, "toaster_on", False))),
            ("grasp_toast",    _grasped(env, "obj") or _in_recep(env, "obj", "plate")),
            ("toast_on_plate", _in_recep(env, "obj", "plate")),
        ])
        quiescence = OrderedDict([
            ("gripper_far", OU.gripper_obj_far(env, "obj")),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- KettleBoiling -----
    if cls_name == "KettleBoiling":
        kettle = env.objects["obj"]
        kettle_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[kettle.name]])[0:2]
        knobs_state = env.stove.get_knobs_state(env=env)
        obj_on_stove = OU.check_obj_fixture_contact(env, "obj", env.stove)
        any_kettle_on_active_burner = False
        if obj_on_stove:
            for location, site in env.stove.burner_sites.items():
                if site is not None:
                    burner_pos = np.array(env.sim.data.get_site_xpos(site.get("name")))[0:2]
                    dist = np.linalg.norm(burner_pos - kettle_pos)
                    burner_on = (env.stove.is_burner_on(env=env, burner_loc=location)
                                 if location in knobs_state else False)
                    if dist < 0.15 and burner_on:
                        any_kettle_on_active_burner = True
                        break
        progress = OrderedDict([
            ("grasp_kettle",            _grasped(env, "obj") or obj_on_stove),
            ("kettle_on_stove",         obj_on_stove),
            ("kettle_on_active_burner", any_kettle_on_active_burner),
        ])
        quiescence = OrderedDict([
            ("gripper_far", OU.gripper_obj_far(env)),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- LoadDishwasher -----
    if cls_name == "LoadDishwasher":
        try:
            door_half = env.dishwasher.is_closed(env, th=0.3)
        except Exception:
            door_half = False
        progress = OrderedDict([
            ("grasp_dish0",       _grasped(env, "dish0") or env.dishwasher.check_rack_contact(env, "dish0")),
            ("dish0_on_rack",     env.dishwasher.check_rack_contact(env, "dish0")),
            ("grasp_dish1",       _grasped(env, "dish1") or env.dishwasher.check_rack_contact(env, "dish1")),
            ("dish1_on_rack",     env.dishwasher.check_rack_contact(env, "dish1")),
            ("door_half_closed",  door_half),
            ("dishwasher_closed", env.dishwasher.is_closed(env, th=0.05)),
        ])
        return {"progress": progress, "quiescence": OrderedDict()}

    # ----- PackIdenticalLunches -----
    if cls_name == "PackIdenticalLunches":
        packed = OrderedDict()
        veg_in_0, veg_in_1, meat_in_0, meat_in_1 = [], [], [], []
        for veg in ["vegetable0", "vegetable1"]:
            in0 = _in_recep(env, veg, "tupperware0")
            in1 = _in_recep(env, veg, "tupperware1")
            packed[f"{veg}_packed"] = in0 or in1
            if in0:
                veg_in_0.append(veg)
            if in1:
                veg_in_1.append(veg)
        for meat in ["meat0", "meat1"]:
            in0 = _in_recep(env, meat, "tupperware0")
            in1 = _in_recep(env, meat, "tupperware1")
            packed[f"{meat}_packed"] = in0 or in1
            if in0:
                meat_in_0.append(meat)
            if in1:
                meat_in_1.append(meat)
        all_objs = veg_in_0 + veg_in_1 + meat_in_0 + meat_in_1
        gripper_far = all(OU.gripper_obj_far(env, o) for o in all_objs) if all_objs else True
        progress = OrderedDict([
            ("grasp_vegetable0", _grasped(env, "vegetable0") or packed["vegetable0_packed"]),
            ("grasp_vegetable1", _grasped(env, "vegetable1") or packed["vegetable1_packed"]),
            ("grasp_meat0",      _grasped(env, "meat0") or packed["meat0_packed"]),
            ("grasp_meat1",      _grasped(env, "meat1") or packed["meat1_packed"]),
        ])
        progress.update(packed)
        progress["tupper0_complete"] = len(veg_in_0) == 1 and len(meat_in_0) == 1
        progress["tupper1_complete"] = len(veg_in_1) == 1 and len(meat_in_1) == 1
        quiescence = OrderedDict([
            ("gripper_far", gripper_far),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- PreSoakPan -----
    if cls_name == "PreSoakPan":
        h = env.sink.get_handle_state(env=env)
        progress = OrderedDict([
            ("grasp_pan",      _grasped(env, "obj1") or OU.obj_inside_of(env, "obj1", env.sink, partial_check=False)),
            ("pan_in_sink",    OU.obj_inside_of(env, "obj1", env.sink, partial_check=False)),
            ("grasp_sponge",   _grasped(env, "obj2") or OU.obj_inside_of(env, "obj2", env.sink, partial_check=False)),
            ("sponge_in_sink", OU.obj_inside_of(env, "obj2", env.sink, partial_check=False)),
            ("water_on",       bool(h["water_on"])),
        ])
        quiescence = OrderedDict([
            ("gripper_far_pan",    OU.gripper_obj_far(env, "obj1")),
            ("gripper_far_sponge", OU.gripper_obj_far(env, "obj2")),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- PrepareCoffee -----
    if cls_name == "PrepareCoffee":
        progress = OrderedDict([
            ("grasp_mug",         _grasped(env, "obj") or env.coffee_machine.check_receptacle_placement_for_pouring(env, "obj")),
            ("mug_at_machine",    env.coffee_machine.check_receptacle_placement_for_pouring(env, "obj")),
            ("machine_turned_on", bool(env.coffee_machine._turned_on)),
        ])
        quiescence = OrderedDict([
            ("gripper_obj_far",    OU.gripper_obj_far(env)),
            ("gripper_button_far", env.coffee_machine.gripper_button_far(env)),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- RinseSinkBasin -----
    if cls_name == "RinseSinkBasin":
        wl = getattr(env, "washed_loc", [False, False, False])
        try:
            water_on = bool(env.sink.get_handle_state(env=env)["water_on"])
        except Exception:
            water_on = False
        progress = OrderedDict([
            ("water_on",      water_on),
            ("washed_left",   bool(wl[0])),
            ("washed_center", bool(wl[1])),
            ("washed_right",  bool(wl[2])),
        ])
        return {"progress": progress, "quiescence": OrderedDict()}

    # ----- ScrubCuttingBoard -----
    if cls_name == "ScrubCuttingBoard":
        sweep_range = 0.0
        if getattr(env, "board_contact_positions", None):
            positions = np.array(env.board_contact_positions)
            sweep_range = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
        timer = int(getattr(env, "board_contact_timer", 0))
        progress = OrderedDict([
            ("grasp_sponge",     _grasped(env, "sponge") or timer >= 1),
            ("contact_1step",    timer >= 1),
            ("contact_3steps",   timer >= 3),
            ("contact_5steps",   timer >= 5),
            ("sweep_range_0p05m", sweep_range >= 0.05),
            ("sweep_range_0p1m", sweep_range >= 0.1),
        ])
        quiescence = OrderedDict([
            ("gripper_sponge_far", OU.gripper_obj_far(env, "sponge", th=0.15)),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- SearingMeat -----
    if cls_name == "SearingMeat":
        try:
            pan_on_stove = OU.check_obj_fixture_contact(env, "pan", env.stove)
        except Exception:
            pan_on_stove = False
        progress = OrderedDict([
            ("grasp_pan",          _grasped(env, "pan") or pan_on_stove),
            ("pan_on_stove",       pan_on_stove),
            ("pan_on_active_knob", env.stove.check_obj_location_on_stove(env, "pan", threshold=0.15) == env.knob),
            ("grasp_meat",         _grasped(env, "meat") or _in_recep(env, "meat", "pan", th=0.07)),
            ("meat_in_pan",        _in_recep(env, "meat", "pan", th=0.07)),
        ])
        quiescence = OrderedDict([
            ("gripper_meat_far", OU.gripper_obj_far(env, obj_name="meat")),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- SetUpCuttingStation -----
    if cls_name == "SetUpCuttingStation":
        progress = OrderedDict([
            ("grasp_meat",     _grasped(env, "meat") or _in_recep(env, "meat", "receptacle")),
            ("meat_on_board",  _in_recep(env, "meat", "receptacle")),
            ("grasp_knife",    _grasped(env, "knife") or _in_recep(env, "knife", "receptacle")),
            ("knife_on_board", _in_recep(env, "knife", "receptacle")),
        ])
        quiescence = OrderedDict([
            ("gripper_far", OU.gripper_obj_far(env, "knife") and OU.gripper_obj_far(env, "receptacle")),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- StackBowlsCabinet -----
    if cls_name == "StackBowlsCabinet":
        try:
            b1_in_cab = OU.obj_inside_of(env, "bowl1", env.cabinet)
        except Exception:
            b1_in_cab = False
        try:
            b2_in_cab = OU.obj_inside_of(env, "bowl2", env.cabinet)
        except Exception:
            b2_in_cab = False
        try:
            stacked = (
                _in_recep(env, "bowl2", "bowl1")
                or _in_recep(env, "bowl1", "bowl2")
            )
        except Exception:
            stacked = False
        progress = OrderedDict([
            ("grasp_any_bowl",       _grasped(env, "bowl1") or _grasped(env, "bowl2") or stacked),
            ("bowls_stacked",        bool(stacked)),
            ("any_bowl_in_cabinet",  bool(b1_in_cab or b2_in_cab)),
            ("both_bowls_in_cabinet", bool(b1_in_cab and b2_in_cab)),
        ])
        quiescence = OrderedDict([
            ("gripper_far",
             OU.gripper_obj_far(env, "bowl1") and OU.gripper_obj_far(env, "bowl2")),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- SteamInMicrowave -----
    if cls_name == "SteamInMicrowave":
        try:
            veg_in_bowl = _in_recep(env, "vegetable", "bowl")
        except Exception:
            veg_in_bowl = False
        try:
            bowl_in_micro = OU.obj_inside_of(env, "bowl", env.microwave)
        except Exception:
            bowl_in_micro = False
        try:
            door_half = env.microwave.is_closed(env, th=0.3)
        except Exception:
            door_half = False
        try:
            microwave_closed = env.microwave.is_closed(env)
        except Exception:
            microwave_closed = False
        try:
            microwave_on = bool(env.microwave.get_state()["turned_on"])
        except Exception:
            microwave_on = False
        progress = OrderedDict([
            ("grasp_vegetable",  _grasped(env, "vegetable") or veg_in_bowl),
            ("veg_in_bowl",      veg_in_bowl),
            ("grasp_bowl",       _grasped(env, "bowl") or bowl_in_micro),
            ("bowl_in_micro",    bowl_in_micro),
            ("door_half_closed", door_half),
            ("door_closed",      microwave_closed),
            ("microwave_on",     microwave_on),
        ])
        return {"progress": progress, "quiescence": OrderedDict()}

    # ----- StirVegetables -----
    if cls_name == "StirVegetables":
        st = int(getattr(env, "success_time", 0))
        try:
            done = bool(env._check_success())
        except Exception:
            done = False
        progress = OrderedDict([
            ("grasp_veg1",      _grasped(env, "veg1") or _in_recep(env, "veg1", "pot")),
            ("veg1_in_pot",     _in_recep(env, "veg1", "pot")),
            ("grasp_veg2",      _grasped(env, "veg2") or _in_recep(env, "veg2", "pot")),
            ("veg2_in_pot",     _in_recep(env, "veg2", "pot")),
            ("spatula_grasped", _grasped(env, "spatula")),
            ("stir_t1",         st >= 1),
            ("stir_t3",         st >= 3),
            ("task_complete",   done),
        ])
        return {"progress": progress, "quiescence": OrderedDict()}

    # ----- StoreLeftoversInBowl -----
    if cls_name == "StoreLeftoversInBowl":
        try:
            bowl_in_fridge = env.fridge.check_rack_contact(env, "bowl")
        except Exception:
            bowl_in_fridge = False
        try:
            gripper_far = OU.gripper_obj_far(env, "bowl")
        except Exception:
            gripper_far = False
        progress = OrderedDict([
            ("grasp_chicken",     _grasped(env, "chicken_drumstick") or _in_recep(env, "chicken_drumstick", "bowl")),
            ("chicken_in_bowl",   _in_recep(env, "chicken_drumstick", "bowl")),
            ("grasp_vegetable",   _grasped(env, "vegetable") or _in_recep(env, "vegetable", "bowl")),
            ("vegetable_in_bowl", _in_recep(env, "vegetable", "bowl")),
            ("grasp_bowl",        _grasped(env, "bowl") or bowl_in_fridge),
            ("bowl_in_fridge",    bowl_in_fridge),
        ])
        quiescence = OrderedDict([
            ("gripper_far", gripper_far),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- WashLettuce -----
    if cls_name == "WashLettuce":
        try:
            water_on = bool(env.sink.get_handle_state(env=env)["water_on"])
        except Exception:
            water_on = False
        try:
            under_water = bool(env.sink.check_obj_under_water(env, "lettuce"))
        except Exception:
            under_water = False
        wt = int(getattr(env, "washed_time", 0))
        try:
            done = bool(env._check_success())
        except Exception:
            done = False
        progress = OrderedDict([
            ("grasp_lettuce",       _grasped(env, "lettuce")),
            ("water_on",            water_on),
            ("lettuce_under_water", under_water),
            ("wash_t5",             wt >= 5),
            ("wash_t10",            wt >= 10),
            ("wash_t15",            wt >= 15),
            ("wash_t20",            wt >= 20),
            ("task_complete",       done),
        ])
        return {"progress": progress, "quiescence": OrderedDict()}

    # ----- Fallback -----
    try:
        success = bool(env._check_success())
    except Exception:
        success = False
    return {"progress": OrderedDict([("success", success)]), "quiescence": OrderedDict()}
