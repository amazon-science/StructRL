"""Per-task subtask state extractor for the 16 composite_seen tasks.

Returns a structured dict with two groups:

    {
        "progress":   OrderedDict[name -> bool]   # monotone task-progress signals
        "quiescence": OrderedDict[name -> bool]   # gripper-far / end-of-task conditions
    }

PROGRESS signals are good for dense reward shaping (R_subtask_bonus): they
turn True only when the policy has actually pushed the world state forward
(object-in-receptacle, fixture-turned-on, etc.) and stay True afterwards in
nearly all cases.

QUIESCENCE signals (gripper_*_far) are NOT suitable for dense reward
because they are True at episode start (robot spawns far from objects),
flip back and forth as the gripper interacts, and only matter at the very
END of an episode (the simulator requires the gripper to retreat before
declaring success).

The full _check_success is the AND of progress AND quiescence.
"""
from collections import OrderedDict
import numpy as np
import robocasa.utils.object_utils as OU


def get_subtask_states(env) -> dict:
    """Return {'progress': OrderedDict, 'quiescence': OrderedDict} for env."""
    cls_name = type(env).__name__

    # ----- DeliverStraw -----
    if cls_name == "DeliverStraw":
        progress = OrderedDict([
            ("straw_in_glass_cup", OU.check_obj_in_receptacle(env, "straw", "glass_cup", th=0.5)),
        ])
        quiescence = OrderedDict([
            ("gripper_far", OU.gripper_obj_far(env, obj_name="straw")),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- GetToastedBread -----
    # NOTE: bread spawns IN the toaster, so `toast_in_slot` is True at t=0
    # (not a useful progress signal). The real success test reads
    # `self.toaster_on` which is a LATCHED env-side flag set the first time
    # the toaster gets turned on while the bread is in a slot. Even after
    # the policy removes the bread, `toaster_on` stays True. So progress
    # is just (toaster_on, toast_on_plate).
    if cls_name == "GetToastedBread":
        progress = OrderedDict([
            ("toaster_on",     bool(getattr(env, "toaster_on", False))),
            ("toast_on_plate", OU.check_obj_in_receptacle(env, "obj", "plate")),
        ])
        quiescence = OrderedDict([
            ("gripper_far", OU.gripper_obj_far(env, "obj")),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- KettleBoiling (note: original loops over burners with early-return;
    # we expose 2 progress quasi-independents — kettle on stove + kettle on
    # an *active* burner. AND of these two equals the loop body's accept) -----
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
            ("kettle_on_stove",         obj_on_stove),
            ("kettle_on_active_burner", any_kettle_on_active_burner),
        ])
        quiescence = OrderedDict([
            ("gripper_far", OU.gripper_obj_far(env)),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- LoadDishwasher (no quiescence in original) -----
    if cls_name == "LoadDishwasher":
        # Two dishes are placed independently. Expose them as two parallel
        # progress flags (not all()-merged) so each placement earns its own
        # event reward and is timed independently under D3 (rank0 = first
        # dish from start, rank1 = second dish from the first). Merging into
        # one flag mis-attributes the first dish's slow time to the second.
        progress = OrderedDict([
            ("dish0_on_rack",     env.dishwasher.check_rack_contact(env, "dish0")),
            ("dish1_on_rack",     env.dishwasher.check_rack_contact(env, "dish1")),
            ("dishwasher_closed", env.dishwasher.is_closed(env, th=0.05)),
        ])
        return {"progress": progress, "quiescence": OrderedDict()}

    # ----- PackIdenticalLunches -----
    if cls_name == "PackIdenticalLunches":
        veg_in_0, veg_in_1, meat_in_0, meat_in_1 = [], [], [], []
        for veg in ["vegetable0", "vegetable1"]:
            if OU.check_obj_in_receptacle(env, veg, "tupperware0"):
                veg_in_0.append(veg)
            if OU.check_obj_in_receptacle(env, veg, "tupperware1"):
                veg_in_1.append(veg)
        for meat in ["meat0", "meat1"]:
            if OU.check_obj_in_receptacle(env, meat, "tupperware0"):
                meat_in_0.append(meat)
            if OU.check_obj_in_receptacle(env, meat, "tupperware1"):
                meat_in_1.append(meat)
        # NOTE: `no_duplicates` is True at episode start (no obj in any
        # tupperware -> empty list -> 0 == 0 is True). It's an anti-cheat
        # check, not a real progress signal. Drop it from progress.
        all_objs = veg_in_0 + veg_in_1 + meat_in_0 + meat_in_1
        gripper_far = all(OU.gripper_obj_far(env, o) for o in all_objs) if all_objs else True
        progress = OrderedDict([
            ("tupper0_complete", len(veg_in_0) == 1 and len(meat_in_0) == 1),
            ("tupper1_complete", len(veg_in_1) == 1 and len(meat_in_1) == 1),
        ])
        quiescence = OrderedDict([
            ("gripper_far", gripper_far),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- PreSoakPan -----
    if cls_name == "PreSoakPan":
        h = env.sink.get_handle_state(env=env)
        progress = OrderedDict([
            ("water_on",       bool(h["water_on"])),
            ("pan_in_sink",    OU.obj_inside_of(env, "obj1", env.sink, partial_check=False)),
            ("sponge_in_sink", OU.obj_inside_of(env, "obj2", env.sink, partial_check=False)),
        ])
        quiescence = OrderedDict([
            ("gripper_far_pan",    OU.gripper_obj_far(env, "obj1")),
            ("gripper_far_sponge", OU.gripper_obj_far(env, "obj2")),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- PrepareCoffee -----
    if cls_name == "PrepareCoffee":
        progress = OrderedDict([
            ("mug_at_machine",    env.coffee_machine.check_receptacle_placement_for_pouring(env, "obj")),
            ("machine_turned_on", bool(env.coffee_machine._turned_on)),
        ])
        quiescence = OrderedDict([
            ("gripper_obj_far",    OU.gripper_obj_far(env)),
            ("gripper_button_far", env.coffee_machine.gripper_button_far(env)),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- RinseSinkBasin (no quiescence) -----
    if cls_name == "RinseSinkBasin":
        wl = getattr(env, "washed_loc", [False, False, False])
        progress = OrderedDict([
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
            ("contact_5steps",   timer >= 5),
            ("sweep_range_0p1m", sweep_range >= 0.1),
        ])
        quiescence = OrderedDict([
            ("gripper_sponge_far", OU.gripper_obj_far(env, "sponge", th=0.15)),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- SearingMeat -----
    if cls_name == "SearingMeat":
        progress = OrderedDict([
            ("pan_on_active_knob", env.stove.check_obj_location_on_stove(env, "pan", threshold=0.15) == env.knob),
            ("meat_in_pan",        OU.check_obj_in_receptacle(env, "meat", "pan", th=0.07)),
        ])
        quiescence = OrderedDict([
            ("gripper_meat_far", OU.gripper_obj_far(env, obj_name="meat")),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- SetUpCuttingStation -----
    if cls_name == "SetUpCuttingStation":
        progress = OrderedDict([
            ("meat_on_board",  OU.check_obj_in_receptacle(env, "meat", "receptacle")),
            ("knife_on_board", OU.check_obj_in_receptacle(env, "knife", "receptacle")),
        ])
        quiescence = OrderedDict([
            ("gripper_far", OU.gripper_obj_far(env, "knife") and OU.gripper_obj_far(env, "receptacle")),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- StackBowlsCabinet -----
    # 2 stages by demo behavior: (1) stack bowls on counter, (2) carry the
    # stacked pair into the cabinet. The two stages happen physically far
    # apart in time (stack on counter → walk to cabinet → place inside),
    # so timing is meaningful. Stage 1 fires when ANY bowl crosses the
    # cabinet's interior boundary (the second bowl follows ~0.2 s later
    # because they are stacked, so we don't need a 'both-in-cabinet'
    # signal during dense reward — outcome reward enforces both).
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
                OU.check_obj_in_receptacle(env, "bowl2", "bowl1")
                or OU.check_obj_in_receptacle(env, "bowl1", "bowl2")
            )
        except Exception:
            stacked = False
        progress = OrderedDict([
            ("bowls_stacked",       bool(stacked)),
            ("any_bowl_in_cabinet", bool(b1_in_cab or b2_in_cab)),
        ])
        quiescence = OrderedDict([
            ("gripper_far",
             OU.gripper_obj_far(env, "bowl1") and OU.gripper_obj_far(env, "bowl2")),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- SteamInMicrowave -----
    # Real _check_success = vegetable_in_bowl AND bowl_in_microwave AND
    # door_closed AND microwave.turned_on. Match exactly (was previously
    # written wrong as a single 'obj_in_microwave' that never fired).
    if cls_name == "SteamInMicrowave":
        try:
            veg_in_bowl = OU.check_obj_in_receptacle(env, "vegetable", "bowl")
        except Exception:
            veg_in_bowl = False
        try:
            bowl_in_micro = OU.obj_inside_of(env, "bowl", env.microwave)
        except Exception:
            bowl_in_micro = False
        try:
            microwave_closed = env.microwave.is_closed(env)
        except Exception:
            microwave_closed = False
        try:
            microwave_on = bool(env.microwave.get_state()["turned_on"])
        except Exception:
            microwave_on = False
        progress = OrderedDict([
            ("veg_in_bowl",      veg_in_bowl),
            ("bowl_in_micro",    bowl_in_micro),
            ("door_closed",      microwave_closed),
            ("microwave_on",     microwave_on),
        ])
        return {"progress": progress, "quiescence": OrderedDict()}

    # ----- StirVegetables -----
    # Real _check_success = (self.success_time >= 5), a cumulative counter
    # that ticks only when all of (pot on active knob, both vegetables in
    # pot, spatula grasped, stirring motion detected). The pot spawns
    # already on the active knob (sample_region locs=[knob]), so
    # pot_on_active_knob is True at t=0 and is NOT a useful progress
    # signal. The real intermediate milestones are: (stage0) both
    # vegetables placed in the pot (parallel), (stage1) spatula grasped,
    # (stage2) stirring completes the cumulative timer.
    if cls_name == "StirVegetables":
        try:
            veg1_in_pot = OU.check_obj_in_receptacle(env, "veg1", "pot")
        except Exception:
            veg1_in_pot = False
        try:
            veg2_in_pot = OU.check_obj_in_receptacle(env, "veg2", "pot")
        except Exception:
            veg2_in_pot = False
        try:
            spatula_grasped = OU.check_obj_grasped(env, "spatula")
        except Exception:
            spatula_grasped = False
        try:
            done = bool(env._check_success())
        except Exception:
            done = False
        progress = OrderedDict([
            ("veg1_in_pot",     veg1_in_pot),
            ("veg2_in_pot",     veg2_in_pot),
            ("spatula_grasped", spatula_grasped),
            ("task_complete",   done),
        ])
        return {"progress": progress, "quiescence": OrderedDict()}

    # ----- StoreLeftoversInBowl -----
    # Real _check_success has 4 sub-conditions:
    #   chicken_in_bowl, vegetable_in_bowl, bowl_in_fridge, gripper_far(bowl)
    if cls_name == "StoreLeftoversInBowl":
        try:
            chicken_in_bowl = OU.check_obj_in_receptacle(env, "chicken_drumstick", "bowl")
        except Exception:
            chicken_in_bowl = False
        try:
            vegetable_in_bowl = OU.check_obj_in_receptacle(env, "vegetable", "bowl")
        except Exception:
            vegetable_in_bowl = False
        try:
            bowl_in_fridge = env.fridge.check_rack_contact(env, "bowl")
        except Exception:
            bowl_in_fridge = False
        try:
            gripper_far = OU.gripper_obj_far(env, "bowl")
        except Exception:
            gripper_far = False
        progress = OrderedDict([
            ("chicken_in_bowl",   chicken_in_bowl),
            ("vegetable_in_bowl", vegetable_in_bowl),
            ("bowl_in_fridge",    bowl_in_fridge),
        ])
        quiescence = OrderedDict([
            ("gripper_far", gripper_far),
        ])
        return {"progress": progress, "quiescence": quiescence}

    # ----- WashLettuce -----
    # Real _check_success = (self.washed_time >= 25), cumulative water-on-
    # lettuce counter. Like StirVegetables, we expose one intermediate
    # progress flag — opening the sink water — as the indispensable setup
    # before the cumulative timer can begin to tick.
    if cls_name == "WashLettuce":
        try:
            water_on = bool(env.sink.get_handle_state(env=env)["water_on"])
        except Exception:
            water_on = False
        try:
            done = bool(env._check_success())
        except Exception:
            done = False
        progress = OrderedDict([
            ("water_on",      water_on),
            ("task_complete", done),
        ])
        return {"progress": progress, "quiescence": OrderedDict()}

    # ----- Fallback -----
    try:
        success = bool(env._check_success())
    except Exception:
        success = False
    return {"progress": OrderedDict([("success", success)]), "quiescence": OrderedDict()}
