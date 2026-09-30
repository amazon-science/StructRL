"""V6 micro-split subtask checker (density ablation) — assembled from
dense-v6v7-split/parts_checker/V6/*.py. Same output contract as v2:
{"progress": OrderedDict[name->bool], "quiescence": OrderedDict[name->bool]}.
Latching is downstream; aux state (env._aux_*) resets on timestep rollback in
training and is explicitly cleared between demos by the extraction harness."""
from collections import OrderedDict
import numpy as np
import robocasa.utils.object_utils as OU


# ============================ DeliverStraw ============================


def _aux_deliverstraw(env):
    a = getattr(env, "_aux_deliverstraw", None)
    t = int(getattr(env, "timestep", 0))
    if a is None or t < a["_last_t"]:
        a = {"_last_t": t, "_calls": 0}
        env._aux_deliverstraw = a
    a["_last_t"] = t
    a["_calls"] += 1
    return a


def _deliverstraw_grasped(env, name):
    try:
        return bool(OU.check_obj_grasped(env, name, threshold=0.035))
    except Exception:
        return False


def _deliverstraw_in_recep(env, o, r, **kw):
    try:
        return bool(OU.check_obj_in_receptacle(env, o, r, **kw))
    except Exception:
        return False


def _deliverstraw_contact(env, name):
    try:
        return bool(
            env.check_contact(env.robots[0].gripper["right"], env.objects[name])
        )
    except Exception:
        return False


def _deliverstraw_drawer_frac(env):
    # Fixture.get_joint_state normalized slide joint: 0=closed, 1=open.
    try:
        drawer = env.drawer
        jn = drawer.naming_prefix + "slidejoint"
        if jn not in drawer._joint_infos:
            jn = drawer.door_joint_names[0]
        return float(drawer.get_joint_state(env, [jn])[jn])
    except Exception:
        return 0.0


def check_DeliverStraw(env):
    a = _aux_deliverstraw(env)

    straw_pos = np.array(env.sim.data.body_xpos[env.obj_body_id["straw"]])
    cup_pos = np.array(env.sim.data.body_xpos[env.obj_body_id["glass_cup"]])
    eef_pos = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])

    p0_straw = a.setdefault("p0_straw", straw_pos.copy())

    frac = _deliverstraw_drawer_frac(env)
    d_eef_straw = float(np.linalg.norm(eef_pos - straw_pos))
    disp_straw = float(np.linalg.norm(straw_pos - p0_straw))
    d_xy_cup = float(np.linalg.norm(straw_pos[:2] - cup_pos[:2]))
    dz_cup = float(straw_pos[2] - cup_pos[2])

    contact = _deliverstraw_contact(env, "straw")
    grasped = _deliverstraw_grasped(env, "straw")
    in_cup = _deliverstraw_in_recep(env, "straw", "glass_cup", th=0.5)

    progress = OrderedDict([
        ("grasp_straw__s1", bool(frac >= 0.15)),
        ("grasp_straw__s2", bool(frac >= 0.35)),
        ("grasp_straw__s3", bool(d_eef_straw <= 0.15)),
        ("grasp_straw__s4", bool(contact)),
        ("grasp_straw__s5", bool(grasped)),
        ("grasp_straw__s6", bool(disp_straw >= 0.08)),
        ("straw_in_glass_cup__s1", bool(d_xy_cup < 1.50)),
        ("straw_in_glass_cup__s2", bool(d_xy_cup < 0.80)),
        ("straw_in_glass_cup__s3", bool(d_xy_cup < 0.40)),
        ("straw_in_glass_cup__s4", bool(d_xy_cup < 0.15)),
        ("straw_in_glass_cup__s5", bool(dz_cup < 0.20)),
        ("straw_in_glass_cup__s6", bool(in_cup)),
    ])
    quiescence = OrderedDict([
        ("gripper_far", OU.gripper_obj_far(env, obj_name="straw")),
    ])
    return {"progress": progress, "quiescence": quiescence}

# ============================ GetToastedBread ============================


def _aux_gettoastedbread(env):
    a = getattr(env, "_aux_gettoastedbread", None)
    t = int(getattr(env, "timestep", 0))
    if a is None or t < a["_last_t"]:
        a = {"_last_t": t, "_calls": 0}
        env._aux_gettoastedbread = a
    a["_last_t"] = t
    a["_calls"] += 1
    return a


def _gettoastedbread_grasped(env, name):
    try:
        return bool(OU.check_obj_grasped(env, name, threshold=0.035))
    except Exception:
        return False


def _gettoastedbread_in_recep(env, o, r, **kw):
    try:
        return bool(OU.check_obj_in_receptacle(env, o, r, **kw))
    except Exception:
        return False


def _gettoastedbread_contact(env, name):
    try:
        return bool(
            env.check_contact(env.robots[0].gripper["right"], env.objects[name])
        )
    except Exception:
        return False


def _gettoastedbread_slot_pair(env):
    # slot pair index in contact with the toast (env._check_success loop pattern)
    try:
        toaster = env.toaster
        for sp in range(len(toaster._slot_pairs)):
            if toaster.check_slot_contact(env, "obj", slot_pair=sp):
                return sp
    except Exception:
        pass
    return None


def _gettoastedbread_lever_state(env, sp):
    lever = 0.0
    engaged = False
    if sp is not None:
        try:
            lever = float(env.toaster._state[sp]["lever"])
        except Exception:
            lever = 0.0
        try:
            engaged = bool(env.toaster._turned_on[sp])
        except Exception:
            engaged = False
    return lever, engaged


def check_GetToastedBread(env):
    a = _aux_gettoastedbread(env)

    obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id["obj"]])
    plate_pos = np.array(env.sim.data.body_xpos[env.obj_body_id["plate"]])
    eef_pos = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])

    p0_obj = a.setdefault("p0_obj", obj_pos.copy())
    z0_obj = float(p0_obj[2])

    d_eef_obj = float(np.linalg.norm(eef_pos - obj_pos))
    d_xy_plate = float(np.linalg.norm(obj_pos[:2] - plate_pos[:2]))
    dz_plate = float(obj_pos[2] - plate_pos[2])
    obj_z = float(obj_pos[2])

    sp = _gettoastedbread_slot_pair(env)
    in_slot = sp is not None
    lever, engaged = _gettoastedbread_lever_state(env, sp)
    toaster_on = bool(getattr(env, "toaster_on", False))

    contact = _gettoastedbread_contact(env, "obj")
    grasped = _gettoastedbread_grasped(env, "obj")
    if grasped:
        a["grasp_seen"] = True
    grasp_seen = bool(a.get("grasp_seen", False))
    on_plate = _gettoastedbread_in_recep(env, "obj", "plate")

    progress = OrderedDict([
        ("toaster_on__s1", bool(d_eef_obj <= 0.40)),
        ("toaster_on__s2", bool(d_eef_obj <= 0.20)),
        ("toaster_on__s3", bool(lever >= 0.30)),
        ("toaster_on__s4", bool(lever >= 0.60)),
        ("toaster_on__s5", bool(lever >= 0.90)),
        ("toaster_on__s6", bool(engaged)),
        ("toaster_on__s7", bool(toaster_on)),
        ("grasp_toast__s1", bool(d_eef_obj <= 0.10)),
        ("grasp_toast__s2", bool(contact)),
        ("grasp_toast__s3", bool(grasped)),
        ("grasp_toast__s4", bool(grasp_seen and not in_slot)),
        ("grasp_toast__s5", bool(obj_z >= z0_obj + 0.05)),
        ("grasp_toast__s6", bool(obj_z >= z0_obj + 0.12)),
        ("toast_on_plate__s1", bool(d_xy_plate < 1.00)),
        ("toast_on_plate__s2", bool(d_xy_plate < 0.50)),
        ("toast_on_plate__s3", bool(d_xy_plate < 0.25)),
        ("toast_on_plate__s4", bool(d_xy_plate < 0.10)),
        ("toast_on_plate__s5", bool(dz_plate < 0.10)),
        ("toast_on_plate__s6", bool(on_plate)),
    ])
    quiescence = OrderedDict([
        ("gripper_far", OU.gripper_obj_far(env, "obj")),
    ])
    return {"progress": progress, "quiescence": quiescence}

# ============================ KettleBoiling ============================


def _aux_kettleboiling(env):
    a = getattr(env, "_aux_kettleboiling", None)
    t = int(getattr(env, "timestep", 0))
    if a is None or t < a["_last_t"]:
        a = {"_last_t": t, "_calls": 0}
        env._aux_kettleboiling = a
    a["_last_t"] = t
    a["_calls"] += 1
    return a


def _kettleboiling_grasped(env, name):
    try:
        return bool(OU.check_obj_grasped(env, name, threshold=0.035))
    except Exception:
        return False


def _kettleboiling_contact(env, name):
    try:
        return bool(
            env.check_contact(env.robots[0].gripper["right"], env.objects[name])
        )
    except Exception:
        return False


def _kettleboiling_on_stove(env):
    try:
        return bool(OU.check_obj_fixture_contact(env, "obj", env.stove))
    except Exception:
        return False


def _kettleboiling_burner_info(env, kettle_xy):
    # nearest burner site: (xy dist, location, site z); v2 checker loop pattern
    d_min = float("inf")
    loc_min = None
    site_z = None
    try:
        for location, site in env.stove.burner_sites.items():
            if site is None:
                continue
            sp = np.array(env.sim.data.get_site_xpos(site.get("name")))
            d = float(np.linalg.norm(sp[0:2] - kettle_xy))
            if d < d_min:
                d_min = d
                loc_min = location
                site_z = float(sp[2])
    except Exception:
        pass
    return d_min, loc_min, site_z


def _kettleboiling_knob_dist(env, loc, eef_pos):
    if loc is None:
        return float("inf")
    try:
        jid = env.sim.model.joint_name2id(
            "{}knob_{}_joint".format(env.stove.naming_prefix, loc)
        )
        knob_pos = np.array(env.sim.data.body_xpos[env.sim.model.jnt_bodyid[jid]])
        return float(np.linalg.norm(eef_pos - knob_pos))
    except Exception:
        return float("inf")


def _kettleboiling_knob_delta(env, loc):
    # (knob displacement delta = min(q, 2pi-q), burner_on) for burner loc
    try:
        knobs_state = env.stove.get_knobs_state(env=env)
        if loc is None or loc not in knobs_state:
            return 0.0, False
        q = float(knobs_state[loc])
        delta = float(min(q, 2.0 * np.pi - q))
        burner_on = bool(env.stove.is_burner_on(env=env, burner_loc=loc))
        return delta, burner_on
    except Exception:
        return 0.0, False


def check_KettleBoiling(env):
    a = _aux_kettleboiling(env)

    kettle_pos = np.array(env.sim.data.body_xpos[env.obj_body_id["obj"]])
    eef_pos = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])

    p0_obj = a.setdefault("p0_obj", kettle_pos.copy())
    z0_obj = float(p0_obj[2])
    kettle_z = float(kettle_pos[2])

    d_eef = float(np.linalg.norm(eef_pos - kettle_pos))
    d_burner, loc_star, site_z = _kettleboiling_burner_info(env, kettle_pos[0:2])
    dz_site = (kettle_z - site_z) if site_z is not None else float("inf")

    contact = _kettleboiling_contact(env, "obj")
    grasped = _kettleboiling_grasped(env, "obj")
    on_stove = _kettleboiling_on_stove(env)
    d_knob = _kettleboiling_knob_dist(env, loc_star, eef_pos)
    delta, burner_on = _kettleboiling_knob_delta(env, loc_star)

    progress = OrderedDict([
        ("grasp_kettle__s1", bool(d_eef <= 0.30)),
        ("grasp_kettle__s2", bool(d_eef <= 0.15)),
        ("grasp_kettle__s3", bool(contact)),
        ("grasp_kettle__s4", bool(grasped)),
        ("grasp_kettle__s5", bool(kettle_z >= z0_obj + 0.03)),
        ("grasp_kettle__s6", bool(kettle_z >= z0_obj + 0.08)),
        ("kettle_on_stove__s1", bool(d_burner < 0.60)),
        ("kettle_on_stove__s2", bool(d_burner < 0.35)),
        ("kettle_on_stove__s3", bool(d_burner < 0.20)),
        ("kettle_on_stove__s4", bool(dz_site < 0.15)),
        ("kettle_on_stove__s5", bool(dz_site < 0.06)),
        ("kettle_on_stove__s6", bool(on_stove)),
        ("kettle_on_active_burner__s1", bool(d_burner < 0.15)),
        ("kettle_on_active_burner__s2", bool(d_knob < 0.35)),
        ("kettle_on_active_burner__s3", bool(d_knob < 0.15)),
        ("kettle_on_active_burner__s4", bool(delta >= 0.10)),
        ("kettle_on_active_burner__s5", bool(delta >= 0.22)),
        ("kettle_on_active_burner__s6", bool(on_stove and d_burner < 0.15 and burner_on)),
    ])
    quiescence = OrderedDict([
        ("gripper_far", OU.gripper_obj_far(env)),
    ])
    return {"progress": progress, "quiescence": quiescence}

# ============================ LoadDishwasher ============================


def _aux_loaddishwasher(env):
    a = getattr(env, "_aux_loaddishwasher", None)
    t = int(getattr(env, "timestep", 0))
    if a is None or t < a["_last_t"]:
        a = {"_last_t": t, "_calls": 0}
        env._aux_loaddishwasher = a
    a["_last_t"] = t
    a["_calls"] += 1
    return a


def _loaddishwasher_grasped(env, name):
    try:
        return bool(OU.check_obj_grasped(env, name, threshold=0.035))
    except Exception:
        return False


def _loaddishwasher_contact(env, name):
    try:
        return bool(
            env.check_contact(env.robots[0].gripper["right"], env.objects[name])
        )
    except Exception:
        return False


def _loaddishwasher_rack_contact(env, name):
    try:
        return bool(env.dishwasher.check_rack_contact(env, name))
    except Exception:
        return False


def _loaddishwasher_rack_pos(env):
    try:
        jid = env.sim.model.joint_name2id(env.dishwasher._joint_names["rack"])
        return np.array(env.sim.data.body_xpos[env.sim.model.jnt_bodyid[jid]])
    except Exception:
        return None


def _loaddishwasher_rack_frac(env):
    # normalized rack slide joint: 0=slid in, 1=slid out (reset ~1.0)
    try:
        jn = env.dishwasher._joint_names["rack"]
        return float(env.dishwasher.get_joint_state(env, [jn])[jn])
    except Exception:
        return 1.0


def _loaddishwasher_door_frac(env):
    # normalized door joint: 0=closed, 1=open; max over door joints so that
    # (door_frac <= th) is exactly env.dishwasher.is_closed(env, th=th)
    try:
        jns = env.dishwasher.door_joint_names
        st = env.dishwasher.get_joint_state(env, jns)
        return max(float(st[j]) for j in jns)
    except Exception:
        return 1.0


def check_LoadDishwasher(env):
    a = _aux_loaddishwasher(env)

    eef_pos = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])
    dish0_pos = np.array(env.sim.data.body_xpos[env.obj_body_id["dish0"]])
    dish1_pos = np.array(env.sim.data.body_xpos[env.obj_body_id["dish1"]])

    p0_dish0 = a.setdefault("p0_dish0", dish0_pos.copy())
    p0_dish1 = a.setdefault("p0_dish1", dish1_pos.copy())
    z0_dish0 = float(p0_dish0[2])
    z0_dish1 = float(p0_dish1[2])

    d_eef_d0 = float(np.linalg.norm(eef_pos - dish0_pos))
    d_eef_d1 = float(np.linalg.norm(eef_pos - dish1_pos))

    rack_pos = _loaddishwasher_rack_pos(env)
    if rack_pos is not None:
        d_d0_rack = float(np.linalg.norm(dish0_pos - rack_pos))
        d_d1_rack = float(np.linalg.norm(dish1_pos - rack_pos))
        dxy_d0_rack = float(np.linalg.norm(dish0_pos[:2] - rack_pos[:2]))
        dxy_d1_rack = float(np.linalg.norm(dish1_pos[:2] - rack_pos[:2]))
        dz_d0_rack = float(dish0_pos[2] - rack_pos[2])
        dz_d1_rack = float(dish1_pos[2] - rack_pos[2])
    else:
        d_d0_rack = d_d1_rack = float("inf")
        dxy_d0_rack = dxy_d1_rack = float("inf")
        dz_d0_rack = dz_d1_rack = float("inf")

    contact0 = _loaddishwasher_contact(env, "dish0")
    contact1 = _loaddishwasher_contact(env, "dish1")
    grasped0 = _loaddishwasher_grasped(env, "dish0")
    grasped1 = _loaddishwasher_grasped(env, "dish1")
    rack0 = _loaddishwasher_rack_contact(env, "dish0")
    rack1 = _loaddishwasher_rack_contact(env, "dish1")

    rack_frac = _loaddishwasher_rack_frac(env)
    door_frac = _loaddishwasher_door_frac(env)

    progress = OrderedDict([
        ("grasp_dish0__s1", bool(d_eef_d0 <= 0.30)),
        ("grasp_dish0__s2", bool(d_eef_d0 <= 0.15)),
        ("grasp_dish0__s3", bool(contact0)),
        ("grasp_dish0__s4", bool(grasped0)),
        ("grasp_dish0__s5", bool(float(dish0_pos[2]) >= z0_dish0 + 0.03)),
        ("grasp_dish0__s6", bool(float(dish0_pos[2]) >= z0_dish0 + 0.08)),
        ("dish0_on_rack__s1", bool(d_d0_rack < 0.80)),
        ("dish0_on_rack__s2", bool(d_d0_rack < 0.50)),
        ("dish0_on_rack__s3", bool(d_d0_rack < 0.30)),
        ("dish0_on_rack__s4", bool(dxy_d0_rack < 0.15)),
        ("dish0_on_rack__s5", bool(dz_d0_rack < 0.10)),
        ("dish0_on_rack__s6", bool(rack0)),
        ("grasp_dish1__s1", bool(d_eef_d1 <= 0.30)),
        ("grasp_dish1__s2", bool(d_eef_d1 <= 0.15)),
        ("grasp_dish1__s3", bool(contact1)),
        ("grasp_dish1__s4", bool(grasped1)),
        ("grasp_dish1__s5", bool(float(dish1_pos[2]) >= z0_dish1 + 0.03)),
        ("grasp_dish1__s6", bool(float(dish1_pos[2]) >= z0_dish1 + 0.08)),
        ("dish1_on_rack__s1", bool(d_d1_rack < 0.80)),
        ("dish1_on_rack__s2", bool(d_d1_rack < 0.50)),
        ("dish1_on_rack__s3", bool(d_d1_rack < 0.30)),
        ("dish1_on_rack__s4", bool(dxy_d1_rack < 0.15)),
        ("dish1_on_rack__s5", bool(dz_d1_rack < 0.10)),
        ("dish1_on_rack__s6", bool(rack1)),
        ("door_half_closed__s1", bool(rack_frac <= 0.70)),
        ("door_half_closed__s2", bool(rack_frac <= 0.40)),
        ("door_half_closed__s3", bool(rack_frac <= 0.15)),
        ("door_half_closed__s4", bool(door_frac <= 0.80)),
        ("door_half_closed__s5", bool(door_frac <= 0.55)),
        ("door_half_closed__s6", bool(door_frac <= 0.3)),
        ("dishwasher_closed__s1", bool(door_frac <= 0.25)),
        ("dishwasher_closed__s2", bool(door_frac <= 0.20)),
        ("dishwasher_closed__s3", bool(door_frac <= 0.15)),
        ("dishwasher_closed__s4", bool(door_frac <= 0.10)),
        ("dishwasher_closed__s5", bool(door_frac <= 0.05)),
    ])
    quiescence = OrderedDict()
    return {"progress": progress, "quiescence": quiescence}

# ============================ PackIdenticalLunches ============================


def _packidenticallunches_grasped(env, name, threshold=0.035):
    try:
        return bool(OU.check_obj_grasped(env, name, threshold=threshold))
    except Exception:
        return False


def _packidenticallunches_in_recep(env, o, r, **kw):
    try:
        return bool(OU.check_obj_in_receptacle(env, o, r, **kw))
    except Exception:
        return False


def _packidenticallunches_obj_pos(env, name):
    try:
        return np.array(env.sim.data.body_xpos[env.obj_body_id[name]])
    except Exception:
        return None


def _packidenticallunches_gripper_contact(env, name):
    try:
        return bool(env.check_contact(env.robots[0].gripper["right"], env.objects[name]))
    except Exception:
        return False


def _packidenticallunches_obj_contact(env, a, b):
    try:
        return bool(env.check_contact(env.objects[a], env.objects[b]))
    except Exception:
        return False


def _aux_packidenticallunches(env):
    a = getattr(env, "_aux_packidenticallunches", None)
    t = int(getattr(env, "timestep", 0))
    if a is None or t < a["_last_t"]:
        a = {"_last_t": t, "_calls": 0}
        env._aux_packidenticallunches = a
    a["_last_t"] = t
    a["_calls"] += 1
    return a


def check_PackIdenticalLunches(env):
    a = _aux_packidenticallunches(env)

    try:
        eef = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])
    except Exception:
        eef = None
    try:
        f1 = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint1")])
        f2 = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint2")])
    except Exception:
        f1 = f2 = None
    fingers_050 = (f1 is not None) and (f1 < 0.050) and (f2 < 0.050)

    tup_pos = {
        "tupperware0": _packidenticallunches_obj_pos(env, "tupperware0"),
        "tupperware1": _packidenticallunches_obj_pos(env, "tupperware1"),
    }

    progress = OrderedDict()
    veg_in_0, veg_in_1, meat_in_0, meat_in_1 = [], [], [], []

    for name in ("vegetable0", "vegetable1", "meat0", "meat1"):
        obj_pos = _packidenticallunches_obj_pos(env, name)
        if obj_pos is not None:
            a.setdefault("z0_" + name, float(obj_pos[2]))
        z0 = a.get("z0_" + name, None)

        if eef is not None and obj_pos is not None:
            d = float(np.linalg.norm(eef - obj_pos))
        else:
            d = float("inf")

        contact = _packidenticallunches_gripper_contact(env, name)
        grasped = _packidenticallunches_grasped(env, name, threshold=0.035)

        dxy = {}
        for r in ("tupperware0", "tupperware1"):
            rp = tup_pos[r]
            if obj_pos is not None and rp is not None:
                dxy[r] = float(np.linalg.norm(obj_pos[0:2] - rp[0:2]))
            else:
                dxy[r] = float("inf")
        r_min = "tupperware0" if dxy["tupperware0"] <= dxy["tupperware1"] else "tupperware1"
        dmin = dxy[r_min]
        if obj_pos is not None and tup_pos[r_min] is not None:
            dz_min = abs(float(obj_pos[2]) - float(tup_pos[r_min][2]))
        else:
            dz_min = float("inf")

        tup_contact = (
            _packidenticallunches_obj_contact(env, name, "tupperware0")
            or _packidenticallunches_obj_contact(env, name, "tupperware1")
        )
        in0 = _packidenticallunches_in_recep(env, name, "tupperware0")
        in1 = _packidenticallunches_in_recep(env, name, "tupperware1")
        packed = in0 or in1
        if name.startswith("vegetable"):
            if in0:
                veg_in_0.append(name)
            if in1:
                veg_in_1.append(name)
        else:
            if in0:
                meat_in_0.append(name)
            if in1:
                meat_in_1.append(name)

        lifted = (
            grasped
            and obj_pos is not None
            and z0 is not None
            and float(obj_pos[2]) > z0 + 0.04
        )

        g = "grasp_" + name
        p = name + "_packed"
        progress[g + "__s1"] = d < 0.3
        progress[g + "__s2"] = d < 0.15
        progress[g + "__s3"] = d < 0.08
        progress[g + "__s4"] = contact
        progress[g + "__s5"] = contact and fingers_050
        progress[g + "__s6"] = grasped
        progress[p + "__s1"] = lifted
        progress[p + "__s2"] = dmin < 0.35
        progress[p + "__s3"] = dmin < 0.12
        progress[p + "__s4"] = dmin < 0.12 and dz_min < 0.08
        progress[p + "__s5"] = tup_contact
        progress[p + "__s6"] = packed

    all_objs = veg_in_0 + veg_in_1 + meat_in_0 + meat_in_1
    gripper_far = all(OU.gripper_obj_far(env, o) for o in all_objs) if all_objs else True
    quiescence = OrderedDict([
        ("gripper_far", gripper_far),
    ])
    return {"progress": progress, "quiescence": quiescence}

# ============================ PreSoakPan ============================


def _presoakpan_grasped(env, name, threshold=0.035):
    try:
        return bool(OU.check_obj_grasped(env, name, threshold=threshold))
    except Exception:
        return False


def _presoakpan_obj_pos(env, name):
    try:
        return np.array(env.sim.data.body_xpos[env.obj_body_id[name]])
    except Exception:
        return None


def _presoakpan_gripper_contact(env, name):
    try:
        return bool(env.check_contact(env.robots[0].gripper["right"], env.objects[name]))
    except Exception:
        return False


def _presoakpan_inside_sink(env, name, partial_check=False, **kw):
    try:
        return bool(OU.obj_inside_of(env, name, env.sink, partial_check=partial_check, **kw))
    except Exception:
        return False


def _presoakpan_sink_contact(env, name):
    try:
        return bool(OU.check_obj_fixture_contact(env, name, env.sink))
    except Exception:
        return False


def _presoakpan_handle_contact(env):
    try:
        prefix = env.sink.naming_prefix + "handle"
        geoms = [g for g in env.sim.model.geom_names if g and g.startswith(prefix)]
        if not geoms:
            return False
        return bool(env.check_contact(env.robots[0].gripper["right"], geoms))
    except Exception:
        return False


def _aux_presoakpan(env):
    a = getattr(env, "_aux_presoakpan", None)
    t = int(getattr(env, "timestep", 0))
    if a is None or t < a["_last_t"]:
        a = {"_last_t": t, "_calls": 0}
        env._aux_presoakpan = a
    a["_last_t"] = t
    a["_calls"] += 1
    return a


def check_PreSoakPan(env):
    a = _aux_presoakpan(env)

    try:
        eef = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])
    except Exception:
        eef = None
    try:
        f1 = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint1")])
        f2 = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint2")])
    except Exception:
        f1 = f2 = None
    fingers_050 = (f1 is not None) and (f1 < 0.050) and (f2 < 0.050)

    try:
        sink_xy = np.array(env.sim.data.get_body_xpos(env.sink.name))[0:2]
    except Exception:
        sink_xy = None
    try:
        h = env.sink.get_handle_state(env=env)
    except Exception:
        h = None
    water_on = bool(h["water_on"]) if h is not None else False
    handle_q = float(h["handle_joint"]) if h is not None else None
    if water_on:
        a["settle_water"] = a.get("settle_water", 0) + 1
    try:
        handle_anchor = np.array(
            env.sim.data.xanchor[env.sim.model.joint_name2id(env.sink.naming_prefix + "handle_joint")]
        )
    except Exception:
        handle_anchor = None
    if eef is not None and handle_anchor is not None:
        d_handle = float(np.linalg.norm(eef - handle_anchor))
    else:
        d_handle = float("inf")
    handle_contact = _presoakpan_handle_contact(env)

    progress = OrderedDict()

    for name, g, p in (("obj1", "grasp_pan", "pan_in_sink"), ("obj2", "grasp_sponge", "sponge_in_sink")):
        obj_pos = _presoakpan_obj_pos(env, name)
        if obj_pos is not None:
            a.setdefault("z0_" + name, float(obj_pos[2]))
        z0 = a.get("z0_" + name, None)

        if eef is not None and obj_pos is not None:
            d = float(np.linalg.norm(eef - obj_pos))
        else:
            d = float("inf")

        contact = _presoakpan_gripper_contact(env, name)
        grasped = _presoakpan_grasped(env, name, threshold=0.035)

        if obj_pos is not None and sink_xy is not None:
            d_sink = float(np.linalg.norm(obj_pos[0:2] - sink_xy))
        else:
            d_sink = float("inf")

        lifted = (
            grasped
            and obj_pos is not None
            and z0 is not None
            and float(obj_pos[2]) > z0 + 0.04
        )

        progress[g + "__s1"] = d < 0.3
        progress[g + "__s2"] = d < 0.15
        progress[g + "__s3"] = d < 0.08
        progress[g + "__s4"] = contact
        progress[g + "__s5"] = contact and fingers_050
        progress[g + "__s6"] = grasped
        progress[p + "__s1"] = lifted
        progress[p + "__s2"] = d_sink < 0.4
        progress[p + "__s3"] = d_sink < 0.2
        progress[p + "__s4"] = _presoakpan_inside_sink(env, name, partial_check=True)
        progress[p + "__s5"] = _presoakpan_sink_contact(env, name)
        progress[p + "__s6"] = _presoakpan_inside_sink(env, name, partial_check=False)

    progress["water_on__s1"] = d_handle < 0.3
    progress["water_on__s2"] = d_handle < 0.12
    progress["water_on__s3"] = handle_contact
    progress["water_on__s4"] = handle_q is not None and handle_q >= 0.2
    progress["water_on__s5"] = water_on
    progress["water_on__s6"] = a.get("settle_water", 0) >= 2

    quiescence = OrderedDict([
        ("gripper_far_pan",    OU.gripper_obj_far(env, "obj1")),
        ("gripper_far_sponge", OU.gripper_obj_far(env, "obj2")),
    ])
    return {"progress": progress, "quiescence": quiescence}

# ============================ PrepareCoffee ============================


def _preparecoffee_grasped(env, name, threshold=0.035):
    try:
        return bool(OU.check_obj_grasped(env, name, threshold=threshold))
    except Exception:
        return False


def _preparecoffee_obj_pos(env, name):
    try:
        return np.array(env.sim.data.body_xpos[env.obj_body_id[name]])
    except Exception:
        return None


def _preparecoffee_gripper_contact(env, name):
    try:
        return bool(env.check_contact(env.robots[0].gripper["right"], env.objects[name]))
    except Exception:
        return False


def _preparecoffee_place_site_pos(env):
    try:
        site_name = env.coffee_machine.naming_prefix + "receptacle_place_site"
        return np.array(env.sim.data.site_xpos[env.sim.model.site_name2id(site_name)])
    except Exception:
        return None


def _preparecoffee_pour_placement(env, name):
    try:
        return bool(env.coffee_machine.check_receptacle_placement_for_pouring(env, name))
    except Exception:
        return False


def _preparecoffee_mug_far(env, th):
    try:
        return bool(OU.gripper_obj_far(env, obj_name="obj", th=th))
    except Exception:
        return False


def _preparecoffee_button_near(env, th):
    try:
        return not bool(env.coffee_machine.gripper_button_far(env, th=th))
    except Exception:
        return False


def _preparecoffee_button_contact(env):
    try:
        cm = env.coffee_machine
        return bool(any(
            env.check_contact(env.robots[0].gripper["right"], cm.naming_prefix + button_name)
            for button_name in cm._start_button_names
        ))
    except Exception:
        return False


def _aux_preparecoffee(env):
    a = getattr(env, "_aux_preparecoffee", None)
    t = int(getattr(env, "timestep", 0))
    if a is None or t < a["_last_t"]:
        a = {"_last_t": t, "_calls": 0}
        env._aux_preparecoffee = a
    a["_last_t"] = t
    a["_calls"] += 1
    return a


def check_PrepareCoffee(env):
    a = _aux_preparecoffee(env)

    try:
        eef = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])
    except Exception:
        eef = None
    try:
        f1 = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint1")])
        f2 = float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint2")])
    except Exception:
        f1 = f2 = None
    fingers_050 = (f1 is not None) and (f1 < 0.050) and (f2 < 0.050)
    fingers_open = (f1 is not None) and (f1 > 0.045) and (f2 > 0.045)

    obj_pos = _preparecoffee_obj_pos(env, "obj")
    if obj_pos is not None:
        a.setdefault("z0_obj", float(obj_pos[2]))
    z0 = a.get("z0_obj", None)
    obj_z = float(obj_pos[2]) if obj_pos is not None else None

    if eef is not None and obj_pos is not None:
        d = float(np.linalg.norm(eef - obj_pos))
    else:
        d = float("inf")

    contact = _preparecoffee_gripper_contact(env, "obj")
    grasped = _preparecoffee_grasped(env, "obj", threshold=0.035)

    place_pos = _preparecoffee_place_site_pos(env)
    if obj_pos is not None and place_pos is not None:
        d_place_xy = float(np.linalg.norm(obj_pos[0:2] - place_pos[0:2]))
        dz_place = abs(float(obj_pos[2]) - float(place_pos[2]))
    else:
        d_place_xy = float("inf")
        dz_place = float("inf")

    at_machine = _preparecoffee_pour_placement(env, "obj")
    try:
        turned_on = bool(env.coffee_machine._turned_on)
    except Exception:
        turned_on = False
    if turned_on:
        a["settle_on"] = a.get("settle_on", 0) + 1

    lifted = grasped and obj_z is not None and z0 is not None and obj_z > z0 + 0.04

    progress = OrderedDict()
    progress["grasp_mug__s1"] = d < 0.3
    progress["grasp_mug__s2"] = d < 0.15
    progress["grasp_mug__s3"] = d < 0.08
    progress["grasp_mug__s4"] = contact
    progress["grasp_mug__s5"] = contact and fingers_050
    progress["grasp_mug__s6"] = grasped
    progress["mug_at_machine__s1"] = lifted
    progress["mug_at_machine__s2"] = d_place_xy < 0.4
    progress["mug_at_machine__s3"] = d_place_xy < 0.15
    progress["mug_at_machine__s4"] = d_place_xy < 0.08
    progress["mug_at_machine__s5"] = d_place_xy < 0.08 and dz_place < 0.10
    progress["mug_at_machine__s6"] = at_machine
    progress["machine_turned_on__s1"] = fingers_open
    progress["machine_turned_on__s2"] = _preparecoffee_mug_far(env, th=0.10)
    progress["machine_turned_on__s3"] = _preparecoffee_button_near(env, th=0.2)
    progress["machine_turned_on__s4"] = _preparecoffee_button_near(env, th=0.08)
    progress["machine_turned_on__s5"] = _preparecoffee_button_contact(env)
    progress["machine_turned_on__s6"] = turned_on
    progress["machine_turned_on__s7"] = a.get("settle_on", 0) >= 2

    quiescence = OrderedDict([
        ("gripper_obj_far",    OU.gripper_obj_far(env)),
        ("gripper_button_far", env.coffee_machine.gripper_button_far(env)),
    ])
    return {"progress": progress, "quiescence": quiescence}

# ============================ RinseSinkBasin ============================


def _rinsesinkbasin_handle_contact(env):
    try:
        prefix = env.sink.naming_prefix + "handle"
        geoms = [g for g in env.sim.model.geom_names if g and g.startswith(prefix)]
        if not geoms:
            return False
        return bool(env.check_contact(env.robots[0].gripper["right"], geoms))
    except Exception:
        return False


def _rinsesinkbasin_circ(a, b):
    two_pi = 2.0 * np.pi
    d = abs(a - b) % two_pi
    return min(d, two_pi - d)


def _rinsesinkbasin_circ_dist(q, lo, hi, wrap=False):
    # wrap=False: region = [lo, hi]; wrap=True: region = [0, lo) U (hi, 2pi)
    if q is None:
        return float("inf")
    if wrap:
        if q < lo or q > hi:
            return 0.0
    else:
        if lo <= q <= hi:
            return 0.0
    return min(_rinsesinkbasin_circ(q, lo), _rinsesinkbasin_circ(q, hi))


def _aux_rinsesinkbasin(env):
    a = getattr(env, "_aux_rinsesinkbasin", None)
    t = int(getattr(env, "timestep", 0))
    if a is None or t < a["_last_t"]:
        a = {"_last_t": t, "_calls": 0}
        env._aux_rinsesinkbasin = a
    a["_last_t"] = t
    a["_calls"] += 1
    return a


def check_RinseSinkBasin(env):
    a = _aux_rinsesinkbasin(env)

    try:
        eef = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])
    except Exception:
        eef = None
    try:
        h = env.sink.get_handle_state(env=env)
    except Exception:
        h = None
    water_on = bool(h["water_on"]) if h is not None else False
    handle_q = float(h["handle_joint"]) if h is not None else None
    spout_q = float(h["spout_joint"]) if h is not None else None
    spout_ori = str(h["spout_ori"]) if h is not None else None
    if water_on:
        a["settle_water"] = a.get("settle_water", 0) + 1
    try:
        handle_anchor = np.array(
            env.sim.data.xanchor[env.sim.model.joint_name2id(env.sink.naming_prefix + "handle_joint")]
        )
    except Exception:
        handle_anchor = None
    if eef is not None and handle_anchor is not None:
        d_handle = float(np.linalg.norm(eef - handle_anchor))
    else:
        d_handle = float("inf")
    handle_contact = _rinsesinkbasin_handle_contact(env)

    wl = getattr(env, "washed_loc", [False, False, False])

    pi = np.pi
    cd_left = _rinsesinkbasin_circ_dist(spout_q, pi, 11.0 * pi / 6.0, wrap=False)
    cd_center = _rinsesinkbasin_circ_dist(spout_q, pi / 6.0, 11.0 * pi / 6.0, wrap=True)
    cd_right = _rinsesinkbasin_circ_dist(spout_q, pi / 6.0, pi, wrap=False)

    progress = OrderedDict()
    progress["water_on__s1"] = d_handle < 0.3
    progress["water_on__s2"] = d_handle < 0.12
    progress["water_on__s3"] = handle_contact
    progress["water_on__s4"] = handle_q is not None and handle_q >= 0.2
    progress["water_on__s5"] = water_on
    progress["water_on__s6"] = a.get("settle_water", 0) >= 2

    for loc, cd, ori, idx in (
        ("left", cd_left, "left", 0),
        ("center", cd_center, "center", 1),
        ("right", cd_right, "right", 2),
    ):
        p = "washed_" + loc
        progress[p + "__s1"] = cd < 1.5
        progress[p + "__s2"] = cd < 1.0
        progress[p + "__s3"] = cd < 0.6
        progress[p + "__s4"] = cd < 0.3
        progress[p + "__s5"] = spout_ori == ori
        progress[p + "__s6"] = bool(wl[idx])

    quiescence = OrderedDict()
    return {"progress": progress, "quiescence": quiescence}

# ============================ ScrubCuttingBoard ============================


def _aux_scrubcuttingboard(env):
    a = getattr(env, "_aux_scrubcuttingboard", None)
    t = int(getattr(env, "timestep", 0))
    if a is None or t < a["_last_t"]:
        a = {"_last_t": t, "_calls": 0}
        env._aux_scrubcuttingboard = a
    a["_last_t"] = t
    a["_calls"] += 1
    return a


def _scrubcuttingboard_grasped(env, name):
    try:
        return bool(OU.check_obj_grasped(env, name))
    except Exception:
        return False


def _scrubcuttingboard_eef_pos(env):
    try:
        return np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])
    except Exception:
        return None


def _scrubcuttingboard_obj_pos(env, name):
    try:
        return np.array(env.sim.data.body_xpos[env.obj_body_id[name]])
    except Exception:
        return None


def _scrubcuttingboard_fingers(env):
    try:
        q1 = float(env.sim.data.qpos[
            env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint1")])
        q2 = float(env.sim.data.qpos[
            env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint2")])
        return q1, q2
    except Exception:
        return None


def _scrubcuttingboard_gripper_contact(env, name):
    try:
        return bool(env.check_contact(env.robots[0].gripper["right"], env.objects[name]))
    except Exception:
        return False


def _scrubcuttingboard_obj_contact(env, a, b):
    try:
        return bool(env.check_contact(env.objects[a], env.objects[b]))
    except Exception:
        return False


def check_ScrubCuttingBoard(env):
    aux = _aux_scrubcuttingboard(env)

    eef = _scrubcuttingboard_eef_pos(env)
    sponge = _scrubcuttingboard_obj_pos(env, "sponge")
    board = _scrubcuttingboard_obj_pos(env, "cutting_board")

    if eef is not None and sponge is not None:
        d_eef_sponge = float(np.linalg.norm(eef - sponge))
    else:
        d_eef_sponge = float("inf")

    if sponge is not None and board is not None:
        d_sponge_board_xy = float(np.linalg.norm(sponge[:2] - board[:2]))
    else:
        d_sponge_board_xy = float("inf")

    if sponge is not None:
        z0_sponge = aux.setdefault("z0_sponge", float(sponge[2]))
        lift = float(sponge[2]) - z0_sponge
    else:
        lift = 0.0

    fingers = _scrubcuttingboard_fingers(env)
    closed_038 = fingers is not None and fingers[0] < 0.038 and fingers[1] < 0.038
    grip_contact = _scrubcuttingboard_gripper_contact(env, "sponge")
    board_contact = _scrubcuttingboard_obj_contact(env, "sponge", "cutting_board")
    grasped = _scrubcuttingboard_grasped(env, "sponge")

    timer = int(getattr(env, "board_contact_timer", 0))
    sweep = 0.0
    axis_max = 0.0
    if getattr(env, "board_contact_positions", None):
        try:
            positions = np.array(env.board_contact_positions)
            extents = positions.max(axis=0) - positions.min(axis=0)
            sweep = float(np.linalg.norm(extents))
            axis_max = float(np.max(extents))
        except Exception:
            pass

    progress = OrderedDict([
        ("grasp_sponge__s1", d_eef_sponge < 0.30),
        ("grasp_sponge__s2", d_eef_sponge < 0.15),
        ("grasp_sponge__s3", d_eef_sponge < 0.08),
        ("grasp_sponge__s4", grip_contact),
        ("grasp_sponge__s5", closed_038),
        ("grasp_sponge__s6", grasped or timer >= 1),
        ("contact_1step__s1", lift >= 0.02),
        ("contact_1step__s2", d_sponge_board_xy < 0.30),
        ("contact_1step__s3", d_sponge_board_xy < 0.15),
        ("contact_1step__s4", d_sponge_board_xy < 0.08),
        ("contact_1step__s5", board_contact),
        ("contact_1step__s6", timer >= 1),
        ("contact_3steps__s1", timer >= 2),
        ("contact_3steps__s2", sweep >= 0.022),
        ("contact_3steps__s3", sweep >= 0.026),
        ("contact_3steps__s4", sweep >= 0.030),
        ("contact_3steps__s5", timer >= 3),
        ("sweep_range_0p05m__s1", sweep >= 0.035),
        ("sweep_range_0p05m__s2", sweep >= 0.039),
        ("sweep_range_0p05m__s3", sweep >= 0.043),
        ("sweep_range_0p05m__s4", sweep >= 0.047),
        ("sweep_range_0p05m__s5", sweep >= 0.05),
        ("contact_5steps__s1", timer >= 4),
        ("contact_5steps__s2", axis_max >= 0.035),
        ("contact_5steps__s3", axis_max >= 0.045),
        ("contact_5steps__s4", axis_max >= 0.055),
        ("contact_5steps__s5", timer >= 5),
        ("sweep_range_0p1m__s1", sweep >= 0.06),
        ("sweep_range_0p1m__s2", sweep >= 0.07),
        ("sweep_range_0p1m__s3", sweep >= 0.08),
        ("sweep_range_0p1m__s4", sweep >= 0.09),
        ("sweep_range_0p1m__s5", sweep >= 0.10),
    ])
    quiescence = OrderedDict([
        ("gripper_sponge_far", OU.gripper_obj_far(env, "sponge", th=0.15)),
    ])
    return {"progress": progress, "quiescence": quiescence}

# ============================ SearingMeat ============================


def _aux_searingmeat(env):
    a = getattr(env, "_aux_searingmeat", None)
    t = int(getattr(env, "timestep", 0))
    if a is None or t < a["_last_t"]:
        a = {"_last_t": t, "_calls": 0}
        env._aux_searingmeat = a
    a["_last_t"] = t
    a["_calls"] += 1
    return a


def _searingmeat_grasped(env, name):
    try:
        return bool(OU.check_obj_grasped(env, name))
    except Exception:
        return False


def _searingmeat_in_recep(env, o, r, **kw):
    try:
        return bool(OU.check_obj_in_receptacle(env, o, r, **kw))
    except Exception:
        return False


def _searingmeat_eef_pos(env):
    try:
        return np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])
    except Exception:
        return None


def _searingmeat_obj_pos(env, name):
    try:
        return np.array(env.sim.data.body_xpos[env.obj_body_id[name]])
    except Exception:
        return None


def _searingmeat_burner_pos(env):
    try:
        site = env.stove.burner_sites[env.knob]
        return np.array(env.sim.data.get_site_xpos(site.get("name")))
    except Exception:
        return None


def _searingmeat_knob_angle(env):
    try:
        return float(abs(env.stove.get_knobs_state(env=env)[env.knob]))
    except Exception:
        return 0.0


def _searingmeat_fingers(env):
    try:
        q1 = float(env.sim.data.qpos[
            env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint1")])
        q2 = float(env.sim.data.qpos[
            env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint2")])
        return q1, q2
    except Exception:
        return None


def _searingmeat_gripper_contact(env, name):
    try:
        return bool(env.check_contact(env.robots[0].gripper["right"], env.objects[name]))
    except Exception:
        return False


def check_SearingMeat(env):
    aux = _aux_searingmeat(env)

    eef = _searingmeat_eef_pos(env)
    pan = _searingmeat_obj_pos(env, "pan")
    meat = _searingmeat_obj_pos(env, "meat")
    burner = _searingmeat_burner_pos(env)

    if eef is not None and pan is not None:
        d_eef_pan = float(np.linalg.norm(eef - pan))
    else:
        d_eef_pan = float("inf")
    if eef is not None and meat is not None:
        d_eef_meat = float(np.linalg.norm(eef - meat))
    else:
        d_eef_meat = float("inf")

    if pan is not None and burner is not None:
        d_pan_burner_xy = float(np.linalg.norm(pan[:2] - burner[:2]))
        gap_pan_burner = float(pan[2]) - float(burner[2])
    else:
        d_pan_burner_xy = float("inf")
        gap_pan_burner = float("inf")

    if meat is not None and pan is not None:
        d_meat_pan_xy = float(np.linalg.norm(meat[:2] - pan[:2]))
        gap_meat_pan = float(meat[2]) - float(pan[2])
    else:
        d_meat_pan_xy = float("inf")
        gap_meat_pan = float("inf")

    if pan is not None:
        z0_pan = aux.setdefault("z0_pan", float(pan[2]))
        pan_lift = float(pan[2]) - z0_pan
    else:
        pan_lift = 0.0
    if meat is not None:
        z0_meat = aux.setdefault("z0_meat", float(meat[2]))
        meat_lift = float(meat[2]) - z0_meat
    else:
        meat_lift = 0.0

    fingers = _searingmeat_fingers(env)
    closed_038 = fingers is not None and fingers[0] < 0.038 and fingers[1] < 0.038
    pan_grip_contact = _searingmeat_gripper_contact(env, "pan")
    meat_grip_contact = _searingmeat_gripper_contact(env, "meat")
    pan_grasped = _searingmeat_grasped(env, "pan")
    meat_grasped = _searingmeat_grasped(env, "meat")

    try:
        pan_on_stove = OU.check_obj_fixture_contact(env, "pan", env.stove)
    except Exception:
        pan_on_stove = False
    try:
        pan_on_active = bool(
            env.stove.check_obj_location_on_stove(env, "pan", threshold=0.15)
            == env.knob
        )
    except Exception:
        pan_on_active = False

    knob_angle = _searingmeat_knob_angle(env)
    meat_in_pan = _searingmeat_in_recep(env, "meat", "pan", th=0.07)

    progress = OrderedDict([
        ("grasp_pan__s1", d_eef_pan < 0.30),
        ("grasp_pan__s2", d_eef_pan < 0.15),
        ("grasp_pan__s3", d_eef_pan < 0.08),
        ("grasp_pan__s4", pan_grip_contact),
        ("grasp_pan__s5", closed_038),
        ("grasp_pan__s6", pan_grasped or pan_on_stove),
        ("pan_on_stove__s1", pan_lift >= 0.03),
        ("pan_on_stove__s2", d_pan_burner_xy < 0.60),
        ("pan_on_stove__s3", d_pan_burner_xy < 0.40),
        ("pan_on_stove__s4", d_pan_burner_xy < 0.25),
        ("pan_on_stove__s5", gap_pan_burner < 0.10),
        ("pan_on_stove__s6", pan_on_stove),
        ("pan_on_active_knob__s1", d_pan_burner_xy < 0.15),
        ("pan_on_active_knob__s2", d_pan_burner_xy < 0.10),
        ("pan_on_active_knob__s3", knob_angle >= 0.10),
        ("pan_on_active_knob__s4", knob_angle >= 0.22),
        ("pan_on_active_knob__s5", knob_angle >= 0.35),
        ("pan_on_active_knob__s6", pan_on_active),
        ("grasp_meat__s1", d_eef_meat < 0.30),
        ("grasp_meat__s2", d_eef_meat < 0.15),
        ("grasp_meat__s3", d_eef_meat < 0.08),
        ("grasp_meat__s4", meat_grip_contact),
        ("grasp_meat__s5", closed_038),
        ("grasp_meat__s6", meat_grasped or meat_in_pan),
        ("meat_in_pan__s1", meat_lift >= 0.03),
        ("meat_in_pan__s2", d_meat_pan_xy < 0.30),
        ("meat_in_pan__s3", d_meat_pan_xy < 0.15),
        ("meat_in_pan__s4", d_meat_pan_xy < 0.08),
        ("meat_in_pan__s5", gap_meat_pan < 0.10),
        ("meat_in_pan__s6", meat_in_pan),
    ])
    quiescence = OrderedDict([
        ("gripper_meat_far", OU.gripper_obj_far(env, obj_name="meat")),
    ])
    return {"progress": progress, "quiescence": quiescence}

# ============================ SetUpCuttingStation ============================


def _aux_setupcuttingstation(env):
    a = getattr(env, "_aux_setupcuttingstation", None)
    t = int(getattr(env, "timestep", 0))
    if a is None or t < a["_last_t"]:
        a = {"_last_t": t, "_calls": 0}
        env._aux_setupcuttingstation = a
    a["_last_t"] = t
    a["_calls"] += 1
    return a


def _setupcuttingstation_grasped(env, name):
    try:
        return bool(OU.check_obj_grasped(env, name))
    except Exception:
        return False


def _setupcuttingstation_in_recep(env, o, r, **kw):
    try:
        return bool(OU.check_obj_in_receptacle(env, o, r, **kw))
    except Exception:
        return False


def _setupcuttingstation_eef_pos(env):
    try:
        return np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])
    except Exception:
        return None


def _setupcuttingstation_obj_pos(env, name):
    try:
        return np.array(env.sim.data.body_xpos[env.obj_body_id[name]])
    except Exception:
        return None


def _setupcuttingstation_fingers(env):
    try:
        q1 = float(env.sim.data.qpos[
            env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint1")])
        q2 = float(env.sim.data.qpos[
            env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint2")])
        return q1, q2
    except Exception:
        return None


def _setupcuttingstation_gripper_contact(env, name):
    try:
        return bool(env.check_contact(env.robots[0].gripper["right"], env.objects[name]))
    except Exception:
        return False


def check_SetUpCuttingStation(env):
    aux = _aux_setupcuttingstation(env)

    eef = _setupcuttingstation_eef_pos(env)
    meat = _setupcuttingstation_obj_pos(env, "meat")
    knife = _setupcuttingstation_obj_pos(env, "knife")
    board = _setupcuttingstation_obj_pos(env, "receptacle")

    if eef is not None and meat is not None:
        d_eef_meat = float(np.linalg.norm(eef - meat))
    else:
        d_eef_meat = float("inf")
    if eef is not None and knife is not None:
        d_eef_knife = float(np.linalg.norm(eef - knife))
    else:
        d_eef_knife = float("inf")

    if meat is not None and board is not None:
        d_meat_board_xy = float(np.linalg.norm(meat[:2] - board[:2]))
        gap_meat_board = float(meat[2]) - float(board[2])
    else:
        d_meat_board_xy = float("inf")
        gap_meat_board = float("inf")

    if knife is not None and board is not None:
        d_knife_board_xy = float(np.linalg.norm(knife[:2] - board[:2]))
        gap_knife_board = float(knife[2]) - float(board[2])
    else:
        d_knife_board_xy = float("inf")
        gap_knife_board = float("inf")

    if meat is not None:
        z0_meat = aux.setdefault("z0_meat", float(meat[2]))
        meat_lift = float(meat[2]) - z0_meat
    else:
        meat_lift = 0.0
    if knife is not None:
        z0_knife = aux.setdefault("z0_knife", float(knife[2]))
        knife_lift = float(knife[2]) - z0_knife
    else:
        knife_lift = 0.0

    fingers = _setupcuttingstation_fingers(env)
    closed_038 = fingers is not None and fingers[0] < 0.038 and fingers[1] < 0.038
    meat_grip_contact = _setupcuttingstation_gripper_contact(env, "meat")
    knife_grip_contact = _setupcuttingstation_gripper_contact(env, "knife")
    meat_grasped = _setupcuttingstation_grasped(env, "meat")
    knife_grasped = _setupcuttingstation_grasped(env, "knife")
    meat_on_board = _setupcuttingstation_in_recep(env, "meat", "receptacle")
    knife_on_board = _setupcuttingstation_in_recep(env, "knife", "receptacle")

    progress = OrderedDict([
        ("grasp_meat__s1", d_eef_meat < 0.30),
        ("grasp_meat__s2", d_eef_meat < 0.15),
        ("grasp_meat__s3", d_eef_meat < 0.08),
        ("grasp_meat__s4", meat_grip_contact),
        ("grasp_meat__s5", closed_038),
        ("grasp_meat__s6", meat_grasped or meat_on_board),
        ("meat_on_board__s1", meat_lift >= 0.03),
        ("meat_on_board__s2", d_meat_board_xy < 0.40),
        ("meat_on_board__s3", d_meat_board_xy < 0.20),
        ("meat_on_board__s4", d_meat_board_xy < 0.10),
        ("meat_on_board__s5", gap_meat_board < 0.10),
        ("meat_on_board__s6", meat_on_board),
        ("grasp_knife__s1", d_eef_knife < 0.30),
        ("grasp_knife__s2", d_eef_knife < 0.15),
        ("grasp_knife__s3", d_eef_knife < 0.08),
        ("grasp_knife__s4", knife_grip_contact),
        ("grasp_knife__s5", closed_038),
        ("grasp_knife__s6", knife_grasped or knife_on_board),
        ("knife_on_board__s1", knife_lift >= 0.03),
        ("knife_on_board__s2", d_knife_board_xy < 0.40),
        ("knife_on_board__s3", d_knife_board_xy < 0.20),
        ("knife_on_board__s4", d_knife_board_xy < 0.10),
        ("knife_on_board__s5", gap_knife_board < 0.10),
        ("knife_on_board__s6", knife_on_board),
    ])
    quiescence = OrderedDict([
        ("gripper_far", OU.gripper_obj_far(env, "knife") and OU.gripper_obj_far(env, "receptacle")),
    ])
    return {"progress": progress, "quiescence": quiescence}

# ============================ StackBowlsCabinet ============================


def _aux_stackbowlscabinet(env):
    a = getattr(env, "_aux_stackbowlscabinet", None)
    t = int(getattr(env, "timestep", 0))
    if a is None or t < a["_last_t"]:
        a = {"_last_t": t, "_calls": 0}
        env._aux_stackbowlscabinet = a
    a["_last_t"] = t
    a["_calls"] += 1
    return a


def _stackbowlscabinet_grasped(env, name):
    try:
        return bool(OU.check_obj_grasped(env, name))
    except Exception:
        return False


def _stackbowlscabinet_in_recep(env, o, r, **kw):
    try:
        return bool(OU.check_obj_in_receptacle(env, o, r, **kw))
    except Exception:
        return False


def _stackbowlscabinet_inside_cab(env, name, partial):
    try:
        return bool(OU.obj_inside_of(env, name, env.cabinet, partial_check=partial))
    except Exception:
        return False


def _stackbowlscabinet_eef_pos(env):
    try:
        return np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])
    except Exception:
        return None


def _stackbowlscabinet_obj_pos(env, name):
    try:
        return np.array(env.sim.data.body_xpos[env.obj_body_id[name]])
    except Exception:
        return None


def _stackbowlscabinet_cabinet_pos(env):
    try:
        return np.array(env.sim.data.get_body_xpos(env.cabinet.name))
    except Exception:
        return None


def _stackbowlscabinet_fingers(env):
    try:
        q1 = float(env.sim.data.qpos[
            env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint1")])
        q2 = float(env.sim.data.qpos[
            env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint2")])
        return q1, q2
    except Exception:
        return None


def _stackbowlscabinet_gripper_contact(env, name):
    try:
        return bool(env.check_contact(env.robots[0].gripper["right"], env.objects[name]))
    except Exception:
        return False


def check_StackBowlsCabinet(env):
    aux = _aux_stackbowlscabinet(env)

    eef = _stackbowlscabinet_eef_pos(env)
    b1 = _stackbowlscabinet_obj_pos(env, "bowl1")
    b2 = _stackbowlscabinet_obj_pos(env, "bowl2")
    cab = _stackbowlscabinet_cabinet_pos(env)

    if eef is not None and b1 is not None:
        d_eef_b1 = float(np.linalg.norm(eef - b1))
    else:
        d_eef_b1 = float("inf")
    if eef is not None and b2 is not None:
        d_eef_b2 = float(np.linalg.norm(eef - b2))
    else:
        d_eef_b2 = float("inf")
    d_eef_min = min(d_eef_b1, d_eef_b2)

    if b1 is not None and b2 is not None:
        d_b1b2_xy = float(np.linalg.norm(b1[:2] - b2[:2]))
        d_b1b2_3d = float(np.linalg.norm(b1 - b2))
    else:
        d_b1b2_xy = float("inf")
        d_b1b2_3d = float("inf")

    if b1 is not None:
        z0_b1 = aux.setdefault("z0_b1", float(b1[2]))
        lift1 = float(b1[2]) - z0_b1
    else:
        lift1 = 0.0
    if b2 is not None:
        z0_b2 = aux.setdefault("z0_b2", float(b2[2]))
        lift2 = float(b2[2]) - z0_b2
    else:
        lift2 = 0.0
    lift_max = max(lift1, lift2)

    if b1 is not None and cab is not None:
        d_b1_cab = float(np.linalg.norm(b1[:2] - cab[:2]))
    else:
        d_b1_cab = float("inf")
    if b2 is not None and cab is not None:
        d_b2_cab = float(np.linalg.norm(b2[:2] - cab[:2]))
    else:
        d_b2_cab = float("inf")
    d_cab_min = min(d_b1_cab, d_b2_cab)
    d_cab_max = max(d_b1_cab, d_b2_cab)

    fingers = _stackbowlscabinet_fingers(env)
    closed_038 = fingers is not None and fingers[0] < 0.038 and fingers[1] < 0.038
    bowl_grip_contact = (
        _stackbowlscabinet_gripper_contact(env, "bowl1")
        or _stackbowlscabinet_gripper_contact(env, "bowl2")
    )
    grasped1 = _stackbowlscabinet_grasped(env, "bowl1")
    grasped2 = _stackbowlscabinet_grasped(env, "bowl2")
    stacked = (
        _stackbowlscabinet_in_recep(env, "bowl2", "bowl1")
        or _stackbowlscabinet_in_recep(env, "bowl1", "bowl2")
    )

    b1_in_cab = _stackbowlscabinet_inside_cab(env, "bowl1", False)
    b2_in_cab = _stackbowlscabinet_inside_cab(env, "bowl2", False)
    b1_part_cab = _stackbowlscabinet_inside_cab(env, "bowl1", True)
    b2_part_cab = _stackbowlscabinet_inside_cab(env, "bowl2", True)

    progress = OrderedDict([
        ("grasp_any_bowl__s1", d_eef_min < 0.35),
        ("grasp_any_bowl__s2", d_eef_min < 0.20),
        ("grasp_any_bowl__s3", d_eef_min < 0.10),
        ("grasp_any_bowl__s4", bowl_grip_contact),
        ("grasp_any_bowl__s5", closed_038),
        ("grasp_any_bowl__s6", grasped1 or grasped2 or stacked),
        ("bowls_stacked__s1", lift_max >= 0.03),
        ("bowls_stacked__s2", d_b1b2_xy < 0.30),
        ("bowls_stacked__s3", d_b1b2_xy < 0.15),
        ("bowls_stacked__s4", d_b1b2_xy < 0.08),
        ("bowls_stacked__s5", d_b1b2_3d < 0.10),
        ("bowls_stacked__s6", stacked),
        ("any_bowl_in_cabinet__s1", lift_max >= 0.06),
        ("any_bowl_in_cabinet__s2", d_cab_min < 0.60),
        ("any_bowl_in_cabinet__s3", d_cab_min < 0.40),
        ("any_bowl_in_cabinet__s4", d_cab_min < 0.25),
        ("any_bowl_in_cabinet__s5", b1_part_cab or b2_part_cab),
        ("any_bowl_in_cabinet__s6", b1_in_cab or b2_in_cab),
        ("both_bowls_in_cabinet__s1", d_cab_max < 0.60),
        ("both_bowls_in_cabinet__s2", d_cab_max < 0.40),
        ("both_bowls_in_cabinet__s3", d_cab_max < 0.25),
        ("both_bowls_in_cabinet__s4", b1_part_cab and b2_part_cab),
        ("both_bowls_in_cabinet__s5", b1_in_cab and b2_in_cab),
    ])
    quiescence = OrderedDict([
        ("gripper_far",
         OU.gripper_obj_far(env, "bowl1") and OU.gripper_obj_far(env, "bowl2")),
    ])
    return {"progress": progress, "quiescence": quiescence}

# ============================ SteamInMicrowave ============================


def _aux_steaminmicrowave(env):
    a = getattr(env, "_aux_steaminmicrowave", None)
    t = int(getattr(env, "timestep", 0))
    if a is None or t < a["_last_t"]:
        a = {"_last_t": t, "_calls": 0}
        env._aux_steaminmicrowave = a
    a["_last_t"] = t
    a["_calls"] += 1
    return a


def _steaminmicrowave_grasped(env, name):
    try:
        return bool(OU.check_obj_grasped(env, name))
    except Exception:
        return False


def _steaminmicrowave_in_recep(env, o, r, **kw):
    try:
        return bool(OU.check_obj_in_receptacle(env, o, r, **kw))
    except Exception:
        return False


def _steaminmicrowave_body_pos(env, name):
    try:
        return np.array(env.sim.data.body_xpos[env.obj_body_id[name]])
    except Exception:
        return None


def _steaminmicrowave_eef_dist(env, name):
    try:
        obj_pos = env.sim.data.body_xpos[env.obj_body_id[name]]
        eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
        return float(np.linalg.norm(eef_pos - obj_pos))
    except Exception:
        return float("inf")


def _steaminmicrowave_gripper_contact(env, name):
    try:
        return bool(env.check_contact(env.robots[0].gripper["right"], env.objects[name]))
    except Exception:
        return False


def _steaminmicrowave_fingers_closed(env, threshold):
    try:
        q1 = env.sim.data.qpos[env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint1")]
        q2 = env.sim.data.qpos[env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint2")]
        return bool(q1 < threshold and q2 < threshold)
    except Exception:
        return False


def _steaminmicrowave_door_closed(env, th=None):
    try:
        if th is None:
            return bool(env.microwave.is_closed(env))
        return bool(env.microwave.is_closed(env, th=th))
    except Exception:
        return False


def _steaminmicrowave_inside_micro(env, partial):
    try:
        return bool(OU.obj_inside_of(env, "bowl", env.microwave, partial_check=partial))
    except Exception:
        return False


def _steaminmicrowave_xy_dist(p, q):
    if p is None or q is None:
        return float("inf")
    return float(np.linalg.norm(np.asarray(p)[:2] - np.asarray(q)[:2]))


def check_SteamInMicrowave(env):
    a = _aux_steaminmicrowave(env)

    veg_pos = _steaminmicrowave_body_pos(env, "vegetable")
    bowl_pos = _steaminmicrowave_body_pos(env, "bowl")
    if veg_pos is not None:
        a.setdefault("p0_vegetable", veg_pos.copy())
    if bowl_pos is not None:
        a.setdefault("p0_bowl", bowl_pos.copy())
    p0_veg = a.get("p0_vegetable")
    p0_bowl = a.get("p0_bowl")

    d_veg = _steaminmicrowave_eef_dist(env, "vegetable")
    d_bowl = _steaminmicrowave_eef_dist(env, "bowl")
    contact_veg = _steaminmicrowave_gripper_contact(env, "vegetable")
    contact_bowl = _steaminmicrowave_gripper_contact(env, "bowl")
    fingers_038 = _steaminmicrowave_fingers_closed(env, 0.038)

    grasp_veg = _steaminmicrowave_grasped(env, "vegetable")
    grasp_bowl_g = _steaminmicrowave_grasped(env, "bowl")
    veg_in_bowl = _steaminmicrowave_in_recep(env, "vegetable", "bowl")
    micro_partial = _steaminmicrowave_inside_micro(env, True)
    micro_full = _steaminmicrowave_inside_micro(env, False)

    try:
        bowl_hr_th = 0.7 * float(env.objects["bowl"].horizontal_radius)
    except Exception:
        bowl_hr_th = -1.0
    try:
        micro_pos = np.array(env.microwave.pos)
    except Exception:
        micro_pos = None

    xy_veg_bowl = _steaminmicrowave_xy_dist(veg_pos, bowl_pos)
    xy_bowl_micro = _steaminmicrowave_xy_dist(bowl_pos, micro_pos)

    veg_lifted = bool(
        grasp_veg and veg_pos is not None and p0_veg is not None
        and float(veg_pos[2]) > float(p0_veg[2]) + 0.04
    )
    veg_above_bowl = bool(
        veg_pos is not None and bowl_pos is not None
        and float(veg_pos[2]) > float(bowl_pos[2])
    )
    bowl_lifted = bool(
        grasp_bowl_g and bowl_pos is not None and p0_bowl is not None
        and float(bowl_pos[2]) > float(p0_bowl[2]) + 0.04
    )

    progress = OrderedDict([
        # --- grasp_vegetable ---
        ("grasp_vegetable__s1", bool(d_veg <= 0.40)),
        ("grasp_vegetable__s2", bool(d_veg <= 0.25)),
        ("grasp_vegetable__s3", bool(d_veg <= 0.10)),
        ("grasp_vegetable__s4", bool(contact_veg)),
        ("grasp_vegetable__s5", bool(contact_veg and fingers_038)),
        ("grasp_vegetable__s6", bool(grasp_veg or veg_in_bowl)),
        # --- veg_in_bowl ---
        ("veg_in_bowl__s1", veg_lifted),
        ("veg_in_bowl__s2", bool(xy_veg_bowl < 0.40)),
        ("veg_in_bowl__s3", bool(xy_veg_bowl < 0.20)),
        ("veg_in_bowl__s4", bool(xy_veg_bowl < bowl_hr_th and veg_above_bowl)),
        ("veg_in_bowl__s5", bool(veg_in_bowl)),
        ("veg_in_bowl__s6", bool(veg_in_bowl and not grasp_veg)),
        # --- grasp_bowl ---
        ("grasp_bowl__s1", bool(d_bowl <= 0.40)),
        ("grasp_bowl__s2", bool(d_bowl <= 0.25)),
        ("grasp_bowl__s3", bool(d_bowl <= 0.10)),
        ("grasp_bowl__s4", bool(contact_bowl)),
        ("grasp_bowl__s5", bool(contact_bowl and fingers_038)),
        ("grasp_bowl__s6", bool(grasp_bowl_g or micro_full)),
        # --- bowl_in_micro ---
        ("bowl_in_micro__s1", bowl_lifted),
        ("bowl_in_micro__s2", bool(xy_bowl_micro < 0.80)),
        ("bowl_in_micro__s3", bool(xy_bowl_micro < 0.50)),
        ("bowl_in_micro__s4", bool(micro_partial)),
        ("bowl_in_micro__s5", bool(micro_full)),
        ("bowl_in_micro__s6", bool(micro_full and not grasp_bowl_g)),
        # --- door_half_closed ---
        ("door_half_closed__s1", _steaminmicrowave_door_closed(env, th=0.90)),
        ("door_half_closed__s2", _steaminmicrowave_door_closed(env, th=0.75)),
        ("door_half_closed__s3", _steaminmicrowave_door_closed(env, th=0.60)),
        ("door_half_closed__s4", _steaminmicrowave_door_closed(env, th=0.50)),
        ("door_half_closed__s5", _steaminmicrowave_door_closed(env, th=0.40)),
        ("door_half_closed__s6", _steaminmicrowave_door_closed(env, th=0.30)),
        # --- door_closed ---
        ("door_closed__s1", _steaminmicrowave_door_closed(env, th=0.25)),
        ("door_closed__s2", _steaminmicrowave_door_closed(env, th=0.20)),
        ("door_closed__s3", _steaminmicrowave_door_closed(env, th=0.15)),
        ("door_closed__s4", _steaminmicrowave_door_closed(env, th=0.10)),
        ("door_closed__s5", _steaminmicrowave_door_closed(env, th=0.05)),
        ("door_closed__s6", _steaminmicrowave_door_closed(env)),
    ])
    return {"progress": progress, "quiescence": OrderedDict()}

# ============================ StirVegetables ============================


def _aux_stirvegetables(env):
    a = getattr(env, "_aux_stirvegetables", None)
    t = int(getattr(env, "timestep", 0))
    if a is None or t < a["_last_t"]:
        a = {"_last_t": t, "_calls": 0}
        env._aux_stirvegetables = a
    a["_last_t"] = t
    a["_calls"] += 1
    return a


def _stirvegetables_grasped(env, name):
    try:
        return bool(OU.check_obj_grasped(env, name))
    except Exception:
        return False


def _stirvegetables_in_recep(env, o, r, **kw):
    try:
        return bool(OU.check_obj_in_receptacle(env, o, r, **kw))
    except Exception:
        return False


def _stirvegetables_body_pos(env, name):
    try:
        return np.array(env.sim.data.body_xpos[env.obj_body_id[name]])
    except Exception:
        return None


def _stirvegetables_eef_dist(env, name):
    try:
        obj_pos = env.sim.data.body_xpos[env.obj_body_id[name]]
        eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
        return float(np.linalg.norm(eef_pos - obj_pos))
    except Exception:
        return float("inf")


def _stirvegetables_gripper_contact(env, name):
    try:
        return bool(env.check_contact(env.robots[0].gripper["right"], env.objects[name]))
    except Exception:
        return False


def _stirvegetables_obj_contact(env, o1, o2):
    try:
        return bool(env.check_contact(env.objects[o1], env.objects[o2]))
    except Exception:
        return False


def _stirvegetables_fingers_closed(env, threshold):
    try:
        q1 = env.sim.data.qpos[env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint1")]
        q2 = env.sim.data.qpos[env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint2")]
        return bool(q1 < threshold and q2 < threshold)
    except Exception:
        return False


def _stirvegetables_xy_dist(p, q):
    if p is None or q is None:
        return float("inf")
    return float(np.linalg.norm(np.asarray(p)[:2] - np.asarray(q)[:2]))


def check_StirVegetables(env):
    a = _aux_stirvegetables(env)

    veg1_pos = _stirvegetables_body_pos(env, "veg1")
    veg2_pos = _stirvegetables_body_pos(env, "veg2")
    spat_pos = _stirvegetables_body_pos(env, "spatula")
    pot_pos = _stirvegetables_body_pos(env, "pot")
    if veg1_pos is not None:
        a.setdefault("p0_veg1", veg1_pos.copy())
    if veg2_pos is not None:
        a.setdefault("p0_veg2", veg2_pos.copy())
    p0_veg1 = a.get("p0_veg1")
    p0_veg2 = a.get("p0_veg2")

    d_veg1 = _stirvegetables_eef_dist(env, "veg1")
    d_veg2 = _stirvegetables_eef_dist(env, "veg2")
    d_spat = _stirvegetables_eef_dist(env, "spatula")
    contact_veg1 = _stirvegetables_gripper_contact(env, "veg1")
    contact_veg2 = _stirvegetables_gripper_contact(env, "veg2")
    contact_spat = _stirvegetables_gripper_contact(env, "spatula")
    fingers_038 = _stirvegetables_fingers_closed(env, 0.038)

    grasp_veg1 = _stirvegetables_grasped(env, "veg1")
    grasp_veg2 = _stirvegetables_grasped(env, "veg2")
    grasp_spat = _stirvegetables_grasped(env, "spatula")
    veg1_in_pot = _stirvegetables_in_recep(env, "veg1", "pot")
    veg2_in_pot = _stirvegetables_in_recep(env, "veg2", "pot")

    spat_veg_contact = (
        _stirvegetables_obj_contact(env, "spatula", "veg1")
        or _stirvegetables_obj_contact(env, "spatula", "veg2")
    )

    try:
        pot_hr_th = 0.7 * float(env.objects["pot"].horizontal_radius)
    except Exception:
        pot_hr_th = -1.0

    xy_veg1_pot = _stirvegetables_xy_dist(veg1_pos, pot_pos)
    xy_veg2_pot = _stirvegetables_xy_dist(veg2_pos, pot_pos)
    xy_spat_pot = _stirvegetables_xy_dist(spat_pos, pot_pos)

    veg1_lifted = bool(
        grasp_veg1 and veg1_pos is not None and p0_veg1 is not None
        and float(veg1_pos[2]) > float(p0_veg1[2]) + 0.04
    )
    veg2_lifted = bool(
        grasp_veg2 and veg2_pos is not None and p0_veg2 is not None
        and float(veg2_pos[2]) > float(p0_veg2[2]) + 0.04
    )
    veg1_above_pot = bool(
        veg1_pos is not None and pot_pos is not None
        and float(veg1_pos[2]) > float(pot_pos[2])
    )
    veg2_above_pot = bool(
        veg2_pos is not None and pot_pos is not None
        and float(veg2_pos[2]) > float(pot_pos[2])
    )
    spat_lowered = bool(
        spat_pos is not None and pot_pos is not None
        and float(spat_pos[2]) < float(pot_pos[2]) + 0.15
    )

    # gated stir travel accumulators S(veg1)/S(veg2), see parts header
    prev_veg1 = a.get("prev_veg1")
    prev_veg2 = a.get("prev_veg2")
    d1 = (
        float(np.linalg.norm(veg1_pos[:2] - prev_veg1[:2]))
        if (veg1_pos is not None and prev_veg1 is not None) else 0.0
    )
    d2 = (
        float(np.linalg.norm(veg2_pos[:2] - prev_veg2[:2]))
        if (veg2_pos is not None and prev_veg2 is not None) else 0.0
    )
    if grasp_spat and veg1_in_pot and spat_veg_contact:
        a["S_veg1"] = a.get("S_veg1", 0.0) + d1
    if grasp_spat and veg2_in_pot and spat_veg_contact:
        a["S_veg2"] = a.get("S_veg2", 0.0) + d2
    if veg1_pos is not None:
        a["prev_veg1"] = veg1_pos.copy()
    if veg2_pos is not None:
        a["prev_veg2"] = veg2_pos.copy()
    s_veg1 = float(a.get("S_veg1", 0.0))
    s_veg2 = float(a.get("S_veg2", 0.0))

    st = int(getattr(env, "success_time", 0))
    try:
        done = bool(env._check_success())
    except Exception:
        done = False

    progress = OrderedDict([
        # --- grasp_veg1 ---
        ("grasp_veg1__s1", bool(d_veg1 <= 0.40)),
        ("grasp_veg1__s2", bool(d_veg1 <= 0.25)),
        ("grasp_veg1__s3", bool(d_veg1 <= 0.10)),
        ("grasp_veg1__s4", bool(contact_veg1)),
        ("grasp_veg1__s5", bool(contact_veg1 and fingers_038)),
        ("grasp_veg1__s6", bool(grasp_veg1 or veg1_in_pot)),
        # --- veg1_in_pot ---
        ("veg1_in_pot__s1", veg1_lifted),
        ("veg1_in_pot__s2", bool(xy_veg1_pot < 0.40)),
        ("veg1_in_pot__s3", bool(xy_veg1_pot < 0.20)),
        ("veg1_in_pot__s4", bool(xy_veg1_pot < pot_hr_th and veg1_above_pot)),
        ("veg1_in_pot__s5", bool(veg1_in_pot)),
        ("veg1_in_pot__s6", bool(veg1_in_pot and not grasp_veg1)),
        # --- grasp_veg2 ---
        ("grasp_veg2__s1", bool(d_veg2 <= 0.40)),
        ("grasp_veg2__s2", bool(d_veg2 <= 0.25)),
        ("grasp_veg2__s3", bool(d_veg2 <= 0.10)),
        ("grasp_veg2__s4", bool(contact_veg2)),
        ("grasp_veg2__s5", bool(contact_veg2 and fingers_038)),
        ("grasp_veg2__s6", bool(grasp_veg2 or veg2_in_pot)),
        # --- veg2_in_pot ---
        ("veg2_in_pot__s1", veg2_lifted),
        ("veg2_in_pot__s2", bool(xy_veg2_pot < 0.40)),
        ("veg2_in_pot__s3", bool(xy_veg2_pot < 0.20)),
        ("veg2_in_pot__s4", bool(xy_veg2_pot < pot_hr_th and veg2_above_pot)),
        ("veg2_in_pot__s5", bool(veg2_in_pot)),
        ("veg2_in_pot__s6", bool(veg2_in_pot and not grasp_veg2)),
        # --- spatula_grasped ---
        ("spatula_grasped__s1", bool(d_spat <= 0.40)),
        ("spatula_grasped__s2", bool(d_spat <= 0.25)),
        ("spatula_grasped__s3", bool(d_spat <= 0.10)),
        ("spatula_grasped__s4", bool(contact_spat)),
        ("spatula_grasped__s5", bool(contact_spat and fingers_038)),
        ("spatula_grasped__s6", bool(grasp_spat)),
        # --- stir_t1 ---
        ("stir_t1__s1", bool(grasp_spat and xy_spat_pot < 0.40)),
        ("stir_t1__s2", bool(grasp_spat and xy_spat_pot < 0.25)),
        ("stir_t1__s3", bool(grasp_spat and xy_spat_pot < pot_hr_th)),
        ("stir_t1__s4", bool(grasp_spat and xy_spat_pot < pot_hr_th and spat_lowered)),
        ("stir_t1__s5", bool(spat_veg_contact)),
        ("stir_t1__s6", bool(st >= 1)),
        # --- stir_t3 ---
        ("stir_t3__s1", bool(st >= 2)),
        ("stir_t3__s2", bool(s_veg1 >= 0.03)),
        ("stir_t3__s3", bool(s_veg1 >= 0.06)),
        ("stir_t3__s4", bool(s_veg2 >= 0.03)),
        ("stir_t3__s5", bool(s_veg2 >= 0.06)),
        ("stir_t3__s6", bool(st >= 3)),
        # --- task_complete ---
        ("task_complete__s1", bool(st >= 4)),
        ("task_complete__s2", bool(s_veg1 >= 0.09)),
        ("task_complete__s3", bool(s_veg2 >= 0.09)),
        ("task_complete__s4", bool(s_veg1 >= 0.12)),
        ("task_complete__s5", done),
    ])
    return {"progress": progress, "quiescence": OrderedDict()}

# ============================ StoreLeftoversInBowl ============================


def _aux_storeleftoversinbowl(env):
    a = getattr(env, "_aux_storeleftoversinbowl", None)
    t = int(getattr(env, "timestep", 0))
    if a is None or t < a["_last_t"]:
        a = {"_last_t": t, "_calls": 0}
        env._aux_storeleftoversinbowl = a
    a["_last_t"] = t
    a["_calls"] += 1
    return a


def _storeleftoversinbowl_grasped(env, name):
    try:
        return bool(OU.check_obj_grasped(env, name))
    except Exception:
        return False


def _storeleftoversinbowl_in_recep(env, o, r, **kw):
    try:
        return bool(OU.check_obj_in_receptacle(env, o, r, **kw))
    except Exception:
        return False


def _storeleftoversinbowl_body_pos(env, name):
    try:
        return np.array(env.sim.data.body_xpos[env.obj_body_id[name]])
    except Exception:
        return None


def _storeleftoversinbowl_eef_dist(env, name):
    try:
        obj_pos = env.sim.data.body_xpos[env.obj_body_id[name]]
        eef_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
        return float(np.linalg.norm(eef_pos - obj_pos))
    except Exception:
        return float("inf")


def _storeleftoversinbowl_gripper_contact(env, name):
    try:
        return bool(env.check_contact(env.robots[0].gripper["right"], env.objects[name]))
    except Exception:
        return False


def _storeleftoversinbowl_fingers_closed(env, threshold):
    try:
        q1 = env.sim.data.qpos[env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint1")]
        q2 = env.sim.data.qpos[env.sim.model.get_joint_qpos_addr("gripper0_right_finger_joint2")]
        return bool(q1 < threshold and q2 < threshold)
    except Exception:
        return False


def _storeleftoversinbowl_xy_dist(p, q):
    if p is None or q is None:
        return float("inf")
    return float(np.linalg.norm(np.asarray(p)[:2] - np.asarray(q)[:2]))


def check_StoreLeftoversInBowl(env):
    a = _aux_storeleftoversinbowl(env)

    chicken_pos = _storeleftoversinbowl_body_pos(env, "chicken_drumstick")
    veg_pos = _storeleftoversinbowl_body_pos(env, "vegetable")
    bowl_pos = _storeleftoversinbowl_body_pos(env, "bowl")
    if chicken_pos is not None:
        a.setdefault("p0_chicken_drumstick", chicken_pos.copy())
    if veg_pos is not None:
        a.setdefault("p0_vegetable", veg_pos.copy())
    if bowl_pos is not None:
        a.setdefault("p0_bowl", bowl_pos.copy())
    p0_chicken = a.get("p0_chicken_drumstick")
    p0_veg = a.get("p0_vegetable")
    p0_bowl = a.get("p0_bowl")

    d_chicken = _storeleftoversinbowl_eef_dist(env, "chicken_drumstick")
    d_veg = _storeleftoversinbowl_eef_dist(env, "vegetable")
    d_bowl = _storeleftoversinbowl_eef_dist(env, "bowl")
    contact_chicken = _storeleftoversinbowl_gripper_contact(env, "chicken_drumstick")
    contact_veg = _storeleftoversinbowl_gripper_contact(env, "vegetable")
    contact_bowl = _storeleftoversinbowl_gripper_contact(env, "bowl")
    fingers_038 = _storeleftoversinbowl_fingers_closed(env, 0.038)

    grasp_chicken = _storeleftoversinbowl_grasped(env, "chicken_drumstick")
    grasp_veg = _storeleftoversinbowl_grasped(env, "vegetable")
    grasp_bowl_g = _storeleftoversinbowl_grasped(env, "bowl")
    chicken_in_bowl = _storeleftoversinbowl_in_recep(env, "chicken_drumstick", "bowl")
    veg_in_bowl = _storeleftoversinbowl_in_recep(env, "vegetable", "bowl")

    try:
        bowl_in_fridge = env.fridge.check_rack_contact(env, "bowl")
    except Exception:
        bowl_in_fridge = False
    try:
        fridge_partial = bool(OU.obj_inside_of(env, "bowl", env.fridge, partial_check=True))
    except Exception:
        fridge_partial = False
    try:
        bowl_hr_th = 0.7 * float(env.objects["bowl"].horizontal_radius)
    except Exception:
        bowl_hr_th = -1.0
    try:
        fridge_pos = np.array(env.fridge.pos)
    except Exception:
        fridge_pos = None
    try:
        gripper_far = OU.gripper_obj_far(env, "bowl")
    except Exception:
        gripper_far = False

    xy_chicken_bowl = _storeleftoversinbowl_xy_dist(chicken_pos, bowl_pos)
    xy_veg_bowl = _storeleftoversinbowl_xy_dist(veg_pos, bowl_pos)
    xy_bowl_fridge = _storeleftoversinbowl_xy_dist(bowl_pos, fridge_pos)

    chicken_lifted = bool(
        grasp_chicken and chicken_pos is not None and p0_chicken is not None
        and float(chicken_pos[2]) > float(p0_chicken[2]) + 0.04
    )
    veg_lifted = bool(
        grasp_veg and veg_pos is not None and p0_veg is not None
        and float(veg_pos[2]) > float(p0_veg[2]) + 0.04
    )
    bowl_lifted = bool(
        grasp_bowl_g and bowl_pos is not None and p0_bowl is not None
        and float(bowl_pos[2]) > float(p0_bowl[2]) + 0.04
    )
    chicken_above_bowl = bool(
        chicken_pos is not None and bowl_pos is not None
        and float(chicken_pos[2]) > float(bowl_pos[2])
    )
    veg_above_bowl = bool(
        veg_pos is not None and bowl_pos is not None
        and float(veg_pos[2]) > float(bowl_pos[2])
    )

    progress = OrderedDict([
        # --- grasp_chicken ---
        ("grasp_chicken__s1", bool(d_chicken <= 0.40)),
        ("grasp_chicken__s2", bool(d_chicken <= 0.25)),
        ("grasp_chicken__s3", bool(d_chicken <= 0.10)),
        ("grasp_chicken__s4", bool(contact_chicken)),
        ("grasp_chicken__s5", bool(contact_chicken and fingers_038)),
        ("grasp_chicken__s6", bool(grasp_chicken or chicken_in_bowl)),
        # --- chicken_in_bowl ---
        ("chicken_in_bowl__s1", chicken_lifted),
        ("chicken_in_bowl__s2", bool(xy_chicken_bowl < 0.40)),
        ("chicken_in_bowl__s3", bool(xy_chicken_bowl < 0.20)),
        ("chicken_in_bowl__s4", bool(xy_chicken_bowl < bowl_hr_th and chicken_above_bowl)),
        ("chicken_in_bowl__s5", bool(chicken_in_bowl)),
        ("chicken_in_bowl__s6", bool(chicken_in_bowl and not grasp_chicken)),
        # --- grasp_vegetable ---
        ("grasp_vegetable__s1", bool(d_veg <= 0.40)),
        ("grasp_vegetable__s2", bool(d_veg <= 0.25)),
        ("grasp_vegetable__s3", bool(d_veg <= 0.10)),
        ("grasp_vegetable__s4", bool(contact_veg)),
        ("grasp_vegetable__s5", bool(contact_veg and fingers_038)),
        ("grasp_vegetable__s6", bool(grasp_veg or veg_in_bowl)),
        # --- vegetable_in_bowl ---
        ("vegetable_in_bowl__s1", veg_lifted),
        ("vegetable_in_bowl__s2", bool(xy_veg_bowl < 0.40)),
        ("vegetable_in_bowl__s3", bool(xy_veg_bowl < 0.20)),
        ("vegetable_in_bowl__s4", bool(xy_veg_bowl < bowl_hr_th and veg_above_bowl)),
        ("vegetable_in_bowl__s5", bool(veg_in_bowl)),
        ("vegetable_in_bowl__s6", bool(veg_in_bowl and not grasp_veg)),
        # --- grasp_bowl ---
        ("grasp_bowl__s1", bool(d_bowl <= 0.40)),
        ("grasp_bowl__s2", bool(d_bowl <= 0.25)),
        ("grasp_bowl__s3", bool(d_bowl <= 0.10)),
        ("grasp_bowl__s4", bool(contact_bowl)),
        ("grasp_bowl__s5", bool(contact_bowl and fingers_038)),
        ("grasp_bowl__s6", bool(grasp_bowl_g or bowl_in_fridge)),
        # --- bowl_in_fridge ---
        ("bowl_in_fridge__s1", bowl_lifted),
        ("bowl_in_fridge__s2", bool(xy_bowl_fridge < 1.20)),
        ("bowl_in_fridge__s3", bool(xy_bowl_fridge < 0.80)),
        ("bowl_in_fridge__s4", bool(xy_bowl_fridge < 0.50)),
        ("bowl_in_fridge__s5", bool(fridge_partial)),
        ("bowl_in_fridge__s6", bool(bowl_in_fridge)),
    ])
    quiescence = OrderedDict([
        ("gripper_far", gripper_far),
    ])
    return {"progress": progress, "quiescence": quiescence}

# ============================ WashLettuce ============================


def _washlettuce_grasped(env, name):
    try:
        return bool(OU.check_obj_grasped(env, name, threshold=0.035))
    except Exception:
        return False


def _washlettuce_handle_dist(env, eef_pos):
    try:
        jid = env.sim.model.joint_name2id(env.sink.naming_prefix + "handle_joint")
        handle_pos = np.array(env.sim.data.body_xpos[env.sim.model.jnt_bodyid[jid]])
        return float(np.linalg.norm(eef_pos - handle_pos))
    except Exception:
        return float("inf")


def _washlettuce_handle_state(env):
    # (handle_joint qpos wrapped to [0, 2pi), water_on)
    try:
        h = env.sink.get_handle_state(env=env)
        return float(h["handle_joint"]), bool(h["water_on"])
    except Exception:
        return 0.0, False


def _washlettuce_water_site(env):
    # (water site pos, z_lim = site z + site size[1]); the checker z_check bound
    try:
        site = env.sink.water_site
        sid = env.sim.model.site_name2id(site.get("name"))
        pos = np.array(env.sim.data.site_xpos[sid])
        size1 = float(site.get("size").split()[1])
        return pos, float(pos[2]) + size1
    except Exception:
        return None, None


def _washlettuce_under_water(env):
    try:
        return bool(env.sink.check_obj_under_water(env, "lettuce"))
    except Exception:
        return False


def check_WashLettuce(env):
    lettuce_pos = np.array(env.sim.data.body_xpos[env.obj_body_id["lettuce"]])
    eef_pos = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])

    d_eef_lettuce = float(np.linalg.norm(eef_pos - lettuce_pos))
    d_handle = _washlettuce_handle_dist(env, eef_pos)
    q_h, water_on = _washlettuce_handle_state(env)

    water_pos, z_lim = _washlettuce_water_site(env)
    if water_pos is not None:
        d_xy_water = float(np.linalg.norm(lettuce_pos[:2] - water_pos[:2]))
    else:
        d_xy_water = float("inf")
    z_ok = bool(z_lim is not None and float(lettuce_pos[2]) < z_lim)

    grasped = _washlettuce_grasped(env, "lettuce")
    under_water = _washlettuce_under_water(env)
    wt = int(getattr(env, "washed_time", 0))
    try:
        done = bool(env._check_success())
    except Exception:
        done = False

    progress = OrderedDict([
        ("water_on__s1", bool(d_handle < 0.40)),
        ("water_on__s2", bool(d_handle < 0.20)),
        ("water_on__s3", bool(q_h > 0.05)),
        ("water_on__s4", bool(q_h > 0.15)),
        ("water_on__s5", bool(q_h > 0.30)),
        ("water_on__s6", bool(water_on)),
        ("lettuce_under_water__s1", bool(d_eef_lettuce <= 0.15)),
        ("lettuce_under_water__s2", bool(grasped)),
        ("lettuce_under_water__s3", bool(d_xy_water < 0.40)),
        ("lettuce_under_water__s4", bool(d_xy_water < 0.15)),
        ("lettuce_under_water__s5", bool(z_ok)),
        ("lettuce_under_water__s6", bool(under_water)),
        ("wash_t5__s1", bool(wt >= 1)),
        ("wash_t5__s2", bool(wt >= 2)),
        ("wash_t5__s3", bool(wt >= 3)),
        ("wash_t5__s4", bool(wt >= 4)),
        ("wash_t5__s5", bool(wt >= 5)),
        ("wash_t10__s1", bool(wt >= 6)),
        ("wash_t10__s2", bool(wt >= 7)),
        ("wash_t10__s3", bool(wt >= 8)),
        ("wash_t10__s4", bool(wt >= 9)),
        ("wash_t10__s5", bool(wt >= 10)),
        ("wash_t15__s1", bool(wt >= 11)),
        ("wash_t15__s2", bool(wt >= 12)),
        ("wash_t15__s3", bool(wt >= 13)),
        ("wash_t15__s4", bool(wt >= 14)),
        ("wash_t15__s5", bool(wt >= 15)),
        ("wash_t20__s1", bool(wt >= 16)),
        ("wash_t20__s2", bool(wt >= 17)),
        ("wash_t20__s3", bool(wt >= 18)),
        ("wash_t20__s4", bool(wt >= 19)),
        ("wash_t20__s5", bool(wt >= 20)),
        ("task_complete__s1", bool(wt >= 21)),
        ("task_complete__s2", bool(wt >= 22)),
        ("task_complete__s3", bool(wt >= 23)),
        ("task_complete__s4", bool(wt >= 24)),
        ("task_complete__s5", bool(done)),
    ])
    quiescence = OrderedDict()
    return {"progress": progress, "quiescence": quiescence}

_DISPATCH = {
    "DeliverStraw": check_DeliverStraw,
    "GetToastedBread": check_GetToastedBread,
    "KettleBoiling": check_KettleBoiling,
    "LoadDishwasher": check_LoadDishwasher,
    "PackIdenticalLunches": check_PackIdenticalLunches,
    "PreSoakPan": check_PreSoakPan,
    "PrepareCoffee": check_PrepareCoffee,
    "RinseSinkBasin": check_RinseSinkBasin,
    "ScrubCuttingBoard": check_ScrubCuttingBoard,
    "SearingMeat": check_SearingMeat,
    "SetUpCuttingStation": check_SetUpCuttingStation,
    "StackBowlsCabinet": check_StackBowlsCabinet,
    "SteamInMicrowave": check_SteamInMicrowave,
    "StirVegetables": check_StirVegetables,
    "StoreLeftoversInBowl": check_StoreLeftoversInBowl,
    "WashLettuce": check_WashLettuce,
}

def get_subtask_states(env) -> dict:
    fn = _DISPATCH.get(type(env).__name__)
    if fn is not None:
        return fn(env)
    try:
        success = bool(env._check_success())
    except Exception:
        success = False
    return {"progress": OrderedDict([("success", success)]),
            "quiescence": OrderedDict()}
