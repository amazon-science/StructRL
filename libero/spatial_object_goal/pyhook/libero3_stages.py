# LIBERO-Spatial/Object/Goal (Table 5): stage splits, grasp predicate, Goal MeanStd patch.
# =====================================================================
# Standalone module; no edits to RLinf sources. Loaded via sitecustomize
# (PYTHONPATH auto-import) in every process, the same mechanism as ablations/.
#
# 1) TASK_STAGES entries for libero_{spatial,object,goal} (30 tasks / 20 goal
#    signatures) are injected into rlinf.envs.libero.dense_reward.TASK_STAGES.
#    Splits follow a "pick-up -> place" scheme: a GRASP stage
#    (new predicate, robosuite _check_grasp) before each placement conjunct;
#    articulated/push/knob tasks stay single-stage (no graspable milestone).
# 2) eval_conjuncts_true is wrapped so ("grasp", <obj>) conjuncts are evaluated
#    via the leaf env's robosuite _check_grasp (both gripper fingers in
#    contact); all other conjuncts go through the original _eval_predicate
#    path unchanged.
# 3) If LIBERO3_GOAL_MEANSTD=1, LiberoFrankaDataConfig.transform is wrapped to
#    switch action x/y/z/roll/pitch/yaw normalization from min_max to
#    mean_std (gripper stays min_max), as required by the public GR00T-N1.5
#    LIBERO-Goal checkpoint, which was trained with mean/std normalization.

import os


def _c(*parts):
    return tuple(parts)


_GRASP = "grasp"

# ---- libero_spatial: all 10 tasks share ONE goal signature ----------------
_SPATIAL = {
    frozenset({_c("on", "akita_black_bowl_1", "plate_1")}): [
        [_c(_GRASP, "akita_black_bowl_1")],
        [_c("on", "akita_black_bowl_1", "plate_1")],
    ],
}

# ---- libero_object: 10 tasks, one object each -----------------------------
_OBJECT_NAMES = [
    "alphabet_soup_1", "bbq_sauce_1", "butter_1", "chocolate_pudding_1",
    "cream_cheese_1", "ketchup_1", "milk_1", "orange_juice_1",
    "salad_dressing_1", "tomato_sauce_1",
]
_OBJECT = {
    frozenset({_c("in", o, "basket_1_contain_region")}): [
        [_c(_GRASP, o)],
        [_c("in", o, "basket_1_contain_region")],
    ]
    for o in _OBJECT_NAMES
}

# ---- libero_goal: 10 tasks -------------------------------------------------
_GOAL = {
    # open the middle drawer — articulated pull, no graspable milestone.
    frozenset({_c("open", "wooden_cabinet_1_middle_region")}): [
        [_c("open", "wooden_cabinet_1_middle_region")],
    ],
    # open the top drawer AND put the bowl inside — 3 hard-sequential stages
    # (drawer must open first; "open" is an AUXILIARY conjunct, not in goal).
    frozenset({_c("in", "akita_black_bowl_1", "wooden_cabinet_1_top_region")}): [
        [_c("open", "wooden_cabinet_1_top_region")],
        [_c(_GRASP, "akita_black_bowl_1")],
        [_c("in", "akita_black_bowl_1", "wooden_cabinet_1_top_region")],
    ],
    # push the plate to the front of the stove — push, no grasp stage.
    frozenset({_c("on", "plate_1", "main_table_stove_front_region")}): [
        [_c("on", "plate_1", "main_table_stove_front_region")],
    ],
    # put the bowl on the plate — same signature as ALL libero_spatial tasks;
    # identical split, so the shared TASK_STAGES key is consistent by design.
    # (covered by _SPATIAL entry; repeated here as documentation only)
    # put the bowl on the stove
    frozenset({_c("on", "akita_black_bowl_1", "flat_stove_1_cook_region")}): [
        [_c(_GRASP, "akita_black_bowl_1")],
        [_c("on", "akita_black_bowl_1", "flat_stove_1_cook_region")],
    ],
    # put the bowl on top of the cabinet
    frozenset({_c("on", "akita_black_bowl_1", "wooden_cabinet_1_top_side")}): [
        [_c(_GRASP, "akita_black_bowl_1")],
        [_c("on", "akita_black_bowl_1", "wooden_cabinet_1_top_side")],
    ],
    # put the cream cheese in(on) the bowl
    frozenset({_c("on", "cream_cheese_1", "akita_black_bowl_1")}): [
        [_c(_GRASP, "cream_cheese_1")],
        [_c("on", "cream_cheese_1", "akita_black_bowl_1")],
    ],
    # put the wine bottle on the rack
    frozenset({_c("on", "wine_bottle_1", "wine_rack_1_top_region")}): [
        [_c(_GRASP, "wine_bottle_1")],
        [_c("on", "wine_bottle_1", "wine_rack_1_top_region")],
    ],
    # put the wine bottle on top of the cabinet
    frozenset({_c("on", "wine_bottle_1", "wooden_cabinet_1_top_side")}): [
        [_c(_GRASP, "wine_bottle_1")],
        [_c("on", "wine_bottle_1", "wooden_cabinet_1_top_side")],
    ],
    # turn on the stove — knob twist, single conjunct.
    frozenset({_c("turnon", "flat_stove_1")}): [
        [_c("turnon", "flat_stove_1")],
    ],
}

LIBERO3_STAGES = {}
LIBERO3_STAGES.update(_SPATIAL)
LIBERO3_STAGES.update(_OBJECT)
LIBERO3_STAGES.update(_GOAL)


def _eval_grasp(inner, obj_name: str) -> bool:
    """Robosuite two-finger grasp check on the named BDDL object."""
    obj = None
    for attr in ("objects_dict", "object_dict"):
        d = getattr(inner, attr, None)
        if d and obj_name in d:
            obj = d[obj_name]
            break
    if obj is None and hasattr(inner, "get_object"):
        try:
            obj = inner.get_object(obj_name)
        except Exception:
            obj = None
    if obj is None:
        return False
    try:
        return bool(
            inner._check_grasp(
                gripper=inner.robots[0].gripper,
                object_geoms=obj,
            )
        )
    except Exception:
        return False


def patch_dense_reward(dr):
    """Apply stage table + grasp predicate to an ALREADY-IMPORTED
    rlinf.envs.libero.dense_reward module (lazy hook entry point)."""
    dr.TASK_STAGES.update(LIBERO3_STAGES)

    _orig_eval = dr.eval_conjuncts_true

    def eval_conjuncts_true_with_grasp(inner, conjuncts):
        conjuncts = list(conjuncts)
        grasp_cs = [c for c in conjuncts if c and c[0] == _GRASP]
        other_cs = [c for c in conjuncts if not (c and c[0] == _GRASP)]
        out = _orig_eval(inner, other_cs)
        for c in grasp_cs:
            if _eval_grasp(inner, c[1]):
                out.add(c)
        return out

    dr.eval_conjuncts_true = eval_conjuncts_true_with_grasp
    import sys
    sys.stderr.write(
        f"[libero3 hook] TASK_STAGES +{len(LIBERO3_STAGES)} signatures, grasp predicate ON\n")
    sys.stderr.flush()


def patch_modality_config(mc):
    """Flip LiberoFrankaDataConfig action x..yaw to mean_std (gripper stays
    min_max) on an ALREADY-IMPORTED modality_config module — required by the
    public GR00T-N1.5 LIBERO-Goal checkpoint. Mirrors LiberoDataConfigMeanStd."""
    _orig_transform = mc.LiberoFrankaDataConfig.transform

    def transform_meanstd(self):
        composed = _orig_transform(self)
        for t in composed.transforms:
            modes = getattr(t, "normalization_modes", None)
            if isinstance(modes, dict) and "action.x" in modes:
                for k in ("action.x", "action.y", "action.z",
                          "action.roll", "action.pitch", "action.yaw"):
                    modes[k] = "mean_std"
        return composed

    mc.LiberoFrankaDataConfig.transform = transform_meanstd
    import sys
    sys.stderr.write("[libero3 hook] GOAL MeanStd action-norm ON\n")
    sys.stderr.flush()


def install():
    """Eager variant (kept for the timing extractor, which imports
    dense_reward anyway before calling this)."""
    import rlinf.envs.libero.dense_reward as dr
    patch_dense_reward(dr)
    if os.environ.get("LIBERO3_GOAL_MEANSTD") == "1":
        from rlinf.models.embodiment.gr00t import modality_config as mc
        patch_modality_config(mc)
