"""Smooth-band timing reward variant — pure subclass of TimingReward.

Motivation: the stock TimingReward maps (ratio = elapsed/demo_budget) to the
per-subtask event reward with a hand-tuned 3-piece PIECEWISE function
(`_ratio_band_reward` in timing_reward.py):

    ratio <= 1.0      : sub_base * (1 + BONUS_COEF*(1-ratio))   # bonus, up to 1.5x
    1.0 < ratio <= 2.0: sub_base                                # flat plateau
    ratio > 2.0       : max(MIN_EVENT_FRAC*sub_base, decay)     # decay to 0.015 floor

That shape carries FOUR free knobs (BONUS_COEF, TOLERANCE plateau width, decay
slope, MIN_EVENT_FRAC floor) plus two kinks. This variant replaces the whole
piecewise map with ONE smooth, knob-free curve:

    r_event = 2 * sub_base / (1 + ratio)

Properties (with sub_base = FIXED_SUBTASK_REWARD = 0.3):
    ratio -> 0  (much faster than demo) : 2*sub_base = 0.60   <- hard cap, no blow-up
    ratio  = 1  (exactly demo budget)   :   sub_base = 0.30   <- ANCHOR, data-given
    ratio  = 2  (2x demo time)          :  0.667*sub_base = 0.20
    ratio -> inf                        : -> 0                <- floor, never negative

So "finish fast -> reward up to 0.60, finish slow -> reward decays smoothly to 0"
is expressed by a single rational function. ratio=1 mapping to sub_base is NOT a
tuned parameter: it falls out of the curve (2/(1+1)=1), and the anchor itself is
the human-demo budget. The only design number is the cap multiplier (=2, i.e.
fastest-completion reward = 2x the on-demo reward), deliberately fixed to 2.

WHAT CHANGES vs the full `timing` variant:
    #1 stage-ordered event reward .................. KEPT (inherited)
    #2 ratio-band shaping .......................... SMOOTHED (this override)
    #3 persistent stuck penalty (r_persistent) ..... KEPT (inherited verbatim)
    outcome (+2.0 one-shot) ........................ KEPT (inherited)
This is the FULL timing reward (#1+#2+#3) with ONLY the band's functional form
swapped from piecewise to continuous. persistent KEPT on purpose: the robot is
legitimately slower than human demos, so overtime must still be tolerated up to
TOLERANCE*budget before the per-step bleed kicks in -- making persistent
continuous ("punish past ratio=1") would punish normal-but-slow behavior almost
everywhere. Hence persistent stays exactly as in timing_reward.py.

This file does NOT modify timing_reward.py or any other *_reward.py. It is a
drop-in alternative reward object selected via
STRUCTRL_REWARD_VARIANT=smooth_timing.

Ablation framing: smooth_timing - timing (tr-cs8) isolates the effect of the
band's FUNCTIONAL FORM (continuous 0.6/(1+ratio) vs piecewise) with everything
else held fixed. NOTE the cap moved 0.45 -> 0.60 and the floor 0.015 -> ~0, so
the dense/final ratio differs slightly; treat this as a re-designed timing
reward, run fresh, not byte-comparable to tr-cs8.
"""
from __future__ import annotations

from structrl.timing_reward import TimingReward
from structrl.timing_reward_builder import build_timing_reward


# Cap multiplier: fastest-completion (ratio->0) event reward = CAP_MULT * sub_base.
# Fixed to 2 by design (cap 0.60, on-demo anchor 0.30). This is the single
# shape number; the curve is otherwise knob-free and anchored at ratio=1.
CAP_MULT = 2.0


class SmoothTimingReward(TimingReward):
    """Full TimingReward (#1+#2+#3) with the ratio-band map replaced by the
    smooth, knob-free curve CAP_MULT*sub_base/(1+ratio).

    ONLY `_ratio_band_reward` is overridden. step(), the persistent stuck
    penalty, the outcome reward, and all stage/latch/silent-skip machinery are
    inherited from TimingReward unchanged.
    """

    def _ratio_band_reward(self, ratio: float, sub_base: float) -> float:
        # Continuous bonus+decay in one function:
        #   ratio<1 -> >sub_base (bonus, capped at CAP_MULT*sub_base as ratio->0)
        #   ratio=1 -> sub_base  (anchor = demo budget, falls out of the curve)
        #   ratio>1 -> <sub_base, smoothly decaying to 0 as ratio->inf (never <0)
        return CAP_MULT * sub_base / (1.0 + max(ratio, 0.0))


def build_smooth_timing_reward(task_name: str, timing_data: dict) -> SmoothTimingReward:
    """Build a SmoothTimingReward for `task_name`, reusing the exact stage/budget
    assembly from build_timing_reward (so the stage structure is byte-identical
    to the timing/flat/band/flat_persist variants; only the band's functional
    form differs)."""
    base = build_timing_reward(task_name, timing_data)
    return SmoothTimingReward(stages=base.stages, rank_budgets=base.rank_budgets)
