"""Smooth-band reward variant (continuous band, NO persistent penalty).

This is the smooth-curve counterpart of band_reward.py:

  band         = TimingReward       with r_persistent backed out  (#1 + #2 piecewise, no #3)
  smooth_band  = SmoothTimingReward with r_persistent backed out  (#1 + #2 continuous, no #3)

Components:
    #1 stage-ordered event reward .................. KEPT (inherited)
    #2 ratio band ................................. KEPT, CONTINUOUS 0.6/(1+ratio)
                                                     (inherited from SmoothTimingReward)
    #3 persistent stuck penalty (r_persistent) ..... REMOVED (backed out every step)
    outcome (+2.0 one-shot) ........................ KEPT (inherited)

So smooth_band vs smooth_timing differ ONLY by the persistent stuck penalty,
exactly as band vs timing differ ONLY by r_persistent. The band's functional
form is the continuous curve in BOTH smooth variants.

Selected via STRUCTRL_REWARD_VARIANT=smooth_band. Does NOT modify any existing
*_reward.py.
"""
from __future__ import annotations

from typing import Optional

from structrl.smooth_timing_reward import SmoothTimingReward
from structrl.timing_reward_builder import build_timing_reward


class SmoothBandReward(SmoothTimingReward):
    """SmoothTimingReward (continuous band kept) with the persistent stuck
    penalty REMOVED.

    `_ratio_band_reward` is inherited from SmoothTimingReward (continuous
    0.6/(1+ratio)). Only step() is overridden, and only to back out the
    r_persistent term the parent added — identical mechanism to
    band_reward.BandReward.
    """

    def step(self, grouped_states: dict, success: bool,
             sim_t: Optional[int] = None) -> dict:
        out = super().step(grouped_states, success, sim_t=sim_t)
        rp = float(out.get("r_persistent", 0.0))
        if rp != 0.0:
            self.cumulative -= rp
            self.cumulative_persistent -= rp
            out["r_total"] = float(out["r_total"]) - rp
            out["r_persistent"] = 0.0
            out["cumulative"] = self.cumulative
            out["cumulative_persistent"] = self.cumulative_persistent
        return out


def build_smooth_band_reward(task_name: str, timing_data: dict) -> SmoothBandReward:
    """Build a SmoothBandReward for `task_name`, reusing the exact stage/budget
    assembly from build_timing_reward (so the stage structure is byte-identical
    to all other variants; only the band form is continuous and #3 is removed)."""
    base = build_timing_reward(task_name, timing_data)
    return SmoothBandReward(stages=base.stages, rank_budgets=base.rank_budgets)
