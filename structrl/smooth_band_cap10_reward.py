"""Smooth-band reward, CAP=1.0 variant (continuous band, NO persistent penalty).

Identical to smooth_band_reward.SmoothBandReward in EVERY respect (continuous
ratio band, no #3 persistent stuck penalty, +2.0 outcome) EXCEPT the band's cap
multiplier is raised so the ratio->0 (finish-instantly) event reward caps at
1.0 instead of 0.6:

    smooth_band      : r_event = CAP_MULT * sub_base / (1+ratio),  CAP_MULT=2.0
                       -> ratio->0 cap = 2.0*0.3 = 0.60, anchor(ratio=1)=0.30
    smooth_band_cap10: r_event =          1.0 / (1+ratio)
                       -> ratio->0 cap = 1.00,           anchor(ratio=1)=0.50

I.e. the whole band curve is scaled up by 1.0/0.6 = 5/3. The anchor at ratio=1
moves 0.30 -> 0.50 and the cap 0.60 -> 1.00; the functional FORM (rational
1/(1+ratio) decay, never negative) is unchanged. Everything else (stage event
machinery, persistent back-out, outcome) is inherited verbatim from
SmoothBandReward.

Selected via STRUCTRL_REWARD_VARIANT=smooth_band_cap10. Does NOT modify any
existing *_reward.py.
"""
from __future__ import annotations

from structrl.smooth_band_reward import SmoothBandReward
from structrl.timing_reward_builder import build_timing_reward


# Numerator of the continuous band curve r_event = BAND_CAP / (1 + ratio).
# smooth_band uses CAP_MULT(2.0) * sub_base(0.3) = 0.6; this variant uses 1.0.
BAND_CAP = 1.0


class SmoothBandCap10Reward(SmoothBandReward):
    """SmoothBandReward with the continuous band cap raised from 0.6 to 1.0.

    Only `_ratio_band_reward` is overridden; the persistent back-out in
    SmoothBandReward.step() and all stage/outcome machinery are inherited.
    """

    def _ratio_band_reward(self, ratio: float, sub_base: float) -> float:
        # Continuous decay capped at BAND_CAP as ratio->0; anchor(ratio=1)=0.5.
        # sub_base is intentionally NOT used: the cap is now an absolute 1.0
        # rather than CAP_MULT*sub_base, matching the user-requested 1.0 band.
        return BAND_CAP / (1.0 + max(ratio, 0.0))


def build_smooth_band_cap10_reward(task_name: str, timing_data: dict) -> SmoothBandCap10Reward:
    """Build a SmoothBandCap10Reward for `task_name`, reusing the exact
    stage/budget assembly from build_timing_reward (byte-identical stage
    structure to all other variants; only the band cap = 1.0 differs)."""
    base = build_timing_reward(task_name, timing_data)
    return SmoothBandCap10Reward(stages=base.stages, rank_budgets=base.rank_budgets)
