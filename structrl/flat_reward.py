"""Flat-event reward variant (ablation A) — pure subclass of TimingReward.

Ablation goal: isolate whether the elaborate timing shaping (ratio-band
bonus/decay + persistent stuck penalty) actually contributes over the most
naive dense signal. This variant keeps EVERYTHING about the stage/subtask
latching machinery (strict stage order, same-stage parallel completion,
silent-skip of future stages) and changes only the reward NUMBERS:

  * event reward = constant FIXED_SUBTASK_REWARD (0.3) on first latch of a
    subtask, regardless of speed. NO early-completion bonus, NO late-completion
    decay (the entire ratio-band is collapsed to a constant).
  * NO persistent stuck penalty (r_persistent forced to 0). No negative values
    anywhere except none — reward is monotone non-negative.
  * outcome reward = OUTCOME_REWARD (2.0), one-shot on first success (inherited
    unchanged).

So per chunk: R_c = 0.3 * (#subtasks first-latched this chunk) + 2.0*1[success].

This file does NOT modify timing_reward.py or venv.py's timing path. It is a
drop-in alternative reward object selected via STRUCTRL_REWARD_VARIANT=flat.
"""
from __future__ import annotations

from typing import Optional

from structrl.timing_reward import TimingReward
from structrl.timing_reward_builder import build_timing_reward


class FlatEventReward(TimingReward):
    """TimingReward with flat event reward and no persistent penalty.

    Overrides only the two reward-number knobs; all stage/latch/silent-skip
    logic is inherited verbatim from TimingReward.step().
    """

    def _ratio_band_reward(self, ratio: float, sub_base: float) -> float:
        # Collapse the entire ratio-band (bonus / tolerance / decay) to a flat
        # constant: completing a subtask is worth sub_base (=0.3) no matter how
        # fast or slow. `ratio` is ignored on purpose.
        return sub_base

    def step(self, grouped_states: dict, success: bool,
             sim_t: Optional[int] = None) -> dict:
        out = super().step(grouped_states, success, sim_t=sim_t)
        # Back out the persistent stuck penalty entirely (it was already added
        # into r_total / cumulative by the parent). Event reward is already flat
        # via the _ratio_band_reward override above.
        rp = float(out.get("r_persistent", 0.0))
        if rp != 0.0:
            self.cumulative -= rp
            self.cumulative_persistent -= rp
            out["r_total"] = float(out["r_total"]) - rp
            out["r_persistent"] = 0.0
            out["cumulative"] = self.cumulative
            out["cumulative_persistent"] = self.cumulative_persistent
        return out


def build_flat_reward(task_name: str, timing_data: dict) -> FlatEventReward:
    """Build a FlatEventReward for `task_name`, reusing the exact stage/budget
    assembly from build_timing_reward (so the stage structure is byte-identical
    to the timing variant; only the reward numbers differ)."""
    base = build_timing_reward(task_name, timing_data)
    return FlatEventReward(stages=base.stages, rank_budgets=base.rank_budgets)
