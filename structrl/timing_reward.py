"""Stage-based incremental dense reward driven by demo timing budgets.

Design (locked-in spec):
- Each task has K stages. Each stage is a SET of subtask flags from
  structrl.subtask_checker. Stages execute in strict order; subtasks
  WITHIN a stage are parallel (any completion order valid).
- Each stage k has a list of "rank budgets":
    rank_budgets[k] = [B_0, B_1, ..., B_{|S_k|-1}]
  where B_r = median(elapsed-from-stage-start to the r-th completion in S_k)
  computed from human demo data (see structrl.extract_demo_timing).
- A stage is "complete" when ALL its subtasks have latched True. Once
  complete, the next stage opens and its stage_start_step = current step.
- Latched: a subtask, once True for the FIRST time within its allowed
  stage, is permanently latched (subsequent flips don't matter).
- Silent skip: if a subtask belongs to a stage > current_stage, it is
  IGNORED (not latched), no event reward fires.

Reward logic (per step):
  sub_base = FIXED_SUBTASK_REWARD (0.3) per subtask, independent of task
  structure (NOT normalized by stage/subtask count).

  When subtask in current stage transitions False -> True:
    elapsed = t - subtask_start_step (D3: gap since last completion)
    budget = rank_budgets[stage_idx][rank] where rank = #already-latched
    ratio = elapsed / budget
    if ratio <= 1.0:
      r_event = sub_base * (1.0 + BONUS_COEF * (1 - ratio))  # bonus, span [1x,1.5x]
    elif ratio <= TOLERANCE:
      r_event = sub_base                                      # base
    else:
      penalty = (ratio - TOLERANCE) * sub_base / 2.0
      r_event = max(MIN_EVENT_FRAC*sub_base, sub_base - penalty)  # ratio-based decay
      # hits the floor at ratio ~= 3.9 (i.e. ~3.9x demo budget)

  Persistent stuck penalty (each step, when current stage not complete and
  the gap since last completion > TOLERANCE * next-rank budget):
    r_persistent = -DECAY_PER_STEP   # -0.001 / env-step = -0.016 / 16-step chunk

  Outcome (one-shot, when env._check_success() first True):
    r_outcome = OUTCOME_REWARD (2.0)

  Total per step: r = r_event + r_persistent + r_outcome

Total return upper bound (perfect trajectory):
  n_subtasks × 0.3 × max_bonus(1.5) + outcome(2.0)
  e.g. 4-subtask task: 4*0.3*1.5 + 2.0 = 3.8
"""
from __future__ import annotations

from typing import Optional


# Tunable constants (matching the spec we agreed on)
BONUS_COEF = 0.5          # extra reward at ratio=0 vs ratio=1 (early bonus span [1.0x, 1.5x])
TOLERANCE = 2.0           # ratios > this trigger decay. 1.5 was too strict for
                          # this robot (much slower than human demos); 2.0 gives
                          # a more forgiving window before penalties kick in.
DECAY_PER_STEP = 0.001    # persistent stuck bleed per env-step (-0.016 / 16-step chunk)
MIN_EVENT_FRAC = 0.05     # event reward floor (relative to sub_base)
import os as _os  # noqa
OUTCOME_REWARD = float(_os.environ.get("OUTCOME_REWARD_OVERRIDE", "2.0"))  # one-shot success reward — kept > any task's total progress; env-overridable for ablation (default 2.0 unchanged)
FINALIZE_GRACE_STEPS = 300  # after all progress latched, grace (sim steps) the
                            # policy gets to retreat/finish before the stuck
                            # penalty resumes. Flat (NOT demo-derived): human
                            # demos retreat in ~3 frames, but the robot needs a
                            # generous, fixed window. Closes "latch then idle".

# Fixed per-subtask event reward (NOT normalized by stage/subtask count).
# Each completed subtask is worth the same regardless of how many subtasks a
# task has, so the "I finished a subtask" signal is consistent across tasks.
FIXED_SUBTASK_REWARD = 0.3

# ---------------------------------------------------------------- optional reward cap
# Cap on the total intermediate reward (App. C.3, "Total reward capped at B").
# By default every subtask pays FIXED_SUBTASK_REWARD, so a task's total
# intermediate reward grows with its number of subtasks N. With
# INTERMEDIATE_BUDGET=B the per-subtask base becomes B / (N * (1 + BONUS_COEF)),
# so the total no longer depends on N. With the StructRL pacing curve
# (smooth_band, event reward <= 2 * base) the total is at most (4/3) * B, and
# (2/3) * B at demonstration pace. Unset (default): fixed per-subtask reward.
_ib = _os.environ.get("INTERMEDIATE_BUDGET", "").strip()
INTERMEDIATE_BUDGET = float(_ib) if _ib else None
if INTERMEDIATE_BUDGET is not None and INTERMEDIATE_BUDGET >= OUTCOME_REWARD:
    raise ValueError(
        f"INTERMEDIATE_BUDGET={INTERMEDIATE_BUDGET} must be < OUTCOME_REWARD="
        f"{OUTCOME_REWARD} (lambda_c)."
    )
if INTERMEDIATE_BUDGET is not None:
    import sys as _sys
    print(f"[timing_reward] INTERMEDIATE_BUDGET={INTERMEDIATE_BUDGET}", file=_sys.stderr, flush=True)


class TimingReward:
    """Per-env stateful reward computer.

    Construct once per env at episode start (or call `reset()` between
    episodes). Call `step(grouped_states, success)` each env step to get
    that step's reward and a debug dict for visualization.

    Attributes (read-only after step()):
      stage_idx       : int     — current stage being attempted
      stage_start_step: int     — step at which current stage became active
      latched         : dict[str, bool] — latched flag per subtask
      cumulative      : float   — running total reward this episode
      cumulative_outcome: float — total outcome contribution
      cumulative_progress: float — total progress / spike contribution
      cumulative_persistent: float — total persistent penalty (negative)
    """

    def __init__(self, stages: list[list[str]], rank_budgets: list[list[float]]):
        """
        Args:
          stages: list of stages, each stage is a list of subtask names.
                  Last stage may be retreat (gripper_*_far). All stages must
                  have at least 1 subtask.
          rank_budgets: list[len(stages)][len(stage_subtasks)] of float.
                  rank_budgets[k][r] is the demo-derived elapsed budget for
                  the r-th completion in stage k (rank 0 = first finisher).
        """
        assert len(stages) == len(rank_budgets), (
            f"stages ({len(stages)}) and rank_budgets ({len(rank_budgets)}) "
            f"must have same length"
        )
        for k, (s, b) in enumerate(zip(stages, rank_budgets)):
            if not s:
                continue  # empty stage is allowed (e.g., retreat with no fire)
            assert len(b) == len(s), (
                f"stage {k} has {len(s)} subtasks but {len(b)} rank budgets"
            )
            for budget in b:
                assert budget > 0, f"stage {k} has non-positive budget {budget}"

        self.stages = stages
        self.rank_budgets = rank_budgets
        self.K = len(stages)
        self._n_sub_total = sum(len(x) for x in stages)
        self.base_per_stage = 1.0 / max(self.K, 1)

        # all subtask names (across all stages)
        self.all_subtasks: set[str] = set()
        for s in stages:
            self.all_subtasks.update(s)

        self.reset()

    def reset(self):
        self.stage_idx = 0
        # D3 semantics: timer resets every SUBTASK completion (not every
        # stage). We keep the field name `subtask_start_step` to make this
        # explicit. The first subtask starts at t=0.
        self.subtask_start_step = 0
        self.latched: dict[str, bool] = {s: False for s in self.all_subtasks}
        self.outcome_fired = False
        self.t = -1                       # incremented at step()
        self.cumulative = 0.0
        self.cumulative_outcome = 0.0
        self.cumulative_progress = 0.0
        self.cumulative_persistent = 0.0

    def _sub_base(self, stage_idx: int) -> float:
        """Reward each subtask in stage_idx contributes if completed in time.

        Fixed per-subtask (FIXED_SUBTASK_REWARD), independent of how many
        stages/subtasks the task has, so the per-event magnitude is the same
        across all tasks.
        """
        if stage_idx >= self.K or not self.stages[stage_idx]:
            return 0.0
        if INTERMEDIATE_BUDGET is None:
            return FIXED_SUBTASK_REWARD  # default: fixed per-subtask reward
        if self._n_sub_total <= 0:
            return 0.0
        # INTERMEDIATE_BUDGET set: per-subtask base independent of N
        return INTERMEDIATE_BUDGET / (self._n_sub_total * (1.0 + BONUS_COEF))

    def _ratio_band_reward(self, ratio: float, sub_base: float) -> float:
        """Map (ratio, sub_base) to event reward."""
        if ratio <= 1.0:
            return sub_base * (1.0 + BONUS_COEF * (1.0 - ratio))
        elif ratio <= TOLERANCE:
            return sub_base
        else:
            penalty = (ratio - TOLERANCE) * sub_base / 2.0  # decay rate ~ sub_base
            # The decay rate above is: at ratio=2.5 (= 1.5+1), reward
            # decreases by sub_base/2. After ~3-4x demo time, hits the floor.
            return max(MIN_EVENT_FRAC * sub_base, sub_base - penalty)

    def step(self, grouped_states: dict, success: bool,
             sim_t: Optional[int] = None) -> dict:
        """Advance one step.

        Args:
          grouped_states: {"progress": OrderedDict, "quiescence": OrderedDict}
                          from structrl.subtask_checker.get_subtask_states.
                          Both groups merged into a single name->bool map.
          success: bool, env._check_success() this step
          sim_t:    optional REAL sim-step index. Budgets are in sim-step
                    units (extract_demo_timing walks every sim state), so the
                    timer MUST advance in sim steps too. Training calls step()
                    once per sim step and can leave this None (internal +=1).
                    Visualization renders only every steps_per_render sim
                    steps, so it MUST pass the true sim timestep here, else
                    elapsed/ratio are off by the render stride.

        Returns: dict with this step's reward + breakdown for visualization.
        """
        if sim_t is None:
            self.t += 1
        else:
            self.t = int(sim_t)

        # Merge progress + quiescence into a flat name->bool map
        flat: dict[str, bool] = {}
        for grp_name in ("progress", "quiescence"):
            for k, v in grouped_states.get(grp_name, {}).items():
                flat[k] = bool(v)

        elapsed = self.t - self.subtask_start_step
        events: list[dict] = []
        r_event_total = 0.0

        # Are we still in a real stage (not past the end)?
        in_active_stage = self.stage_idx < self.K
        current_stage_subs = (
            self.stages[self.stage_idx] if in_active_stage else []
        )
        # How many already latched in current stage?
        n_latched_current = sum(
            1 for s in current_stage_subs if self.latched.get(s, False)
        )

        # Check for new latches in CURRENT stage only (silent-skip defense:
        # subtasks in future stages are ignored). When a subtask latches,
        # use rank_budgets[stage][rank] (which under D3 extraction is the
        # GAP from previous subtask completion, not absolute) and reset
        # subtask_start_step immediately so the next subtask in the same
        # stage starts timing from this completion.
        for sub in current_stage_subs:
            if flat.get(sub, False) and not self.latched[sub]:
                self.latched[sub] = True
                rank = n_latched_current
                budget = self.rank_budgets[self.stage_idx][rank]
                # Use elapsed-since-last-subtask-completion (D3)
                gap_elapsed = self.t - self.subtask_start_step
                ratio = gap_elapsed / max(budget, 1.0)
                sub_base = self._sub_base(self.stage_idx)
                r = self._ratio_band_reward(ratio, sub_base)
                r_event_total += r
                events.append(dict(
                    subtask=sub, stage=self.stage_idx, rank=rank,
                    elapsed=gap_elapsed, budget=budget, ratio=ratio, r=r,
                ))
                n_latched_current += 1
                # CRITICAL D3: reset timer immediately after THIS subtask
                # completes, so next subtask measures from t.
                self.subtask_start_step = self.t

        # Did we complete the current stage this step?
        stage_just_completed = (
            in_active_stage
            and current_stage_subs
            and all(self.latched[s] for s in current_stage_subs)
            and n_latched_current == len(current_stage_subs)
        )
        # Stage advance (no special timer reset — subtask_start_step was
        # already set to t by the last subtask completion above).
        if stage_just_completed:
            self.stage_idx += 1
            # Skip empty stages (retreat that has no subtasks)
            while (self.stage_idx < self.K
                   and not self.stages[self.stage_idx]):
                self.stage_idx += 1

        # Persistent stuck penalty (per-subtask): if we've been waiting
        # for the next subtask for > TOLERANCE * its expected gap budget,
        # bleed -DECAY_PER_STEP each step.
        r_persistent = 0.0
        in_active_stage2 = self.stage_idx < self.K
        if in_active_stage2 and not stage_just_completed:
            cur_subs = self.stages[self.stage_idx]
            remaining = [s for s in cur_subs if not self.latched.get(s, False)]
            if remaining:
                n_already = len(cur_subs) - len(remaining)
                remaining_budgets = self.rank_budgets[self.stage_idx][n_already:]
                if remaining_budgets:
                    # Use NEXT rank's budget (not max), since we're waiting
                    # for the next subtask in completion order.
                    next_budget = remaining_budgets[0]
                    if (self.t - self.subtask_start_step) > TOLERANCE * next_budget:
                        r_persistent = -DECAY_PER_STEP
        # Finalize stuck penalty: all progress stages latched but outcome
        # not yet fired. Without this, a policy that latches every progress
        # flag (possibly via a transient spike, e.g. cup placed then knocked
        # over) can idle at its accumulated reward until timeout with zero
        # penalty. After a flat grace window from the last progress latch,
        # resume bleeding so "latch then idle" nets negative rather than free.
        elif not in_active_stage2 and not self.outcome_fired:
            if (self.t - self.subtask_start_step) > FINALIZE_GRACE_STEPS:
                r_persistent = -DECAY_PER_STEP

        # Outcome (one-shot)
        r_outcome = 0.0
        if success and not self.outcome_fired:
            r_outcome = OUTCOME_REWARD
            self.outcome_fired = True

        r_total = r_event_total + r_persistent + r_outcome
        self.cumulative += r_total
        self.cumulative_outcome += r_outcome
        self.cumulative_progress += r_event_total
        self.cumulative_persistent += r_persistent

        return dict(
            t=self.t,
            stage_idx_after=self.stage_idx,
            subtask_start_step=self.subtask_start_step,
            elapsed_in_stage=elapsed,
            r_event=r_event_total,
            r_persistent=r_persistent,
            r_outcome=r_outcome,
            r_total=r_total,
            cumulative=self.cumulative,
            cumulative_progress=self.cumulative_progress,
            cumulative_persistent=self.cumulative_persistent,
            cumulative_outcome=self.cumulative_outcome,
            events=events,
            stage_just_completed=stage_just_completed,
            latched=dict(self.latched),
        )
