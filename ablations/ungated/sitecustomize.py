# Ungated-reward ablation hook (Fig. 3): removes structure-aware reward gating.
# ============================================================================
# Standalone file; it does not modify structrl or RLinf sources. Registered
# through PYTHONPATH auto-import (the same mechanism as ablations/beta_sweep):
# put this directory
# FIRST on PYTHONPATH in the sbatch launcher (parent shell AND remote srun
# worker shells) and CPython imports it at startup in every process.
#
# Ablation: remove the stage-precedence gate from the smooth_band dense reward.
# Baseline behavior: a subtask only latches/scores while its stage is the
# CURRENT stage, and a stage only opens once every subtask of all previous
# stages has latched (silent-skip defense). No-precedence behavior: EVERY
# unlatched subtask in EVERY stage is live at every step — the moment its flag
# is True it latches and scores immediately, regardless of order. Everything
# else is unchanged: continuous band r = cap*(sub_base)/(1+ratio) with the D3
# shared gap timer (reset on every latch), budgets looked up by the subtask's
# home stage + latch order within that home stage, one-shot +2.0 outcome, no
# persistent penalty (smooth_band semantics).
#
# Two independent gates (a job sets only its own):
#   STRUCTRL_NO_PRECEDENCE=1 -> patch structrl.smooth_band_reward
#                                .SmoothBandReward.step        (RoboCasa365)
#   LIBERO_NO_PRECEDENCE=1    -> patch rlinf.envs.libero.dense_reward
#                                .LiberoSmoothBandReward.step  (LIBERO-Long)
#
# STALE-SITE-PACKAGES TRAP (RoboCasa365): .venv-rlinf's site-packages contains a
# stale partial structrl. Insert PROJECT_ROOT at sys.path[0] and purge any
# stale-imported structrl before importing here, so venv.py's later import
# hits the patched fresh module (same defensive dance as venv.py itself).
#
# Defensive: any failure is caught and printed, never raised.

import os
import sys

PROJECT_ROOT = os.environ.get("STRUCTRL_PROJECT_ROOT", os.environ.get("PROJECT_ROOT", os.getcwd()))


def _install_rc365_noprec():
    if os.environ.get("STRUCTRL_NO_PRECEDENCE") != "1":
        return
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    stale = getattr(sys.modules.get("structrl", None), "__file__", "") or ""
    if "site-packages" in stale:
        for m in list(sys.modules):
            if m == "structrl" or m.startswith("structrl."):
                del sys.modules[m]
    import structrl.timing_reward as tr
    from structrl.smooth_band_reward import SmoothBandReward

    def _noprec_step(self, grouped_states, success, sim_t=None):
        if sim_t is None:
            self.t += 1
        else:
            self.t = int(sim_t)
        flat = {}
        for grp_name in ("progress", "quiescence"):
            for k, v in grouped_states.get(grp_name, {}).items():
                flat[k] = bool(v)
        elapsed = self.t - self.subtask_start_step
        events = []
        r_event_total = 0.0
        # NO-PRECEDENCE: scan ALL stages; any unlatched flag that is True
        # latches and scores now. Budget = home stage row, rank = latch order
        # within the home stage. D3 shared timer resets on every latch.
        for k_stage, subs in enumerate(self.stages):
            for sub in subs:
                if flat.get(sub, False) and not self.latched[sub]:
                    rank = sum(1 for s in subs if self.latched.get(s, False))
                    self.latched[sub] = True
                    row = self.rank_budgets[k_stage]
                    budget = row[min(rank, len(row) - 1)] if row else 50.0
                    gap_elapsed = self.t - self.subtask_start_step
                    ratio = gap_elapsed / max(budget, 1.0)
                    sub_base = self._sub_base(k_stage)
                    r = self._ratio_band_reward(ratio, sub_base)
                    r_event_total += r
                    events.append(dict(
                        subtask=sub, stage=k_stage, rank=rank,
                        elapsed=gap_elapsed, budget=budget, ratio=ratio, r=r,
                    ))
                    self.subtask_start_step = self.t
        # stage_idx: diagnostics-only here — count leading fully-latched
        # stages so stage_idx_after keeps its progress meaning in env logs.
        k = 0
        while k < self.K and all(
            self.latched.get(s, False) for s in self.stages[k]
        ):
            k += 1
        stage_just_completed = k > self.stage_idx
        self.stage_idx = k

        r_persistent = 0.0  # smooth_band: no persistent penalty
        r_outcome = 0.0
        if success and not self.outcome_fired:
            r_outcome = tr.OUTCOME_REWARD
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

    SmoothBandReward.step = _noprec_step
    # Also bind the SAME no-precedence step to the flat variant: its
    # _ratio_band_reward override returns a constant 0.3 (ratio ignored), so
    # this yields exactly "fixed 0.3 per latch, any order" — the fixed-reward
    # rung of the ablation ladder. Only the variant actually built by venv.py
    # (STRUCTRL_REWARD_VARIANT) is ever used, so double-binding is harmless.
    from structrl.flat_reward import FlatEventReward
    FlatEventReward.step = _noprec_step
    sys.stderr.write(
        "[noprec sitecustomize] RC365 SmoothBandReward.step + "
        "FlatEventReward.step -> NO-PRECEDENCE in pid %d\n" % os.getpid()
    )
    sys.stderr.flush()


def _install_libero_noprec():
    if os.environ.get("LIBERO_NO_PRECEDENCE") != "1":
        return
    import rlinf.envs.libero.dense_reward as dr

    def _noprec_step(self, true_now, success, sim_t=None):
        if sim_t is None:
            self.t += 1
        else:
            self.t = int(sim_t)
        r_event = 0.0
        events = []
        # NO-PRECEDENCE: every unlatched conjunct in every (post-drop) stage
        # is live; latch + score the moment it is true, any order.
        for k_stage, stage in enumerate(self.stages):
            n_latched_stage = sum(
                1 for c in stage if self.latched.get(c, False)
            )
            for c in stage:
                if (c in true_now) and not self.latched.get(c, False):
                    self.latched[c] = True
                    rank = n_latched_stage
                    budget = self._budget(k_stage, rank)
                    gap_elapsed = self.t - self.subtask_start_step
                    ratio = gap_elapsed / budget
                    r = self._ratio_band_reward(ratio)
                    r_event += r
                    events.append(dict(
                        subtask=c, stage=k_stage, rank=rank,
                        elapsed=gap_elapsed, budget=budget, ratio=ratio, r=r,
                    ))
                    n_latched_stage += 1
                    self.subtask_start_step = self.t
        k = 0
        while k < self.K and all(
            self.latched.get(c, False) for c in self.stages[k]
        ):
            k += 1
        self.stage_idx = k

        r_outcome = 0.0
        if success and not self.outcome_fired:
            r_outcome = dr.OUTCOME_REWARD
            self.outcome_fired = True

        r_total = r_event + r_outcome
        self.cumulative += r_total
        self.cumulative_progress += r_event
        self.cumulative_outcome += r_outcome
        n_latched_total = sum(1 for v in self.latched.values() if v)
        return dict(
            r_total=r_total,
            r_event=r_event,
            r_outcome=r_outcome,
            n_newly_latched=len(events),
            stage_idx_after=self.stage_idx,
            n_total=len(self.all_subtasks),
            n_latched=n_latched_total,
            cumulative=self.cumulative,
            cumulative_progress=self.cumulative_progress,
            cumulative_outcome=self.cumulative_outcome,
            events=events,
        )

    dr.LiberoSmoothBandReward.step = _noprec_step

    # Flat variant (LiberoDenseReward): different signature (no sim_t) and a
    # constant 0.3 per latch. Same no-precedence semantics: every unlatched
    # conjunct in every (post-drop) stage latches the moment it is true.
    def _noprec_flat_step(self, true_now, success):
        n_newly_latched = 0
        for stage in self.stages:
            for c in stage:
                if (c in true_now) and not self.latched.get(c, False):
                    self.latched[c] = True
                    n_newly_latched += 1
        k = 0
        while k < self.K and all(
            self.latched.get(c, False) for c in self.stages[k]
        ):
            k += 1
        self.stage_idx = k

        r_event = dr.FIXED_SUBTASK_REWARD * n_newly_latched
        r_outcome = 0.0
        if success and not self.outcome_fired:
            r_outcome = dr.OUTCOME_REWARD
            self.outcome_fired = True
        r_total = r_event + r_outcome
        self.cumulative += r_total
        self.cumulative_progress += r_event
        self.cumulative_outcome += r_outcome
        n_latched_total = sum(1 for v in self.latched.values() if v)
        return dict(
            r_total=r_total,
            r_event=r_event,
            r_outcome=r_outcome,
            n_newly_latched=n_newly_latched,
            stage_idx_after=self.stage_idx,
            n_total=len(self.all_subtasks),
            n_latched=n_latched_total,
            cumulative=self.cumulative,
            cumulative_progress=self.cumulative_progress,
            cumulative_outcome=self.cumulative_outcome,
        )

    dr.LiberoDenseReward.step = _noprec_flat_step
    sys.stderr.write(
        "[noprec sitecustomize] LIBERO LiberoSmoothBandReward.step + "
        "LiberoDenseReward.step -> NO-PRECEDENCE in pid %d\n" % os.getpid()
    )
    sys.stderr.flush()


try:
    _install_rc365_noprec()
except Exception as _e:  # never break interpreter startup
    sys.stderr.write("[noprec sitecustomize] RC365 FAILED: %r\n" % (_e,))
    sys.stderr.flush()
try:
    _install_libero_noprec()
except Exception as _e:
    sys.stderr.write("[noprec sitecustomize] LIBERO FAILED: %r\n" % (_e,))
    sys.stderr.flush()
