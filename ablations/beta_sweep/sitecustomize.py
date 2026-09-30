# beta override hook for the beta sensitivity sweep (App. D.3, Fig. 8).
# =====================================================================
# BRAND-NEW standalone file — does NOT modify any structrl/RLinf file.
# Registered through PYTHONPATH auto-import: CPython imports a top-level
# module named `sitecustomize` at interpreter startup if one is found on
# sys.path, so putting this directory FIRST on PYTHONPATH (done by the
# beta-sweep launchers, parent shell AND remote srun worker shells)
# applies the patch in every process on every node — the exact same
# zero-source-edit mechanism as ablations/ungated.
#
# No-op unless STRUCTRL_BAND_CAP is set. When set (e.g. 0.2 / 2.0) it
# rebinds SmoothBandReward._ratio_band_reward to the absolute-cap curve
#     r_event = BAND_CAP / (1 + ratio)
# which is exactly the functional form of smooth_band_cap10 (BAND_CAP=1.0)
# generalized to any cap; anchor(ratio=1) = BAND_CAP/2. The patch lands on
# SmoothBandReward itself, so SmoothTimingReward (smooth_timing) and
# SmoothBandCap10Reward (which overrides its own _ratio_band_reward) are
# untouched, as are all other reward variants.
#
# STALE-SITE-PACKAGES TRAP: .venv-rlinf's site-packages contains a stale
# partial copy of structrl. PROJECT_ROOT must be inserted at sys.path[0]
# BEFORE importing structrl here, otherwise this hook would patch the stale
# copy while venv.py later imports the fresh one -> silent default cap 0.6.
# With the fresh module imported (and cached in sys.modules) at startup,
# venv.py's later import gets the patched class.
#
# Defensive: any failure is caught and printed, never raised, so it can never
# break interpreter startup.

import os
import sys

PROJECT_ROOT = os.environ.get("STRUCTRL_PROJECT_ROOT", os.environ.get("PROJECT_ROOT", os.getcwd()))


def _install_band_cap():
    cap = os.environ.get("STRUCTRL_BAND_CAP")
    if not cap:
        return
    cap_f = float(cap)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from structrl.smooth_band_reward import SmoothBandReward

    def _ratio_band_reward(self, ratio, sub_base, _cap=cap_f):
        # Absolute cap; sub_base intentionally unused — same convention as
        # smooth_band_cap10_reward.BAND_CAP. anchor(ratio=1) = _cap/2.
        return _cap / (1.0 + max(ratio, 0.0))

    SmoothBandReward._ratio_band_reward = _ratio_band_reward
    sys.stderr.write(
        "[beta_sweep sitecustomize] SmoothBandReward band cap -> %.4f "
        "(anchor %.4f) in pid %d\n" % (cap_f, cap_f / 2.0, os.getpid())
    )
    sys.stderr.flush()


try:
    _install_band_cap()
except Exception as _e:  # never break interpreter startup
    sys.stderr.write("[beta_sweep sitecustomize] FAILED: %r\n" % (_e,))
    sys.stderr.flush()
