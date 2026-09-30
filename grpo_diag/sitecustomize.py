# Auto-loaded registration hook for the "grpo_diag" advantage function.
# =====================================================================
# CPython automatically imports a top-level module named `sitecustomize` at
# interpreter startup IF one is found on sys.path. RLinf spawns actor workers in
# fresh subprocesses (mp.set_start_method("spawn")), so registration must happen
# at import time in EVERY process. By putting this diag/ directory on PYTHONPATH
# (done by the sbatch launcher), this file is auto-imported in the parent AND in
# every spawned worker -- WITHOUT editing any RLinf file.
#
# It is intentionally defensive: any failure is caught and printed, never raised,
# so it can never break interpreter startup or training.

import os
import sys


def _register_grpo_diag():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    # Importing grpo_diag_adv runs @register_advantage("grpo_diag").
    import grpo_diag_adv  # noqa: F401
    sys.stderr.write("[sitecustomize] registered grpo_diag\n")
    sys.stderr.flush()


try:
    _register_grpo_diag()
except Exception as e:  # never crash interpreter startup
    try:
        sys.stderr.write(f"[sitecustomize] grpo_diag registration FAILED (ignored): {e!r}\n")
        sys.stderr.flush()
    except Exception:
        pass
