# Tiny helper whose sole job is to trigger registration of the "grpo_diag"
# advantage function in the current process. Brand-new file; edits nothing.
#
# It ensures the directory containing grpo_diag_adv.py is importable regardless
# of how the interpreter was launched, then imports it (which runs the
# @register_advantage("grpo_diag") decorator).
#
# Usage:
#   import diag_register            # if this dir is on sys.path
#   diag_register.register()        # idempotent; safe to call multiple times
# or simply importing this module already performs the registration.

import os
import sys


def register():
    """Import grpo_diag_adv so @register_advantage('grpo_diag') runs. Idempotent."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import grpo_diag_adv  # noqa: F401  -- triggers @register_advantage("grpo_diag")
    return grpo_diag_adv


# Perform registration on import as well, so a bare `import diag_register` works.
register()
