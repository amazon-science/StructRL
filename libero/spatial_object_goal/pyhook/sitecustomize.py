# Auto-loaded lazy hook for LIBERO-Spatial/Object/Goal RL (stages, grasp predicate, MeanStd).
# =====================================================================
# Importing rlinf (and gr00t/torch for MeanStd) eagerly in every raylet-spawned
# python can exceed Ray's node-registration timeout, so this hook imports
# NOTHING heavy at startup: it wraps builtins.__import__ with a cheap
# sys.modules check and applies each patch the moment its target module is
# actually loaded, then unhooks itself.
#
# Gated on LIBERO3_HOOK=1; MeanStd part additionally on LIBERO3_GOAL_MEANSTD=1.
# Any failure is caught and printed, never raised.

import os
import sys

if os.environ.get("LIBERO3_HOOK") == "1":
    try:
        _HERE = os.path.dirname(os.path.abspath(__file__))
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)

        import builtins

        _targets = {"rlinf.envs.libero.dense_reward": "dense"}
        if os.environ.get("LIBERO3_GOAL_MEANSTD") == "1":
            # modality_config may be loaded via importlib (load_data_config
            # string path), which bypasses builtins.__import__ — so ALSO
            # trigger on its parent package (imported normally, heavy deps
            # already paid) and import the submodule ourselves in the patch.
            _targets["rlinf.models.embodiment.gr00t.modality_config"] = "meanstd"
            _targets["rlinf.models.embodiment.gr00t"] = "meanstd"

        _orig_import = builtins.__import__

        _in_patch = [False]

        def _maybe_patch():
            # Re-entrancy guard: imports INSIDE this function re-trigger the
            # __import__ hook -> unguarded recursion (RecursionError).
            if _in_patch[0]:
                return
            _in_patch[0] = True
            try:
                _do_patch()
            finally:
                _in_patch[0] = False

        def _do_patch():
            import libero3_stages  # light: only os + tuples until install_* called
            for mod_name, kind in list(_targets.items()):
                mod = sys.modules.get(mod_name)
                if mod is None or getattr(mod, "__libero3_patched__", False):
                    continue
                if kind == "dense":
                    libero3_stages.patch_dense_reward(mod)
                    mod.__libero3_patched__ = True
                    del _targets[mod_name]
                else:
                    import importlib
                    mc = importlib.import_module(
                        "rlinf.models.embodiment.gr00t.modality_config")
                    if not getattr(mc, "__libero3_patched__", False):
                        libero3_stages.patch_modality_config(mc)
                        mc.__libero3_patched__ = True
                    for k in [k for k, v in list(_targets.items()) if v == "meanstd"]:
                        del _targets[k]
            if not _targets:
                builtins.__import__ = _orig_import  # all done, unhook

        def _import_hook(name, *args, **kwargs):
            m = _orig_import(name, *args, **kwargs)
            try:
                for mod_name in _targets:
                    if mod_name in sys.modules:
                        _maybe_patch()
                        break
            except Exception as e:
                sys.stderr.write(f"[libero3 sitecustomize] patch FAILED: {e!r}\n")
            return m

        builtins.__import__ = _import_hook
    except Exception as _e:
        sys.stderr.write(f"[libero3 sitecustomize] init FAILED: {_e!r}\n")
        sys.stderr.flush()
