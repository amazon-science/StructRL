"""Loaded at interpreter startup (earlier than LIBERO get_task_init_states).
Patch torch.load to weights_only=False: LIBERO init_files are numpy-pickled
trusted local assets, and torch>=2.6 defaults weights_only=True which rejects
numpy.core.multiarray._reconstruct.
"""
try:
    import torch as _torch
    _orig_load = _torch.load
    def _trusting_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)
    _trusting_load._libero_trusting = True
    _torch.load = _trusting_load
except Exception:
    pass
