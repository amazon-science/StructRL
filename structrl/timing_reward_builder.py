"""Shared builder for TimingReward instances from a demo-timing JSON.

Both the offline video overlay (visualize_timing_reward.py) and the online PPO
env subprocess (RLinf rlinf/envs/robocasa365/venv.py) must construct the exact
same TimingReward for a given task, so the reward the policy is trained on is
byte-for-byte the reward shown in the diagnostic videos. This module is the
single source of truth for that construction.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from structrl.stage_config import get_stage_def
from structrl.timing_reward import TimingReward


# Default: the canonical timing JSON shipped next to this file; override with
# STRUCTRL_TIMING_JSON (absolute path) for custom demo-timing budgets.
_DEFAULT_TIMING_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "demo_timing_composite_seen.json"
)


def load_timing_data(path: Optional[str] = None) -> dict:
    """Load the demo-timing JSON (rank_budgets per task).

    Defensive: the sbatch heredoc + eval pipeline can leave the env var as
    an unexpanded literal that doesn't resolve to a real file. If the env
    var is set but the path doesn't exist, fall back to the bundled default
    so the online PPO env doesn't silently lose its dense reward.
    """
    if path is None:
        path = os.environ.get("STRUCTRL_TIMING_JSON", _DEFAULT_TIMING_JSON)
    if not os.path.isfile(path):
        path = _DEFAULT_TIMING_JSON
    with open(path) as f:
        return json.load(f)


def build_timing_reward(task_name: str, timing_data: dict) -> TimingReward:
    """Construct a TimingReward for `task_name` from parsed timing data.

    Mirrors the budget assembly in visualize_timing_reward._get_or_init_reward:
    rank_budgets[k][r] = median demo gap-to-r-th-completion in stage k. Missing
    ranks in the JSON fall back to max(budgets) (or 100) so we never crash.
    """
    if task_name not in timing_data:
        raise RuntimeError(
            f"Task {task_name} not in timing JSON: {list(timing_data.keys())}"
        )
    stages = get_stage_def(task_name)
    rb_raw = timing_data[task_name]["rank_budgets"]
    rank_budgets: list[list[float]] = []
    for k_idx, stage_subs in enumerate(stages):
        stage_data = rb_raw.get(str(k_idx), {})
        ranks_sorted = sorted(stage_data.items(), key=lambda kv: int(kv[0]))
        budgets = [float(v["median"]) for _, v in ranks_sorted]
        if len(budgets) < len(stage_subs):
            fallback = 100.0 if not budgets else max(budgets)
            while len(budgets) < len(stage_subs):
                budgets.append(fallback)
        rank_budgets.append(budgets[: len(stage_subs)])
    return TimingReward(stages=stages, rank_budgets=rank_budgets)
