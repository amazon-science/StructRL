# Diagnostic GRPO advantage (adv_type=grpo_diag) used by the GR00T-N1.5 GRPO runs.
# =====================================================================
# This is a BRAND-NEW, standalone module. It does NOT modify any RLinf file.
# It registers a new advantage type "grpo_diag" that is NUMERICALLY IDENTICAL
# to the built-in "grpo" (rlinf/algorithms/advantages.py:compute_grpo_advantages)
# but ADDS per-group instrumentation: it appends a JSON line per call describing
# the group-score distribution and the fraction of zero-variance ("dead") groups.
#
# WHY behavior is kept identical (no real filtering):
#   The GRPO advantage is (r - group_mean) / (group_std + 1e-6). For a
#   zero-variance ("dead") group every member equals the group mean, so the
#   numerator (r - group_mean) is exactly 0 -> advantage = 0 / (0 + 1e-6) = 0
#   automatically. So dead groups already contribute 0 to the gradient numerator
#   WITHOUT any explicit filtering. Hence this experiment does NOT change the
#   math; it ONLY measures the fraction of dead groups in each batch.
#
# LIMITATION (documented, not a bug):
#   The advantage fn signature returns exactly (advantages, None). It CANNOT
#   return a modified loss_mask, and the caller
#   (rlinf/algorithms/registry.py:calculate_adv_and_returns) does not read back
#   any mutated loss_mask -> so we CANNOT change the loss denominator
#   (masked_mean denominator = loss_mask.sum()) from inside this function. A
#   dead group left in-batch with advantage=0 but loss_mask=1 still inflates the
#   denominator (dilutes the gradient). Zeroing loss_mask in-place here is NOT a
#   reliable fix because the tensor may be a view / may be reused downstream, and
#   the true "filtering" in this repo happens elsewhere (algorithm.filter_rewards
#   + rewards_lower_bound/upper_bound). We therefore keep behavior identical to
#   "grpo" and only zero the dead-group ADVANTAGES (which the original already
#   does implicitly via the 0/1e-6 numerator), so the numerator contribution of
#   dead groups is 0 -- matching the built-in "grpo" bit-for-bit.

import json
import os

import torch

from rlinf.algorithms.registry import register_advantage

# Module-level, per-process incrementing call counter for the dump.
_GRPO_DIAG_CALL_IDX = 0


def _dump_group_stats(grouped_rewards: torch.Tensor, dead_count: int, num_groups: int):
    """Append one JSONL line describing the per-group score distribution.

    Wrapped in a broad try/except so instrumentation can NEVER crash training.
    grouped_rewards has shape [num_groups, group_size] (per-trajectory scores).
    """
    global _GRPO_DIAG_CALL_IDX
    call_idx = _GRPO_DIAG_CALL_IDX
    _GRPO_DIAG_CALL_IDX += 1

    try:
        dump_path = os.environ.get("GRPO_DIAG_DUMP", "/tmp/grpo_diag_dump.jsonl")

        gr = grouped_rewards.detach().to(torch.float32).cpu()
        # Per-group stats. std uses default (unbiased) to match torch.std used in
        # the advantage computation; this is purely for reporting.
        g_max = gr.max(dim=-1).values
        g_mean = gr.mean(dim=-1)
        g_min = gr.min(dim=-1).values
        g_std = gr.std(dim=-1)

        per_group = [
            {
                "max": float(g_max[i].item()),
                "mean": float(g_mean[i].item()),
                "min": float(g_min[i].item()),
                "std": float(g_std[i].item()),
            }
            for i in range(gr.shape[0])
        ]

        record = {
            "call_idx": call_idx,
            "pid": os.getpid(),
            "num_groups": int(num_groups),
            "dead_groups": int(dead_count),
            "dead_fraction": (float(dead_count) / float(num_groups))
            if num_groups > 0
            else 0.0,
            "group_score_mean_over_groups": float(gr.mean().item()),
            "per_group": per_group,
        }

        with open(dump_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:  # never crash training on a diagnostics failure
        try:
            print(f"[grpo_diag] dump failed (ignored): {e!r}")
        except Exception:
            pass


@register_advantage("grpo_diag")
def compute_grpo_diag_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    **kwargs,
):
    """Compute GRPO advantages (IDENTICAL math to "grpo") + dump group stats.

    Args:
        rewards (torch.Tensor): Per-trajectory SCORES. Shape: [num_groups, group_size]
            (calculate_scores() already summed each trajectory and reshaped to
            [-1, group_size] before this fn is called).
        loss_mask (torch.Tensor): Loss mask for valid entries. In the embodied
            path this is [n_steps, bsz] with bsz = num_groups*group_size, i.e. the
            trajectory axis is dim 1. advantages.view(1, -1) broadcasts over that
            trajectory axis.
        group_size (int): Group size for advantage computation.

    Returns:
        (advantages, None)
    """
    # ---- EXACT copy of compute_grpo_advantages math (advantages.py:107-121) ----
    grouped_rewards = rewards.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )

    advantages = grouped_rewards - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask
    # ---------------------------------------------------------------------------

    # ADDITION A (zero-variance identification): dead groups = per-group std==0.
    # Use grouped_reward_std (already expanded to [num_groups, group_size]); take
    # column 0 to get the per-group std [num_groups]. A dead group already has
    # advantage==0 (numerator r-mean==0), so no math change is needed -- we only
    # MEASURE it. See module docstring for why we cannot change loss_mask here.
    num_groups = grouped_rewards.shape[0]
    per_group_std = grouped_reward_std[:, 0]  # [num_groups]
    dead_mask_per_group = per_group_std < 1e-8  # [num_groups] bool
    dead_count = int(dead_mask_per_group.sum().item())

    # (No-op behaviorally, kept explicit for auditability: dead groups already
    #  have advantage 0 via the numerator. This does NOT alter any live group.)
    # advantages stays exactly as the built-in "grpo" would produce.

    # ADDITION B (the DUMP -- main deliverable):
    _dump_group_stats(grouped_rewards, dead_count, num_groups)

    return advantages, None
