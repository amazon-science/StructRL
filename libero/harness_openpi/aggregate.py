# Aggregate per-task stats_task*.json into a LIBERO-Long (10-task) summary.
# Usage: python aggregate.py <dir containing stats_task0.json .. stats_task9.json>
import glob
import json
import sys

d = sys.argv[1]
rows = []
for p in sorted(glob.glob(f"{d}/stats_task*.json")):
    with open(p) as f:
        rows.append(json.load(f))

rows.sort(key=lambda r: r["task_id"])
tot_ep = sum(r["num_episodes"] for r in rows)
tot_su = sum(r["num_successes"] for r in rows)
print(f"\n===== {d} =====")
print(f"{'tid':>3} {'succ/ep':>8} {'SR':>7}  task")
for r in rows:
    print(
        f"{r['task_id']:>3} {r['num_successes']:>3}/{r['num_episodes']:<4} "
        f"{r['success_rate']*100:>6.1f}%  {r['task_description']}"
    )
avg = tot_su / tot_ep if tot_ep else 0.0
print(f"\nTASKS: {len(rows)}/10   TOTAL: {tot_su}/{tot_ep} = {avg*100:.2f}%")
if rows:
    print(f"config={rows[0]['config_name']} chunk={rows[0]['action_chunk']} num_steps={rows[0]['num_steps']}")
