"""Decomposer-sensitivity study (App. B.3): LLM-authored stage/subtask decomposition.

A local Qwen3.5-9B receives ONLY (a) the original composite task
language command and (b) our decomposition requirements. It proposes the
stage/subtask split itself. No checker vocabulary, no existing splits, no demo
data are shown to the model.

Output: qwen3.5-9b_output.json  {task: {"stages": [[subtask,...],...], "raw": str}}
The grounded stage configuration (App. B.2 rules) is density/stage_configs/qwen.py.
Usage: python density/decomposer/llm_decompose.py   (DECOMPOSER_MODEL overrides the model id)
"""
import json
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("DECOMPOSER_MODEL", "Qwen/Qwen3.5-9B")

SYSTEM = """You are a robot task planner. Decompose a kitchen manipulation task \
(performed by a single-arm mobile robot) into stages of subtasks.

Requirements:
1. Output STAGES in strict execution order: stage k+1 can only start after every
   subtask in stage k is done.
2. Subtasks WITHIN one stage may be completed in any order (parallel set).
3. Each subtask must be ONE atomic, physically checkable manipulation event,
   phrased as a short verb phrase in this controlled form:
   - "grasp <object>"
   - "place <object> in/on <receptacle or location>"
   - "open <fixture>" / "close <fixture>"
   - "turn on <fixture>" / "turn off <fixture>"
   - "press <button>"
   - "<activity> for a while" for continuous activities (stirring, washing,
     scrubbing), optionally split into progressive milestones.
4. Include intermediate manipulation events (like grasping an object before
   placing it), not just final outcomes.
5. Do NOT include a final "task finished" subtask, and do NOT include robot
   retreat/release-and-back-away steps.
6. Output ONLY a JSON object: {"stages": [["subtask", ...], ...]} - no prose.
"""

USER_TMPL = 'Task command: "{cmd}"\n\nDecompose this task now.'


def main():
    cmds = json.load(open(os.path.join(HERE, "task_commands.json")))
    print(f"loading {MODEL_PATH}", flush=True)
    processor = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16).to("cuda:0")
    model.eval()

    out = {}
    for task, cmd in cmds.items():
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(cmd=cmd)},
        ]
        try:
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
                enable_thinking=False).to("cuda:0")
        except TypeError:
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=3072, do_sample=False)
        text = processor.decode(
            gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        cands = re.findall(r"\{[^{}]*\"stages\"[\s\S]*?\]\s*\}", text)
        m = None
        if cands:
            class _M:
                def __init__(s, t): s._t = t
                def group(s, i): return s._t
            m = _M(cands[-1])
        stages = None
        if m:
            try:
                stages = json.loads(m.group(0))["stages"]
            except Exception as e:
                print(f"[{task}] JSON parse failed: {e}", flush=True)
        out[task] = {"stages": stages, "raw": text}
        print(f"\n===== {task} =====\ncmd: {cmd}\nstages: {json.dumps(stages, ensure_ascii=False)}", flush=True)

    with open(os.path.join(HERE, "qwen3.5-9b_output.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nwrote qwen3.5-9b_output.json", flush=True)


if __name__ == "__main__":
    main()
