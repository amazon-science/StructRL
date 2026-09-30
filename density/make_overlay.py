#!/usr/bin/env python3
"""Build the structrl/ package for one decomposition variant (Sec. 4.5, App. B.3 and C).

Each variant replaces the stage configuration (and, for the denser ladders, the predicate
checker) of the default package, and brings its own reference durations T_d. The overlay is
a copy of this repository's structrl/ with those files swapped in; the reward code is shared.

    python density/make_overlay.py n14 --out /tmp/structrl_n14
    eval "$(python density/make_overlay.py n14 --out /tmp/structrl_n14 --print-env)"

--print-env prints the exports that point the training launchers at the overlay.
"""
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# variant -> stage config, checker (None: default), reference durations, extra environment
VARIANTS = {
    "n1": ("stage_configs/n1.py", None, "timing/n1.json", {}),
    "n5": ("stage_configs/n5.py", "checkers/subtask_checker_n5.py", "timing/n5.json", {}),
    "n10": ("stage_configs/n10.py", "checkers/subtask_checker_n29.py", "timing/n10.json", {}),
    "n14": ("stage_configs/n14.py", "checkers/subtask_checker_n29.py", "timing/n14.json", {}),
    "n29": ("stage_configs/n29.py", "checkers/subtask_checker_n29.py", "timing/n29.json", {}),
    "n52": ("stage_configs/n52.py", "checkers/subtask_checker_n52.py", "timing/n52.json", {}),
    "qwen": ("stage_configs/qwen.py", "checkers/subtask_checker_n5.py", "timing/qwen.json", {}),
    # App. C.3 controls
    "n5_earlier": ("stage_configs/n5_earlier.py", "checkers/subtask_checker_n29.py", "timing/n5_earlier.json", {}),
    "n14_budget": ("stage_configs/n14.py", "checkers/subtask_checker_n29.py", "timing/n14.json",
                   {"INTERMEDIATE_BUDGET": "1.4"}),
    # App. C.4 control: default decomposition, T_d replaced by H / N
    "uniform_pacing": (None, None, "timing/uniform_pacing.json", {}),
}


def build(variant, out):
    stage, checker, _, _ = VARIANTS[variant]
    pkg = out / "structrl"
    if pkg.exists():
        shutil.rmtree(pkg)
    shutil.copytree(REPO / "structrl", pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if stage:
        shutil.copy2(HERE / stage, pkg / "stage_config.py")
    if checker:
        shutil.copy2(HERE / checker, pkg / "subtask_checker.py")
    return pkg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("variant", choices=sorted(VARIANTS))
    ap.add_argument("--out", required=True, type=Path, help="overlay root; structrl/ is created inside")
    ap.add_argument("--print-env", action="store_true", help="print shell exports for the launcher")
    args = ap.parse_args()
    out = args.out.resolve()
    build(args.variant, out)
    _, _, timing, extra = VARIANTS[args.variant]
    env = {"STRUCTRL_PROJECT_ROOT": str(out), "STRUCTRL_TIMING_JSON": str(HERE / timing), **extra}
    if args.print_env:
        for k, v in env.items():
            print(f"export {k}={v}")
    else:
        print(f"built {out / 'structrl'} for variant {args.variant}", file=sys.stderr)
        for k, v in env.items():
            print(f"  {k}={v}", file=sys.stderr)


if __name__ == "__main__":
    main()
