"""Wrapper around gr00t.eval.simulation.run_evaluation for our checkpoint.

Workflow:
  1. Start inference server in a background subprocess (loads GR00T policy on GPU)
  2. Call run_evaluation() — uses the official MultiStepWrapper + VideoRecordingWrapper
     against the gymnasium_groot env, talks to the server over ZMQ, writes mp4s
     and returns success flags
  3. Tear down server

Usage:
  python -u structrl/groot_official_eval.py \
    --task PickPlaceCabinetToCounter \
    --n_episodes 2 --max_steps 400 \
    --video_dir eval_out
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# Move cwd out of project root so the local `robocasa/` source dir does not
# get merged with the editable-installed package via namespace-package logic.
# Also DO NOT put PROJECT_ROOT on sys.path — structrl is installed as a real
# package under .venv-groot/lib/python3.11/site-packages/structrl.
os.chdir("/tmp")
sys.path = [p for p in sys.path if not (
    (Path(p or ".").resolve() / "robocasa").is_dir()
    and not (Path(p or ".").resolve() / "robocasa" / "__init__.py").is_file()
)]

import robocasa  # noqa: E402,F401
import inspect  # noqa: E402
import numpy as np  # noqa: E402
print(f"[debug] robocasa loaded from: {inspect.getfile(robocasa)}", flush=True)

# Monkey-patch RoboCasaEnv.render to return a 3-camera horizontal strip
# (left agentview | right agentview | wrist) instead of just the first cam.
# This is what VideoRecordingWrapper grabs each step.
from robocasa.utils.gym_utils.gymnasium_basic import RoboCasaEnv  # noqa: E402

_RECORD_CAMS_3 = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]

def _three_cam_render(self):
    rs = self.env  # underlying robosuite env
    imgs = []
    for cam in _RECORD_CAMS_3:
        img = rs.sim.render(camera_name=cam, width=256, height=256)
        imgs.append(np.flipud(img).copy())
    return np.concatenate(imgs, axis=1)  # (256, 768, 3)

RoboCasaEnv.render = _three_cam_render

from gr00t.eval.simulation import run_evaluation  # noqa: E402
import gr00t.eval.simulation as _gr00t_sim  # noqa: E402
import gymnasium as _gym  # noqa: E402

# eval-split overrides applied per-process before run_evaluation is called.
# Replaces gr00t's hardcoded `gym.make(env_name, split=config.split, ...)`
# with explicit obj_instance_split + layout_and_style_ids kwargs that our
# RoboCasaEnv (via create_env_robosuite) actually consumes.
_EVAL_KWARGS_OVERRIDE: dict = {}

def _patched_create_single_env(config, idx):
    from gr00t.eval.wrappers.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
    from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
    from pathlib import Path as _P
    extra_kwargs = dict(_EVAL_KWARGS_OVERRIDE)
    env = _gym.make(config.env_name, enable_render=True, **extra_kwargs)
    if config.video.video_dir is not None:
        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env, video_recorder, video_dir=_P(config.video.video_dir),
            steps_per_render=config.video.steps_per_render,
        )
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
    )
    return env

_gr00t_sim._create_single_env = _patched_create_single_env

CKPT_PATH = os.environ.get("STRUCTRL_DEFAULT_CKPT", "ckpts/gr00t_n1-5/multitask_learning/checkpoint-120000")
DATA_CONFIG = "structrl.robocasa_data_config:RobocasaPandaOmronConfig"
EMBODIMENT_TAG = "new_embodiment"
DATASET_BY_TASK = {
    "PickPlaceCabinetToCounter": Path(
        os.environ.get("RC365_DATASET_ROOT", "datasets/v1.0") + "/pretrain/atomic"
        "/PickPlaceCabinetToCounter/20250819/lerobot"
    ),
}


def free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_inference_server(port: int, ckpt_path: str = CKPT_PATH) -> subprocess.Popen:
    cmd = [
        sys.executable, "-u",
        os.path.join(os.environ.get("GROOT_DIR", "Isaac-GR00T"), "scripts/inference_service.py"),
        "--server",
        f"--model_path={ckpt_path}",
        f"--data_config={DATA_CONFIG}",
        f"--embodiment_tag={EMBODIMENT_TAG}",
        "--denoising_steps=4",
        f"--port={port}",
        "--host=localhost",
    ]
    print(f"Starting inference server: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    # structrl is installed as a real package in venv-groot site-packages;
    # cwd=/tmp keeps the local robocasa/ source dir from shadowing.
    env.pop("PYTHONPATH", None)
    return subprocess.Popen(cmd, env=env, cwd="/tmp", stdout=sys.stdout, stderr=sys.stderr)


def wait_for_server(port: int, timeout: int = 600) -> bool:
    """Probe the port until the server starts accepting connections."""
    deadline = time.time() + timeout
    last_heartbeat = 0.0
    start = time.time()
    while time.time() < deadline:
        try:
            s = socket.create_connection(("localhost", port), timeout=2)
            s.close()
            print(f"[main] Server on port {port} accepted a connection after {time.time()-start:.1f}s", flush=True)
            return True
        except OSError:
            now = time.time()
            if now - last_heartbeat > 5:
                print(f"[main] still waiting for server on port {port}... ({now-start:.0f}s elapsed)", flush=True)
                last_heartbeat = now
            time.sleep(2)
    return False


def stitch_with_demo(groot_video: Path, task: str, demo_episode: int, out_path: Path):
    """Compose top=demo (3-cam strip) + bottom=GR00T rollout (3-cam strip)."""
    import imageio.v2 as iio2
    import imageio.v3 as iio

    dataset_root = DATASET_BY_TASK.get(task)
    if dataset_root is None or not dataset_root.exists():
        print(f"[stitch] no demo dataset registered for task {task}, skipping")
        return

    # Load demo 3-cam strip
    cam_dirs = [
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_agentview_right",
        "observation.images.robot0_eye_in_hand",
    ]
    cams_frames = []
    for cd in cam_dirs:
        path = dataset_root / "videos" / "chunk-000" / cd / f"episode_{demo_episode:06d}.mp4"
        if not path.exists():
            print(f"[stitch] demo video missing: {path}")
            return
        cams_frames.append(iio.imread(path))
    Td = min(f.shape[0] for f in cams_frames)
    cams_frames = [f[:Td] for f in cams_frames]
    demo_strip = np.concatenate(cams_frames, axis=2)  # (T, 256, 768, 3)

    # Load groot rollout strip
    groot_frames = iio.imread(groot_video)  # (T, H, W, 3) — already 256x768
    Tg = groot_frames.shape[0]
    print(f"[stitch] demo T={Td}, groot T={Tg}")

    # Pad shorter strip with last frame
    T = max(Td, Tg)
    H, W, _ = demo_strip.shape[1:]
    out = np.full((T, 2 * H + 8, W, 3), 255, dtype=np.uint8)
    out[:Td, :H] = demo_strip
    out[Td:, :H] = demo_strip[-1:]
    out[:Tg, H + 8:H + 8 + H] = groot_frames
    out[Tg:, H + 8:H + 8 + H] = groot_frames[-1:]

    writer = iio2.get_writer(str(out_path), fps=20, codec="h264", quality=8)
    for f in out:
        writer.append_data(f)
    writer.close()
    print(f"[stitch] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None,
                    help="Single task to eval. Mutually exclusive with --task_list.")
    ap.add_argument("--task_list", default=None,
                    help="Comma-separated list of tasks (loops over them, reuses server).")
    ap.add_argument("--n_episodes", type=int, default=2)
    ap.add_argument("--n_envs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--n_action_steps", type=int, default=16,
                    help="how many steps to execute from each predicted action chunk before re-querying. "
                         "GR00T action_horizon=16: execute the FULL predicted chunk (matches train/deploy).")
    ap.add_argument("--video_dir", default="eval_out")
    ap.add_argument("--compare_with_demo", action="store_true",
                    help="After rollout, stitch each GR00T video on top of the matching demo video")
    ap.add_argument("--summary_csv", default=None,
                    help="Append per-task results here (csv: task,successes,n,rate)")
    ap.add_argument("--ckpt_path", default=CKPT_PATH,
                    help="GR00T model checkpoint dir (defaults to multitask_learning/checkpoint-120000)")
    ap.add_argument("--eval_split", default="target", choices=["ours", "target"],
                    help="target (DEFAULT, official paper Table 2 protocol): obj_instance_split=target, "
                         "layouts (1,1)..(10,10). ours: structrl non-official "
                         "(obj_instance_split=pretrain, 5 GR00T layout pairs).")
    args = ap.parse_args()

    if args.eval_split == "target":
        _EVAL_KWARGS_OVERRIDE.update(
            obj_instance_split="target",
            layout_and_style_ids=[(i, i) for i in range(1, 11)],
        )
        print(f"[eval] split=target -> obj_instance_split=target, 10 layout pairs", flush=True)
    else:
        print(f"[eval] split=ours -> RoboCasaEnv defaults (obj_instance_split=pretrain, 5 layout pairs)", flush=True)

    if args.task and args.task_list:
        ap.error("--task and --task_list are mutually exclusive")
    if not args.task and not args.task_list:
        args.task = "PickPlaceCabinetToCounter"

    # Parse task_list, each item is "task_name" or "task_name:max_steps".
    # If max_steps suffix missing, fall back to global --max_steps.
    raw_items = [args.task] if args.task else [t.strip() for t in args.task_list.split(",") if t.strip()]
    tasks = []  # list of (task_name, max_steps)
    for item in raw_items:
        if ":" in item:
            n, ms = item.split(":", 1)
            tasks.append((n.strip(), int(ms)))
        else:
            tasks.append((item, args.max_steps))

    base_video_dir = Path(args.video_dir)
    base_video_dir.mkdir(parents=True, exist_ok=True)

    port = free_port()
    server_proc = start_inference_server(port, ckpt_path=args.ckpt_path)

    summary_rows = []
    try:
        if not wait_for_server(port):
            print("ERROR: server never started", file=sys.stderr)
            return

        for task, task_max_steps in tasks:
            print(f"\n========== TASK: {task} (max_steps={task_max_steps}) ==========", flush=True)
            task_video_dir = base_video_dir / task
            task_video_dir.mkdir(parents=True, exist_ok=True)

            try:
                env_name = f"robocasa_panda_omron/{task}_PandaOmron_Env"
                results = run_evaluation(
                    env_name=env_name,
                    host="localhost",
                    port=port,
                    video_dir=str(task_video_dir),
                    n_episodes=args.n_episodes,
                    n_envs=args.n_envs,
                    n_action_steps=args.n_action_steps,
                    max_episode_steps=task_max_steps,
                )
                env_name_out, successes = results
                rate = (sum(successes) / len(successes)) if successes else 0.0
                print(f"  [{task}] {sum(successes)}/{len(successes)} = {rate:.2%}", flush=True)
                summary_rows.append((task, sum(successes), len(successes), rate))

                if args.compare_with_demo:
                    mp4s = sorted(task_video_dir.glob("*_success*.mp4"))
                    for i, mp4 in enumerate(mp4s):
                        stitched = task_video_dir / f"compare_{task}__ep{i:02d}__{mp4.stem}.mp4"
                        stitch_with_demo(mp4, task, demo_episode=i, out_path=stitched)
            except Exception as e:
                print(f"  [{task}] ERROR: {type(e).__name__}: {e}", flush=True)
                import traceback; traceback.print_exc()
                summary_rows.append((task, -1, 0, 0.0))
    finally:
        print("\nShutting down server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()

    print("\n========== SUMMARY ==========")
    for task, succ, n, rate in summary_rows:
        if succ < 0:
            print(f"  {task}: ERROR")
        else:
            print(f"  {task}: {succ}/{n} = {rate:.2%}")

    if args.summary_csv:
        import csv
        new_file = not Path(args.summary_csv).exists()
        with open(args.summary_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["task", "successes", "n", "rate"])
            for row in summary_rows:
                w.writerow(row)
        print(f"\nAppended {len(summary_rows)} rows to {args.summary_csv}")


if __name__ == "__main__":
    main()
