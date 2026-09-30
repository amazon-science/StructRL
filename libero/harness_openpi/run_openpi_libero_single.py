# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from RLinf toolkits/eval_scripts_openpi/libero_eval.py for StructRL.
#
# Single-task LIBERO eval for RLinf openpi (pi0 / pi05) PyTorch checkpoints.
#
# This is a thin fork of RLinf's toolkits/eval_scripts_openpi/libero_eval.py.
# ONLY change: run exactly ONE task_id (one-task-per-GPU orchestration, like the
# GR00T harness eval_sft_d4_openloop.sbatch) and write a per-task stats.json so
# the 10 tasks (2 nodes x [8+2] GPUs) can be aggregated afterwards.
#
# Everything model/env-facing (image rotation, state vector, replan/action_chunk
# semantics, max_steps, num_steps_wait) is byte-identical to the official script.
#
# Expects the caller to have set PYTHONPATH = RLinf : LIBERO_src : rlinf-libero_site-packages
# and to run with openpi-robocasa/.venv python.
import argparse
import collections
import dataclasses
import json
import math
import os
import pathlib

import imageio
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi.training.config import DataConfig

from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT
from toolkits.eval_scripts_openpi import create_trained_policy, setup_logger

os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _quat2axisangle(quat):
    """Copied verbatim from robosuite / official libero_eval.py."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        # robosuite-internal horizon MUST exceed the outer max_steps loop.
        # Default 1000 < 1024+10 made done fire as TIMEOUT at step 1000, and the
        # harness counted it as success (fake-100% bug). 2000 keeps
        # done == _check_success() for any outer horizon we use (520 or 1024).
        "horizon": 2000,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def main(args):
    logger = setup_logger(args.exp_name, args.log_dir)
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logger.info(f"Task suite: {args.task_suite_name} | ONLY task_id={args.task_id}")
    assert 0 <= args.task_id < num_tasks_in_suite, (
        f"task_id {args.task_id} out of range [0,{num_tasks_in_suite})"
    )

    # max_steps per suite — identical to official script.
    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    if args.max_steps is not None:
        max_steps = args.max_steps  # horizon override (1024 = RL-training parity)

    logger.info("policy setup start")
    # setup_policy() inlined so we can fix use_quantile_norm: upstream openpi sets
    # use_quantile_norm = (model_type != PI0), but the robocasa fork's rewritten
    # create_base_config dropped it (default False). pi05 ckpts are trained with
    # quantile norm — z-score unnorm compresses actions 2-3x and pins the gripper
    # at -1.12, giving 0% SR. pi0 keeps z-score as upstream intends.
    config = _CONFIGS_DICT[args.config_name]
    if config.model.pi05:
        base = config.data.base_config or DataConfig()
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(
                config.data,
                base_config=dataclasses.replace(base, use_quantile_norm=True),
            ),
        )
    data_config = config.data.create(config.assets_dirs, config.model)
    logger.info(f"use_quantile_norm={data_config.use_quantile_norm}")
    policy = create_trained_policy(
        config, args.pretrained_path, sample_kwargs={"num_steps": args.num_steps}
    )
    logger.info("policy setup done")

    task_id = args.task_id
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
        logger.info(f"\nTask: {task_description} | episode {task_episodes + 1}")
        policy.reset()
        env.reset()
        action_plan = collections.deque()
        obs = env.set_init_state(initial_states[episode_idx])
        replay_images = []
        done = False

        for t in range(max_steps + args.num_steps_wait):
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                continue

            # IMPORTANT: rotate 180 degrees to match train preprocessing
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(
                obs["robot0_eye_in_hand_image"][::-1, ::-1]
            )
            replay_images.append(img)

            state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            )

            if not action_plan:
                observation = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": str(task_description),
                }
                action_chunk = policy.infer(observation)["actions"]
                assert len(action_chunk) >= args.action_chunk, (
                    f"We want to replan every {args.action_chunk} steps, but policy "
                    f"only predicts {len(action_chunk)} steps."
                )
                action_plan.extend(action_chunk[: args.action_chunk])

            action = action_plan.popleft()
            obs, reward, done, info = env.step(action.tolist())
            if done:
                task_successes += 1
                break

        task_episodes += 1

        # Save the first few rollout videos for eyeballing.
        if episode_idx < args.num_save_videos:
            suffix = "success" if done else "failure"
            out_path = (
                pathlib.Path(f"{args.log_dir}/{args.exp_name}/")
                / f"rollout_t{task_id}_{episode_idx}_{suffix}.mp4"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(
                out_path,
                [np.asarray(x) for x in replay_images[:: args.video_temp_subsample]],
                fps=30 // args.video_temp_subsample,
            )

        logger.info(
            f"[t{task_id}] {task_successes}/{task_episodes} "
            f"({task_successes / task_episodes * 100:.1f}%)"
        )

    env.close()
    sr = task_successes / task_episodes if task_episodes else 0.0
    logger.info(
        f"\n[t{task_id}] {task_description}: {task_successes}/{task_episodes} = {sr:.2%}"
    )

    # per-task stats.json for aggregation
    out_dir = pathlib.Path(args.log_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"stats_task{task_id}.json", "w") as f:
        json.dump(
            {
                "task_id": task_id,
                "task_description": task_description,
                "num_episodes": task_episodes,
                "num_successes": task_successes,
                "success_rate": sr,
                "action_chunk": args.action_chunk,
                "num_steps": args.num_steps,
                "config_name": args.config_name,
                "pretrained_path": args.pretrained_path,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--exp_name", type=str, default="libero10_pi0")
    parser.add_argument("--config_name", type=str, default="pi0_libero")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--task_suite_name", type=str, default="libero_10")
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--action_chunk", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None, help="override suite default horizon (e.g. 1024 to match RL training)")
    parser.add_argument("--num_save_videos", type=int, default=5)
    parser.add_argument("--video_temp_subsample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    main(args)
