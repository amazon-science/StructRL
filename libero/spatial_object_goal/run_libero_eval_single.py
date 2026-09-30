# Adapted from Isaac-GR00T (https://github.com/NVIDIA/Isaac-GR00T), examples/Libero/eval/run_libero_eval.py,
# Apache License 2.0. Modified for StructRL: one task per process, official per-suite horizons.
import os
import pprint
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import tqdm
import tyro
from libero.libero import benchmark

from utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    normalize_gripper_action,
    quat2axisangle,
    save_rollout_video,
)

log_dir = "/tmp/logs"
os.makedirs(log_dir, exist_ok=True)  # ensures directory exists


def summarize_obs(obs_dict):
    summary = {}
    for k, v in obs_dict.items():
        if isinstance(v, torch.Tensor):
            summary[k] = {"shape": tuple(v.shape), "dtype": v.dtype, "device": v.device}
        elif isinstance(v, np.ndarray):
            summary[k] = {"shape": v.shape, "dtype": v.dtype}
        else:
            summary[k] = type(v).__name__
    pprint.pprint(summary)


def show_obs_images_cv2(new_obs):
    # remove batch dim
    img_agent = new_obs["video.image"][0]
    img_wrist = new_obs["video.wrist_image"][0]

    # convert RGB -> BGR for OpenCV
    img_agent_bgr = cv2.cvtColor(img_agent, cv2.COLOR_RGB2BGR)
    img_wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)

    # show in separate windows
    cv2.imshow("Agent View", img_agent_bgr)
    cv2.imshow("Wrist View", img_wrist_bgr)
    cv2.waitKey(1)


@dataclass
class GenerateConfig:
    # fmt: off
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 5                    # Number of rollouts per task
    task_id: int = -1                                # If >=0, run ONLY this single task_id (for per-GPU parallel eval); -1 = whole suite
    #################################################################################################################
    # fmt: on
    """Port to connect to."""
    port: int = 5555
    """Headless mode (no GUI)."""
    headless: bool = False


class GR00TPolicy:
    """GR00T Policy wrapper for Libero environments."""

    LIBERO_CONFIG = {
        "proprio_size": 8,
        "state_key_mapping": {
            "x": 0,
            "y": 1,
            "z": 2,
            "roll": 3,
            "pitch": 4,
            "yaw": 5,
            "gripper": (6, 8),
        },
    }

    def __init__(self, host="localhost", port=5555, headless=False):
        from gr00t.eval.service import ExternalRobotInferenceClient

        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.config = self.LIBERO_CONFIG
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        self.headless = headless

    def get_action(self, observation_dict, lang: str):
        """Get action from GR00T policy given observation and language instruction."""
        obs_dict = self._process_observation(observation_dict, lang)
        # summarize_obs(obs_dict)
        action_chunk = self.policy.get_action(obs_dict)
        return self._convert_to_libero_action(action_chunk, 0)

    def get_action_chunk(self, observation_dict, lang: str):
        """Get the FULL action chunk (all horizon steps) for OPEN-LOOP execution,
        matching the RLinf training/eval protocol (chunk_step executes all 16
        predicted steps before re-inferring). Returns list of 7-dim arrays."""
        obs_dict = self._process_observation(observation_dict, lang)
        action_chunk = self.policy.get_action(obs_dict)
        horizon = len(np.atleast_1d(action_chunk[f"action.{self.action_keys[0]}"]))
        return [self._convert_to_libero_action(action_chunk, i) for i in range(horizon)]

    def _process_observation(self, obs, lang: str):
        """Convert Libero observation to GR00T format."""
        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        img, wrist_img = get_libero_image(obs)
        new_obs = {
            "video.image": np.expand_dims(img, axis=0),
            "video.wrist_image": np.expand_dims(wrist_img, axis=0),
            "state.x": np.array([[xyz[0]]]),
            "state.y": np.array([[xyz[1]]]),
            "state.z": np.array([[xyz[2]]]),
            "state.roll": np.array([[rpy[0]]]),
            "state.pitch": np.array([[rpy[1]]]),
            "state.yaw": np.array([[rpy[2]]]),
            "state.gripper": np.expand_dims(gripper, axis=0),
            "annotation.human.action.task_description": [lang],
        }
        if not self.headless:
            show_obs_images_cv2(new_obs)
        return new_obs

    def _convert_to_libero_action(
        self, action_chunk: dict[str, np.array], idx: int = 0
    ) -> np.ndarray:
        """Convert GR00T action chunk to Libero format.

        Args:
            action_chunk: Dictionary of action components from GR00T policy
            idx: Index of action to extract from chunk (default: 0 for first action)

        Returns:
            7-dim numpy array: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        action_components = [
            np.atleast_1d(action_chunk[f"action.{key}"][idx])[0] for key in self.action_keys
        ]
        action_array = np.array(action_components, dtype=np.float32)
        action_array = normalize_gripper_action(action_array, binarize=True)
        assert len(action_array) == 7, f"Expected 7-dim action, got {len(action_array)}"
        return action_array


def eval_libero(cfg: GenerateConfig) -> None:
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    if cfg.task_id >= 0:
        task_ids = [cfg.task_id]
        log_file = open(f"{log_dir}/libero_eval_{cfg.task_suite_name}_task{cfg.task_id}.log", "w")
    else:
        task_ids = list(range(num_tasks_in_suite))
        log_file = open(f"{log_dir}/libero_eval_{cfg.task_suite_name}.log", "w")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, resolution=256)

        gr00t_policy = GR00TPolicy(host="localhost", port=cfg.port, headless=cfg.headless)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            top_view = []
            wrist_view = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 600  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 1000  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps
            # optional horizon override (e.g. MAX_STEPS_OVERRIDE=520 for
            # the openpi-community protocol), mirrors harness_openpi's --max_steps.
            if os.environ.get("MAX_STEPS_OVERRIDE"):
                max_steps = int(os.environ["MAX_STEPS_OVERRIDE"])

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            done = False
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    # Query model to get the FULL action chunk, then execute it
                    # OPEN-LOOP (all horizon steps) to match the RLinf training/eval
                    # protocol (chunk_step runs all 16 predicted steps before re-inferring).
                    action_chunk = gr00t_policy.get_action_chunk(obs, task.language)
                    # EXEC_HORIZON: execute only the first N steps of the 16-chunk
                    # (1 = pure closed-loop, Omega-QVLA groot protocol; 16 = open-loop).
                    import os as _os
                    _eh = int(_os.environ.get("EXEC_HORIZON", "16"))

                    for action in action_chunk[:_eh]:
                        # # Save preprocessed image for replay video
                        img, wrist_img = get_libero_image(obs)
                        top_view.append(img)
                        wrist_view.append(wrist_img)

                        obs, reward, done, info = env.step(action.tolist())
                        t += 1
                        if done or t >= max_steps + cfg.num_steps_wait:
                            break

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                top_view,
                wrist_view,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=log_file,
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n"
            )
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}\n"
        )
        log_file.write(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}\n"
        )
        log_file.flush()

    # Save local log file
    log_file.close()


if __name__ == "__main__":
    cfg = tyro.cli(GenerateConfig)
    eval_libero(cfg)
