#!/usr/bin/env python3
"""
SmolVLA inference for SO-101 with smoother real-robot control.

Key goals:
1) keep workflow consistent with training (move to preset pose first, then grasp)
2) avoid per-run safety-threshold tweaking
3) reduce "stutter" by using policy action queue (select_action) + fixed control FPS
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.device_utils import auto_select_torch_device, is_torch_device_available

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_START_POSE_FILE = DATA_DIR / "api_preset_pose_1.json"
DEFAULT_RENAME_MAP = {
    "observation.images.agent": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}
DEFAULT_DATASET_REPO_ID = "ima/so101_grasp_cup"

JOINT_STATE_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _parse_camera_source(source: str | int | Path) -> int | Path:
    if isinstance(source, Path):
        return source
    if isinstance(source, int):
        return source
    s = str(source).strip()
    if s.isdigit():
        return int(s)
    return Path(s).expanduser().resolve()


def _resolve_device(device: str) -> str:
    device = str(device).strip()
    if device == "auto":
        return str(auto_select_torch_device())
    if is_torch_device_available(device):
        return device
    print(f"[WARN] Requested device '{device}' is unavailable. Falling back to CPU.")
    return "cpu"


def _infer_stats_path(dataset_repo_id: str | None) -> Path | None:
    if not dataset_repo_id:
        return None
    repo = dataset_repo_id.strip().replace("\\", "/")
    candidate = Path.home() / ".cache" / "huggingface" / "lerobot" / repo / "meta" / "stats.json"
    return candidate if candidate.exists() else None


def _parse_rename_map(value: str | dict[str, str] | None) -> dict[str, str]:
    if value is None:
        return dict(DEFAULT_RENAME_MAP)
    if isinstance(value, dict):
        return value
    raw = value.strip()
    if not raw:
        return dict(DEFAULT_RENAME_MAP)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid --rename-map JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError("--rename-map must be a JSON object.")
    return {str(k): str(v) for k, v in parsed.items()}


def _to_numpy_state(state: Any) -> np.ndarray:
    if hasattr(state, "detach"):
        return state.detach().cpu().numpy().astype(np.float32)
    if hasattr(state, "cpu") and hasattr(state, "numpy"):
        return state.cpu().numpy().astype(np.float32)
    return np.asarray(state, dtype=np.float32)


def _extract_joint_keys(obs: dict[str, Any]) -> list[str]:
    if all(k in obs for k in JOINT_STATE_KEYS):
        return list(JOINT_STATE_KEYS)
    pos_keys = sorted([k for k in obs if k.endswith(".pos")])
    if pos_keys:
        return pos_keys
    raise KeyError(f"joint state not found in observation keys: {list(obs.keys())}")


def get_current_joint_state(robot) -> tuple[np.ndarray, list[str]]:
    obs = robot.get_observation()
    joint_keys = _extract_joint_keys(obs)
    state = np.asarray([float(obs[k]) for k in joint_keys], dtype=np.float32)
    return state, joint_keys


def _joint_vector_to_action(joint_values: np.ndarray, joint_keys: list[str]) -> dict[str, float]:
    if len(joint_values) != len(joint_keys):
        raise ValueError(f"joint dim mismatch: {len(joint_values)} vs {len(joint_keys)}")
    return {joint_keys[i]: float(joint_values[i]) for i in range(len(joint_keys))}


def _parse_pose_file(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        pose = payload
    elif isinstance(payload, dict):
        pose = payload.get("start_pose")
        if pose is None:
            pose = payload.get("joint_positions")
        if pose is None:
            raise ValueError("pose json must contain 'start_pose' or 'joint_positions'")
    else:
        raise ValueError("pose json format is invalid")
    return np.asarray(pose, dtype=np.float32)


def make_robot(
    port: str = "/dev/ttyACM0",
    camera_index: int | str = 0,
    camera_index_wrist: int | str | None = None,
    agent_camera: int | str | None = None,
    wrist_camera: int | str | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    agent_rotation: int = 0,
    wrist_rotation: int = 180,
    robot_id: str = "my_follower",
    p_coefficient: int | None = 20,
):
    """Initialize SO-101 with agent camera + optional wrist camera."""
    agent_source = agent_camera if agent_camera is not None else camera_index
    wrist_source = wrist_camera if wrist_camera is not None else camera_index_wrist

    cameras = {
        "agent": OpenCVCameraConfig(
            index_or_path=_parse_camera_source(agent_source),
            width=width,
            height=height,
            fps=fps,
            rotation=agent_rotation,
        )
    }
    if wrist_source is not None:
        cameras["wrist"] = OpenCVCameraConfig(
            index_or_path=_parse_camera_source(wrist_source),
            width=width,
            height=height,
            fps=fps,
            rotation=wrist_rotation,
        )

    cfg = SO101FollowerConfig(
        port=port,
        id=robot_id,
        cameras=cameras,
        max_relative_target=None,  # keep unset by default, no per-run threshold tweaking
    )
    robot = SO101Follower(cfg)
    robot.connect()

    if p_coefficient is not None:
        p_value = int(p_coefficient)
        for motor in robot.bus.motors:
            robot.bus.write("P_Coefficient", motor, p_value)
        print(f"[ROBOT] P_Coefficient set to {p_value} for smoother, less sluggish motion.")

    return robot


def move_to_joint_pose(
    robot,
    target_pose: np.ndarray,
    steps: int = 80,
    delay_s: float = 0.02,
    speed_scale: float = 1.0,
    tolerance_deg: float = 1.0,
    correction_step_deg: float = 2.0,
    correction_iters: int = 40,
    keep_gripper_closed: bool = False,
):
    """
    Smoothly move to target, then run a short closed-loop correction phase.
    This improves final reachability without requiring global max_relative_target tuning.
    """
    current, joint_keys = get_current_joint_state(robot)
    target = np.asarray(target_pose, dtype=np.float32)
    if current.shape != target.shape:
        raise ValueError(f"joint dim mismatch: current={current.shape}, target={target.shape}")

    # speed_scale < 1.0 means slower movement (e.g. 0.5 -> half speed)
    speed_scale = max(float(speed_scale), 1e-3)
    effective_steps = max(1, int(round(float(steps) / speed_scale)))

    gripper_idx = None
    hold_gripper_value = None
    if keep_gripper_closed and "gripper.pos" in joint_keys:
        gripper_idx = joint_keys.index("gripper.pos")
        hold_gripper_value = float(current[gripper_idx])

    for i in range(1, effective_steps + 1):
        alpha = i / effective_steps
        alpha = 0.5 - 0.5 * np.cos(alpha * np.pi)  # cosine ease-in/out
        interp = current + (target - current) * alpha
        if gripper_idx is not None and hold_gripper_value is not None:
            interp[gripper_idx] = hold_gripper_value
        robot.send_action(_joint_vector_to_action(interp, joint_keys))
        time.sleep(delay_s)

    for _ in range(max(0, correction_iters)):
        cur, joint_keys = get_current_joint_state(robot)
        err = target - cur
        if float(np.max(np.abs(err))) <= tolerance_deg:
            break
        delta = np.clip(err, -correction_step_deg, correction_step_deg)
        cmd = cur + delta
        if gripper_idx is not None and hold_gripper_value is not None:
            cmd[gripper_idx] = hold_gripper_value
        robot.send_action(_joint_vector_to_action(cmd, joint_keys))
        time.sleep(delay_s)


def maybe_move_to_start_pose(
    robot,
    pose_file: str | Path = DEFAULT_START_POSE_FILE,
    steps: int = 80,
    delay_s: float = 0.02,
    speed_scale: float = 1.0,
    tolerance_deg: float = 1.0,
    keep_gripper_closed: bool = False,
):
    """Move to recorded pre-grasp pose if pose file exists."""
    path = Path(pose_file).expanduser().resolve()
    if not path.exists():
        print(f"[POSE] Pose file not found, skipping: {path}")
        return False

    target = _parse_pose_file(path)
    print(f"[POSE] Moving to pose from: {path}")
    move_to_joint_pose(
        robot,
        target_pose=target,
        steps=steps,
        delay_s=delay_s,
        speed_scale=speed_scale,
        tolerance_deg=tolerance_deg,
        keep_gripper_closed=keep_gripper_closed,
    )
    print("[POSE] Pose reached.")
    return True


def _build_runtime_features(obs: dict[str, Any]) -> dict[str, dict]:
    joint_keys = _extract_joint_keys(obs)
    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(joint_keys),),
            "names": joint_keys,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(joint_keys),),
            "names": joint_keys,
        },
    }

    for key, value in obs.items():
        if key.endswith(".pos"):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 3:
            features[f"observation.images.{key}"] = {
                "dtype": "image",
                "shape": tuple(int(x) for x in value.shape),
                "names": ["height", "width", "channels"],
            }
    return features


@dataclass
class PolicyRuntime:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: torch.device
    robot_type: str
    rename_map: dict[str, str]
    ds_features: dict[str, dict] | None = None

    def _ensure_features(self, raw_observation: dict[str, Any]):
        if self.ds_features is None:
            self.ds_features = _build_runtime_features(raw_observation)

    def predict(self, raw_observation: dict[str, Any]) -> dict[str, float]:
        self._ensure_features(raw_observation)
        task = raw_observation.get("task")
        obs_frame = build_inference_frame(
            observation=raw_observation,
            device=self.device,
            ds_features=self.ds_features,
            task=str(task) if task is not None else None,
            robot_type=self.robot_type,
        )
        obs_processed = self.preprocessor(obs_frame)
        action = self.policy.select_action(obs_processed)
        action = self.postprocessor(action)
        return make_robot_action(action, self.ds_features)


def load_policy(
    model_path: str,
    device: str = "cpu",
    rename_map: dict[str, str] | None = None,
    stats_path: str | None = None,
    robot_type: str = "so101_follower",
) -> PolicyRuntime:
    """Load policy + processors for online robot inference."""
    model_path = str(Path(model_path).expanduser().resolve())
    rename_map = dict(DEFAULT_RENAME_MAP if rename_map is None else rename_map)
    resolved_device = _resolve_device(device)

    policy_cfg = PreTrainedConfig.from_pretrained(model_path)
    policy_cfg.device = resolved_device

    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(model_path, config=policy_cfg)
    policy.to(policy_cfg.device)
    policy.eval()

    processor_stats = None
    if stats_path:
        stats_file = Path(stats_path).expanduser().resolve()
        if stats_file.exists():
            with open(stats_file) as f:
                processor_stats = rename_stats(json.load(f), rename_map)
            print(f"[POLICY] Loaded stats override: {stats_file}")
        else:
            print(f"[WARN] stats file not found, using policy default stats: {stats_file}")

    preprocessor_overrides = {
        "device_processor": {"device": resolved_device},
        "rename_observations_processor": {"rename_map": rename_map},
    }
    postprocessor_overrides = {"device_processor": {"device": "cpu"}}
    if processor_stats is not None:
        preprocessor_overrides["normalizer_processor"] = {"stats": processor_stats}
        postprocessor_overrides["unnormalizer_processor"] = {"stats": processor_stats}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=model_path,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
        dataset_stats=processor_stats,
    )

    return PolicyRuntime(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=torch.device(resolved_device),
        robot_type=robot_type,
        rename_map=rename_map,
    )


def run_inference(
    policy_runtime: PolicyRuntime,
    robot,
    task_instruction: str,
    max_steps: int = 300,
    control_fps: float = 12.0,
    action_deadband_deg: float = 0.0,
):
    """
    Inference loop with fixed-rate control.
    `select_action` internally uses action chunk queue, which reduces heavy model calls.
    """
    print(f"Running task: {task_instruction}")
    print(f"Max steps: {max_steps}, control_fps: {control_fps}")

    step = 0
    start_time = time.time()
    loop_period_s = 1.0 / max(control_fps, 1e-3)
    last_action: dict[str, float] | None = None

    try:
        while step < max_steps:
            loop_start = time.perf_counter()

            observation = robot.get_observation()
            observation["task"] = task_instruction
            action = policy_runtime.predict(observation)

            if action_deadband_deg > 0.0 and last_action is not None:
                action = {
                    k: (last_action[k] if abs(v - last_action[k]) < action_deadband_deg else v)
                    for k, v in action.items()
                }

            robot.send_action(action)
            last_action = action
            step += 1

            dt = time.perf_counter() - loop_start
            if step % 10 == 0:
                hz = 1.0 / max(dt, 1e-3)
                elapsed = time.time() - start_time
                print(f"  Step {step}/{max_steps} | loop {hz:.1f} Hz | {elapsed:.1f}s elapsed")

            sleep_s = loop_period_s - dt
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\nInference stopped by user.")

    print(f"Completed {step} steps in {time.time() - start_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="SmolVLA inference for SO-101 (smooth mode)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to fine-tuned model directory")
    parser.add_argument(
        "--task",
        type=str,
        default="Grasp the water cup",
        help="Task instruction string used at inference.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu / cuda / mps / auto")

    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Robot serial port")
    parser.add_argument("--robot-id", type=str, default="my_follower")
    parser.add_argument("--agent-camera", type=str, default="0", help="agent camera index or /dev/v4l/by-path/..")
    parser.add_argument("--wrist-camera", type=str, default="1", help="wrist camera index or /dev/v4l/by-path/..")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--agent-rotation", type=int, default=0)
    parser.add_argument("--wrist-rotation", type=int, default=180)
    parser.add_argument("--p-coefficient", type=int, default=20, help="Follower motor position P gain.")

    parser.add_argument("--max-steps", type=int, default=180, help="Maximum inference steps")
    parser.add_argument(
        "--control-fps",
        type=float,
        default=12.0,
        help="Control loop FPS. Lower is smoother on CPU-heavy setup.",
    )
    parser.add_argument(
        "--action-deadband-deg",
        type=float,
        default=0.2,
        help="Suppress tiny step-to-step action jitter.",
    )

    parser.add_argument("--start-pose-file", type=str, default=str(DEFAULT_START_POSE_FILE))
    parser.add_argument("--start-pose-steps", type=int, default=80)
    parser.add_argument("--start-pose-delay-s", type=float, default=0.02)
    parser.add_argument("--start-pose-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--skip-start-pose", action="store_true")

    parser.add_argument(
        "--rename-map",
        type=str,
        default=json.dumps(DEFAULT_RENAME_MAP, ensure_ascii=True),
        help="JSON mapping from robot obs keys to policy camera keys.",
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        default="",
        help="Optional stats.json path for normalization override.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=DEFAULT_DATASET_REPO_ID,
        help="Used to auto-discover ~/.cache/huggingface/lerobot/<repo>/meta/stats.json if --stats-path is empty.",
    )

    args = parser.parse_args()

    rename_map = _parse_rename_map(args.rename_map)
    stats_path = args.stats_path.strip() if args.stats_path else ""
    if not stats_path:
        inferred = _infer_stats_path(args.dataset_repo_id)
        if inferred is not None:
            stats_path = str(inferred)
            print(f"[POLICY] Auto stats path: {stats_path}")

    print("=== SmolVLA Inference (Smooth Mode) ===")
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Task: {args.task}")

    print("\nLoading policy...")
    policy_runtime = load_policy(
        model_path=args.model_path,
        device=args.device,
        rename_map=rename_map,
        stats_path=stats_path if stats_path else None,
        robot_type="so101_follower",
    )
    print("Policy loaded.")

    print("\nConnecting robot...")
    robot = make_robot(
        port=args.port,
        agent_camera=args.agent_camera,
        wrist_camera=args.wrist_camera,
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
        agent_rotation=args.agent_rotation,
        wrist_rotation=args.wrist_rotation,
        robot_id=args.robot_id,
        p_coefficient=args.p_coefficient,
    )
    print("Robot connected.")

    if not args.skip_start_pose:
        maybe_move_to_start_pose(
            robot,
            pose_file=args.start_pose_file,
            steps=args.start_pose_steps,
            delay_s=args.start_pose_delay_s,
            tolerance_deg=args.start_pose_tolerance_deg,
        )

    print("\nStarting inference...")
    run_inference(
        policy_runtime=policy_runtime,
        robot=robot,
        task_instruction=args.task,
        max_steps=args.max_steps,
        control_fps=args.control_fps,
        action_deadband_deg=args.action_deadband_deg,
    )

    robot.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
