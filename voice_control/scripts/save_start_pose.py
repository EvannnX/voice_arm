#!/usr/bin/env python3
"""
Record current SO-101 joint pose as the API pre-grasp start pose.

This saved pose can be loaded by `smolvla_infer.py`, `main_pipeline.py`,
and `fallback_act.py` so each task starts from the same known state.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "api_start_pose.json"
JOINT_STATE_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _to_list(state) -> list[float]:
    if hasattr(state, "detach"):
        return state.detach().cpu().numpy().astype(float).tolist()
    if hasattr(state, "cpu") and hasattr(state, "numpy"):
        return state.cpu().numpy().astype(float).tolist()
    return [float(v) for v in state]


def _extract_joint_state(obs: dict) -> list[float] | None:
    # Newer observation format.
    if "observation.state" in obs:
        return _to_list(obs["observation.state"])

    # Joint-key format.
    if all(k in obs for k in JOINT_STATE_KEYS):
        return [float(obs[k]) for k in JOINT_STATE_KEYS]

    # Fallback for other ".pos" key sets.
    pos_keys = sorted([k for k in obs.keys() if k.endswith(".pos")])
    if pos_keys:
        return [float(obs[k]) for k in pos_keys]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Save current SO-101 joint pose for API start pose.")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--robot-id", type=str, default="my_follower")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()

    try:
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    except ModuleNotFoundError:
        # Newer LeRobot path.
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = None
    connected = False
    try:
        config = SO101FollowerConfig(port=args.port, id=args.robot_id)
        robot = SO101Follower(config)
        robot.connect()
        connected = True
        obs = robot.get_observation()
        state = _extract_joint_state(obs)
        if state is None:
            print(f"[ERROR] no joint state found in observation keys: {list(obs.keys())}")
            return 1
        payload = {
            "start_pose": state,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "robot_port": args.port,
            "robot_id": args.robot_id,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))

        print(f"[OK] Saved start pose to: {output_path}")
        print(f"[OK] Joint values: {[round(v, 3) for v in state]}")
        return 0
    finally:
        if robot is not None and connected:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
