#!/usr/bin/env python3
"""
Record API workflow preset poses for SO-101: preset 0 / 1 / 2.

Usage:
  1) Run script
  2) Move arm to requested preset pose
  3) Press Enter to record each pose
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


JOINT_STATE_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def extract_joint_state(obs: dict) -> list[float] | None:
    if "observation.state" in obs:
        state = obs["observation.state"]
        if hasattr(state, "detach"):
            return state.detach().cpu().numpy().astype(float).tolist()
        if hasattr(state, "cpu") and hasattr(state, "numpy"):
            return state.cpu().numpy().astype(float).tolist()
        return [float(v) for v in state]

    if all(k in obs for k in JOINT_STATE_KEYS):
        return [float(obs[k]) for k in JOINT_STATE_KEYS]

    pos_keys = sorted([k for k in obs.keys() if k.endswith(".pos")])
    if pos_keys:
        return [float(obs[k]) for k in pos_keys]
    return None


def save_pose(output_path: Path, state: list[float], port: str, robot_id: str, preset_idx: int):
    payload = {
        "start_pose": state,
        "preset_index": preset_idx,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "robot_port": port,
        "robot_id": robot_id,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Record SO-101 preset 0/1/2 for API grasp workflow.")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="my_follower")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Directory to save api_preset_pose_0.json / 1 / 2",
    )
    parser.add_argument(
        "--presets",
        default="0,1,2",
        help="Comma-separated preset indices to record, e.g. '1,2'",
    )
    parser.add_argument(
        "--free-move",
        action="store_true",
        default=True,
        help="Disable torque during recording so the arm can be moved by hand (default: enabled).",
    )
    parser.add_argument(
        "--no-free-move",
        dest="free_move",
        action="store_false",
        help="Keep torque enabled during recording.",
    )
    args = parser.parse_args()

    preset_indices = []
    for token in args.presets.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if idx not in (0, 1, 2):
            raise ValueError("preset index must be one of 0,1,2")
        preset_indices.append(idx)
    if not preset_indices:
        raise ValueError("no preset indices provided")

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = None
    connected = False
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        config = SO101FollowerConfig(port=args.port, id=args.robot_id)
        robot = SO101Follower(config)
        robot.connect()
        connected = True
        print(f"[OK] Connected: {args.port} ({args.robot_id})")
        if args.free_move:
            try:
                robot.bus.disable_torque()
                print("[OK] Torque disabled. You can move the arm by hand now.")
            except Exception as e:
                print(f"[WARN] Failed to disable torque automatically: {e}")
                print("[WARN] If the arm is stiff, stop this script and run emergency stop/torque-off first.")

        for idx in preset_indices:
            print("\n" + "=" * 52)
            print(f"Preset {idx}: move arm to target position now.")
            input("Press Enter to record this preset...")
            obs = robot.get_observation()
            state = extract_joint_state(obs)
            if state is None:
                print(f"[ERROR] no joint state found in observation keys: {list(obs.keys())}")
                return 1

            output_path = output_dir / f"api_preset_pose_{idx}.json"
            save_pose(output_path, state, args.port, args.robot_id, idx)
            print(f"[OK] Saved preset {idx}: {output_path}")
            print(f"[OK] Joint values: {[round(v, 3) for v in state]}")

        print("\nAll requested presets recorded.")
        return 0
    finally:
        if robot is not None and connected:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
