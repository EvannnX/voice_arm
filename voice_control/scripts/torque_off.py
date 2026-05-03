#!/usr/bin/env python3
"""Disable SO-101 motor torque so the arm can be moved by hand."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Disable torque on SO-101 follower.")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="my_follower")
    args = parser.parse_args()

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = None
    connected = False
    try:
        cfg = SO101FollowerConfig(port=args.port, id=args.robot_id)
        robot = SO101Follower(cfg)
        robot.connect()
        connected = True
        robot.bus.disable_torque()
        print(f"[OK] Torque disabled on {args.port}.")
        return 0
    finally:
        if robot is not None and connected:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
