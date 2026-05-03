#!/usr/bin/env python3
"""
Fallback plan: ACT policy + voice command switching.
ACT doesn't accept language instructions, so voice commands
select between pre-trained single-task models.
"""

import argparse
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from voice_detector import VoiceDetector


# Map voice commands to ACT model paths
# Each task has its own separately trained ACT model
ACT_MODEL_MAP = {
    "bring me water": "outputs/train/act_cup_handover",
    "pick up the cup": "outputs/train/act_cup_pickup",
    "put it down": "outputs/train/act_cup_putdown",
}


class ACTFallbackPipeline:
    """Fallback pipeline using ACT models switched by voice commands."""

    def __init__(
        self,
        model_dir,
        robot_port="/dev/ttyACM0",
        camera_index=0,
        start_pose_file=None,
        start_pose_steps=80,
        start_pose_delay_s=0.02,
    ):
        self.model_dir = model_dir
        self.robot_port = robot_port
        self.camera_index = camera_index
        self.start_pose_file = start_pose_file
        self.start_pose_steps = start_pose_steps
        self.start_pose_delay_s = start_pose_delay_s
        self.robot = None
        self.policies = {}
        self.is_executing = False
        self.stop_requested = False

    def initialize(self):
        """Connect robot. Policies are loaded on demand."""
        from smolvla_infer import make_robot
        print("[INIT] Connecting robot...")
        self.robot = make_robot(self.robot_port, self.camera_index)
        print("[INIT] Robot connected.")

    def get_policy(self, model_path):
        """Load and cache ACT policy."""
        if model_path not in self.policies:
            print(f"[LOAD] Loading ACT model: {model_path}")
            from smolvla_infer import load_policy
            self.policies[model_path] = load_policy(model_path, "cpu")
            print(f"[LOAD] Model loaded.")
        return self.policies[model_path]

    def execute_with_act(self, model_path, max_steps=500):
        """Run ACT inference loop."""
        if self.is_executing:
            print("[WARN] Already executing.")
            return

        self.is_executing = True
        self.stop_requested = False

        policy = self.get_policy(model_path)
        step = 0

        try:
            from smolvla_infer import maybe_move_to_start_pose
            maybe_move_to_start_pose(
                self.robot,
                pose_file=self.start_pose_file,
                steps=self.start_pose_steps,
                delay_s=self.start_pose_delay_s,
            )

            while step < max_steps and not self.stop_requested:
                observation = self.robot.get_observation()
                action = policy.predict(observation)
                self.robot.send_action(action)
                step += 1
        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            self.is_executing = False

        print(f"[EXEC] Done after {step} steps.")

    def on_voice_command(self, command, task):
        if task == "__STOP__":
            print("\n[STOP] Emergency stop!")
            self.stop_requested = True
            return True

        model_path = ACT_MODEL_MAP.get(command)
        if not model_path:
            print(f"[WARN] No ACT model for command: {command}")
            return True

        if self.is_executing:
            print("[VOICE] Robot busy, say 'stop' first.")
            return True

        thread = threading.Thread(
            target=self.execute_with_act,
            args=(model_path,),
            daemon=True,
        )
        thread.start()
        return True

    def run(self):
        print("\n=== ACT FALLBACK MODE ===")
        print("Voice commands switch between pre-trained ACT models.")
        print("Press Ctrl+C to exit.\n")

        voice = VoiceDetector()
        try:
            voice.listen_continuous(self.on_voice_command)
        except KeyboardInterrupt:
            self.stop_requested = True
            time.sleep(0.5)

        if self.robot:
            self.robot.disconnect()


def main():
    parser = argparse.ArgumentParser(description="ACT fallback pipeline")
    parser.add_argument("--model-dir", type=str, default="outputs/train",
                        help="Directory containing ACT model checkpoints")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--start-pose-file", type=str, default=None,
                        help="Path to recorded start pose JSON.")
    parser.add_argument("--start-pose-steps", type=int, default=80)
    parser.add_argument("--start-pose-delay-s", type=float, default=0.02)
    args = parser.parse_args()

    pipeline = ACTFallbackPipeline(
        args.model_dir,
        args.port,
        args.camera,
        start_pose_file=args.start_pose_file,
        start_pose_steps=args.start_pose_steps,
        start_pose_delay_s=args.start_pose_delay_s,
    )
    pipeline.initialize()
    pipeline.run()


if __name__ == "__main__":
    main()
