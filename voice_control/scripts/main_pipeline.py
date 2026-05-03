#!/usr/bin/env python3
"""
End-to-end pipeline: Voice → SmolVLA → SO-101 Robot
Listens for voice commands, maps them to task instructions,
and runs SmolVLA inference to control the robot.
"""

import argparse
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from voice_detector import VoiceDetector


class VoiceControlPipeline:
    """Main pipeline integrating voice detection with robot control."""

    def __init__(self, model_path, device="cpu", robot_port="/dev/ttyACM0",
                 camera_index=0, max_steps_per_command=300,
                 start_pose_file=None, start_pose_steps=80, start_pose_delay_s=0.02):
        self.model_path = model_path
        self.device = device
        self.robot_port = robot_port
        self.camera_index = camera_index
        self.max_steps = max_steps_per_command
        self.start_pose_file = start_pose_file
        self.start_pose_steps = start_pose_steps
        self.start_pose_delay_s = start_pose_delay_s

        self.robot = None
        self.policy = None
        self.is_executing = False
        self.stop_requested = False

    def initialize(self):
        """Load model and connect robot."""
        print("[INIT] Loading SmolVLA policy...")
        from smolvla_infer import load_policy, make_robot
        self.policy = load_policy(self.model_path, self.device)
        print("[INIT] Policy loaded.")

        print("[INIT] Connecting robot...")
        self.robot = make_robot(self.robot_port, self.camera_index)
        print("[INIT] Robot connected.")

    def execute_task(self, task_instruction):
        """Execute a task using SmolVLA inference."""
        if self.is_executing:
            print("[WARN] Already executing a task, ignoring new command.")
            return

        self.is_executing = True
        self.stop_requested = False

        print(f"\n[EXEC] Starting task: {task_instruction}")

        step = 0
        try:
            # Force alignment at a recorded pose before every API/voice-triggered task.
            from smolvla_infer import maybe_move_to_start_pose
            maybe_move_to_start_pose(
                self.robot,
                pose_file=self.start_pose_file,
                steps=self.start_pose_steps,
                delay_s=self.start_pose_delay_s,
            )

            while step < self.max_steps and not self.stop_requested:
                observation = self.robot.get_observation()
                observation["task"] = task_instruction

                action = self.policy.predict(observation)
                self.robot.send_action(action)
                step += 1

                if step % 30 == 0:
                    print(f"[EXEC] Step {step}/{self.max_steps}")

        except Exception as e:
            print(f"[ERROR] Task execution failed: {e}")
        finally:
            self.is_executing = False

        status = "STOPPED" if self.stop_requested else "COMPLETED"
        print(f"[EXEC] Task {status} after {step} steps.")

    def emergency_stop(self):
        """Immediately stop current task execution."""
        print("\n[STOP] Emergency stop!")
        self.stop_requested = True

    def on_voice_command(self, command, task_instruction):
        """Callback for voice detector."""
        print(f"\n[VOICE] Heard: '{command}'")

        if task_instruction == "__STOP__":
            self.emergency_stop()
            return True

        if self.is_executing:
            print("[VOICE] Robot is busy, say 'stop' first to cancel current task.")
            return True

        # Run task execution in a separate thread so voice detection continues
        thread = threading.Thread(
            target=self.execute_task,
            args=(task_instruction,),
            daemon=True,
        )
        thread.start()
        return True

    def run(self):
        """Main loop: listen for voice commands and execute tasks."""
        print("\n" + "=" * 50)
        print("  VOICE-CONTROLLED SO-101 ROBOT")
        print("=" * 50)
        print("\nAvailable commands:")
        print("  'bring me water'  → Pick up cup and deliver")
        print("  'pick up the cup' → Pick up cup from desk")
        print("  'put it down'     → Place cup on desk")
        print("  'stop'            → Emergency stop")
        print("\nPress Ctrl+C to exit.")
        print("=" * 50 + "\n")

        voice = VoiceDetector()

        try:
            voice.listen_continuous(self.on_voice_command)
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            self.stop_requested = True
            time.sleep(0.5)

        if self.robot:
            self.robot.disconnect()
        print("Goodbye.")


def main():
    parser = argparse.ArgumentParser(description="Voice-controlled SO-101 robot")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path or HF repo ID of fine-tuned SmolVLA model")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--start-pose-file", type=str, default=None,
                        help="Path to recorded start pose JSON.")
    parser.add_argument("--start-pose-steps", type=int, default=80)
    parser.add_argument("--start-pose-delay-s", type=float, default=0.02)
    args = parser.parse_args()

    pipeline = VoiceControlPipeline(
        model_path=args.model_path,
        device=args.device,
        robot_port=args.port,
        camera_index=args.camera,
        max_steps_per_command=args.max_steps,
        start_pose_file=args.start_pose_file,
        start_pose_steps=args.start_pose_steps,
        start_pose_delay_s=args.start_pose_delay_s,
    )
    pipeline.initialize()
    pipeline.run()


if __name__ == "__main__":
    main()
