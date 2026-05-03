# Academic Poster Material

## Suggested Title

Rudi: A Voice-Triggered Vision-Language-Action Pipeline for SO-101 Water Grasping

## One-Sentence Summary

Rudi combines local wake-word detection, command filtering, preset robot poses, two-camera perception, and a fine-tuned SmolVLA policy to let an SO-101 arm respond to a spoken water request with an end-to-end grasp-and-deliver behavior.

## Abstract Draft

Assistive robot arms must connect natural user intent to reliable physical action. This project implements a real-time voice-triggered grasping pipeline on an SO-101 robotic arm. The system uses local audio processing to detect the wake phrase "HEY RUDI", verifies that the following command contains a water request, and then triggers a trained SmolVLA policy. Before inference, the robot moves to a fixed camera-aligned pose that matches the training demonstrations. The policy receives synchronized agent and wrist camera observations and executes the grasp. After grasping, the arm moves more slowly to a delivery pose while maintaining gripper closure, holds briefly, and returns to its safe start pose. The implementation emphasizes reproducibility, safety, and interpretability through a live dashboard, preset pose files, and separated robot, audio, and policy modules.

## Motivation

- Voice is a natural interface for assistive manipulation.
- Real robots need a bridge between speech intent and low-level policy execution.
- VLA policies are sensitive to camera viewpoint and initial state, so preset alignment matters.
- A live state dashboard helps debug failures during physical experiments.

## Key Contributions

- Local wake-word and command-gating pipeline that avoids accidental robot activation.
- Preset-pose workflow that aligns runtime state with the demonstration distribution.
- Two-view SmolVLA grasp execution using agent and wrist cameras.
- Slowed post-grasp transfer that keeps the gripper closed to reduce object drop.
- Reproducible robot-side code, preset JSON files, and LeRobot compatibility patch.

## System Workflow

1. Safe initialization: robot moves to preset pose 0.
2. Wake listening: microphone continuously listens for "HEY RUDI".
3. Command window: after wake, the system records the next command segment.
4. Intent filter: command must contain "water".
5. Pre-grasp alignment: robot moves to preset pose 1.
6. VLA execution: SmolVLA predicts actions from agent and wrist camera observations.
7. Delivery: robot moves to preset pose 2 at half speed while holding the object.
8. Reset: robot returns to pose 0 and opens the gripper.

## Suggested Poster Figures

- Figure 1: Full pipeline block diagram from microphone to robot motion.
- Figure 2: Photo of the physical setup with labeled agent camera, wrist camera, SO-101 arm, and water object.
- Figure 3: Timeline of one run: wake, command, pose 1, VLA grasp, pose 2 hold, pose 0 reset.
- Figure 4: Two camera views used by the policy.
- Figure 5: Example live dashboard screenshot showing system state.

## Evaluation Metrics to Report

- Wake detection reliability: accepted wakes / spoken wake attempts.
- Command acceptance reliability: water commands accepted / water commands spoken.
- False activation rate: non-water commands that triggered the robot.
- Grasp success rate: successful grasps / total trials.
- Delivery success rate: object still held after transfer to pose 2 / successful grasps.
- Full-task success rate: complete workflow success / total trials.
- Average workflow duration from command acceptance to return to pose 0.

## Demo Protocol for Results

Use a fixed test scene and run at least 10 trials:

- object: water cup or bottle
- initial robot state: pose 0
- grasp start state: pose 1
- spoken command: "HEY RUDI, give me water"
- success definition: robot grasps the object, reaches pose 2 while holding it, waits 15 seconds, returns to pose 0, and opens the gripper

Suggested results table:

| Metric | Value |
| --- | --- |
| Trials | TODO |
| Wake success | TODO |
| Command success | TODO |
| Grasp success | TODO |
| Delivery success | TODO |
| Full workflow success | TODO |
| Average duration | TODO |

## Limitations

- Current command understanding is intentionally narrow and requires a water-related command.
- CPU-only SmolVLA inference can be slow and may create jerky execution.
- The policy depends on camera placement and the preset pose matching the training setup.
- The system currently focuses on one object category rather than open-ended manipulation.

## Future Work

- Add GPU inference for smoother action execution.
- Expand the command vocabulary beyond water requests.
- Add automatic object detection before VLA execution.
- Record more diverse demonstrations across cup shapes, lighting, and table positions.
- Add quantitative logging for every trial to simplify evaluation.

## Three-Column Poster Layout

Column 1:

- Motivation
- Hardware setup
- Problem statement

Column 2:

- System architecture diagram
- Wake-command-VLA workflow
- Dataset and training setup

Column 3:

- Demo results table
- Safety behavior
- Limitations and future work

