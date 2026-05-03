#!/bin/bash
# =============================================================
# SO-101 Cup Handover Dataset Recording Script
# =============================================================
# Before running:
# 1. Set your HuggingFace username below
# 2. Verify robot ports (run: lerobot-find-port)
# 3. Verify camera index (run: ls /dev/video*)
# 4. Login to HF: huggingface-cli login
# =============================================================

# ---- Configuration (EDIT THESE) ----
HF_USER="YOUR_HF_USERNAME"          # <-- Change this!
FOLLOWER_PORT="/dev/ttyACM0"         # <-- Verify with lerobot-find-port
LEADER_PORT="/dev/ttyACM1"           # <-- Verify with lerobot-find-port
CAMERA_INDEX=0                        # <-- Verify with ls /dev/video*
NUM_EPISODES=50
TASK="Pick up the water cup and bring it to the user"

# ---- Do not edit below ----
DATASET_REPO="${HF_USER}/so101_cup_handover"

echo "============================================"
echo "  SO-101 Cup Handover Dataset Recording"
echo "============================================"
echo "HF User:    ${HF_USER}"
echo "Dataset:    ${DATASET_REPO}"
echo "Episodes:   ${NUM_EPISODES}"
echo "Task:       ${TASK}"
echo "Follower:   ${FOLLOWER_PORT}"
echo "Leader:     ${LEADER_PORT}"
echo "Camera:     ${CAMERA_INDEX}"
echo "============================================"
echo ""

if [ "$HF_USER" = "YOUR_HF_USERNAME" ]; then
    echo "ERROR: Please edit this script and set your HuggingFace username."
    exit 1
fi

echo "Starting recording in 3 seconds..."
echo "  - Place cup at one of 5 positions"
echo "  - Record 10 episodes per position"
echo "  - Each episode: 5-15 seconds of smooth motion"
echo "  - Discard failed grasps immediately"
echo ""
sleep 3

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=${FOLLOWER_PORT} \
  --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: ${CAMERA_INDEX}, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=${LEADER_PORT} \
  --teleop.id=my_leader \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.single_task="${TASK}" \
  --dataset.num_episodes=${NUM_EPISODES}
