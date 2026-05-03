# Robot Demo Protocol

## Before Running

1. Confirm the SO-101 arm is clear of obstacles.
2. Confirm the wrist camera is firmly attached.
3. Confirm the water cup or bottle is placed in the training-like location.
4. Confirm the trained model directory exists locally.
5. Confirm the Vosk model exists under `voice_control/models/`.
6. Start the workflow from preset pose 0.

## Run Command

```bash
cd /home/ima/Desktop/ITR_LeRobot
source .venv/bin/activate

python voice_control/scripts/voice_api_grasp_workflow.py \
  --model-path /home/ima/Desktop/ITR_LeRobot/pretrained_model_ready \
  --device cpu \
  --robot-port /dev/ttyACM0 \
  --agent-camera /dev/v4l/by-path/pci-0000:00:14.0-usb-0:6:1.0-video-index0 \
  --wrist-camera /dev/v4l/by-path/pci-0000:00:14.0-usb-0:5.1:1.0-video-index0 \
  --command-provider local \
  --require-water-word \
  --no-accept-speech-after-wake
```

## Live Monitoring

Open:

```text
http://127.0.0.1:8800
```

Important states:

- `wake`: waiting for "HEY RUDI"
- `command`: recording the spoken command
- `running`: robot workflow is active
- `done`: workflow completed
- `error`: workflow failed and needs inspection

## Spoken Test Command

```text
HEY RUDI, give me water
```

## Expected Robot Behavior

1. Detect wake phrase.
2. Accept command only after hearing "water".
3. Move to pose 1.
4. Run SmolVLA grasp.
5. Move slowly to pose 2 while keeping the gripper closed.
6. Hold for 15 seconds.
7. Return to pose 0 and open the gripper.

