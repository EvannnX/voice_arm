# Reproducibility Notes

## Files Included

- Robot workflow scripts
- Voice wake and command scripts
- Preset pose JSON files
- Camera reference images
- Training config references
- Local LeRobot patch

## Files Not Included

The following files are intentionally excluded because they are large, machine-specific, or private:

- trained model weights
- recorded datasets
- evaluation videos
- local virtual environments
- API keys
- Vosk model binaries

## Camera Feature Names

The robot records two cameras:

- `agent`
- `wrist`

The SmolVLA policy expects:

- `camera1`
- `camera2`

The runtime code applies the same rename mapping used during training.

## Preset Poses

Preset poses are saved as JSON files in `voice_control/data/`:

- `api_preset_pose_0.json`: safe start and reset pose
- `api_preset_pose_1.json`: pre-grasp pose matching the training distribution
- `api_preset_pose_2.json`: delivery/hold pose
- `api_preset_pose_3.json`: optional alternate pose from earlier tests

