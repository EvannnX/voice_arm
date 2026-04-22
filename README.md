# voice_arm

Voice control layer for an assistive robotic arm, powered by the Gemini Live API.

## Setup (Linux)

```bash
sudo apt install -y portaudio19-dev libportaudio2
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env     # then fill in GEMINI_API_KEY
```

## Run

```bash
voice-arm               # uses MockArm by default
voice-arm --arm mock    # explicit
```

Say commands in English: *"move right five centimeters"*, *"pick it up"*, *"go home"*, *"stop"*.

## Test

```bash
pytest
```

## Layout

- `src/voice_arm/audio/` — mic capture + speaker playback
- `src/voice_arm/llm/` — Gemini Live session, system prompt
- `src/voice_arm/tools/` — function declarations + dispatcher
- `src/voice_arm/arm/` — ArmController ABC, MockArm, SO-101 stub
- `src/voice_arm/app.py` — async entry point

The real SO-101 driver is a stub for this milestone.
