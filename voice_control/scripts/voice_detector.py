#!/usr/bin/env python3
"""
Vosk-based voice command detector for SO-101 robot control.
Uses constrained grammar for high-accuracy command recognition.
"""

import json
import queue
import os
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# Path to Vosk model - adjust if needed
VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "vosk-model-small-en-us-0.15")

# Command mapping: spoken phrase → task instruction for SmolVLA
COMMAND_MAP = {
    "give me water": "Grasp the water cup",
    "give me the water": "Grasp the water cup",
    "get me water": "Grasp the water cup",
    "bring me water": "Pick up the water cup and bring it to the user",
    "pick up the cup": "Pick up the cup from the desk",
    "put it down": "Place the cup on the desk",
    "stop": "__STOP__",  # Emergency stop signal
}

# Grammar list for constrained recognition
GRAMMAR = json.dumps(list(COMMAND_MAP.keys()) + ["[unk]"])

SAMPLE_RATE = 16000
BLOCK_SIZE = 8000


def _pick_input_device(preferred_substring="USB2.0 Device"):
    """Pick a real microphone input instead of a Pulse monitor/default device."""
    devices = sd.query_devices()
    preferred = []
    secondary = []
    fallback = []
    for idx, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        name = str(device.get("name", ""))
        lowered = name.lower()
        if "monitor" in lowered:
            continue
        fallback.append(idx)
        if preferred_substring.lower() in name.lower():
            preferred.append(idx)
        elif "usb2.0" in lowered or "usb2" in lowered:
            secondary.append(idx)
        elif lowered in {"pulse", "default"}:
            secondary.append(idx)
    if preferred:
        return preferred[0]
    if secondary:
        return secondary[0]
    if fallback:
        return fallback[0]
    return None


class VoiceDetector:
    """Detects voice commands using Vosk with constrained grammar."""

    def __init__(self, model_path=VOSK_MODEL_PATH, command_map=None, input_device=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Vosk model not found at {model_path}\n"
                f"Download it:\n"
                f"  wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip\n"
                f"  unzip vosk-model-small-en-us-0.15.zip -d {os.path.dirname(model_path)}"
            )

        self.command_map = command_map or COMMAND_MAP
        grammar = json.dumps(list(self.command_map.keys()) + ["[unk]"])

        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE, grammar)
        self.audio_queue = queue.Queue()
        self._callback_registered = False
        self.input_device = input_device
        if self.input_device is None:
            env_device = os.environ.get("VOICE_INPUT_DEVICE")
            self.input_device = int(env_device) if env_device else _pick_input_device()
        if self.input_device is not None:
            info = sd.query_devices(self.input_device)
            print(f"[AUDIO] Using input device {self.input_device}: {info['name']}")

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(bytes(indata))

    def listen_once(self, timeout=None):
        """
        Block until a valid command is detected.
        Returns (command_text, task_instruction) or (None, None) on timeout.
        """
        with sd.RawInputStream(
            device=self.input_device,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        ):
            while True:
                try:
                    data = self.audio_queue.get(timeout=timeout)
                except queue.Empty:
                    return None, None

                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()
                    if text and text in self.command_map:
                        return text, self.command_map[text]

    def listen_continuous(self, callback):
        """
        Continuously listen for commands and call callback(command, task_instruction).
        callback should return False to stop listening.
        """
        with sd.RawInputStream(
            device=self.input_device,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        ):
            print("Listening for commands...")
            print(f"Available commands: {list(self.command_map.keys())}")
            while True:
                data = self.audio_queue.get()
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()
                    if text and text in self.command_map:
                        task = self.command_map[text]
                        if callback(text, task) is False:
                            break


def main():
    """Standalone test: print detected commands."""
    print("=== Voice Command Detector Test ===")
    print("Say one of: bring me water, pick up the cup, put it down, stop")
    print("Press Ctrl+C to exit\n")

    detector = VoiceDetector()

    def on_command(command, task):
        if task == "__STOP__":
            print(f"\n[STOP] Emergency stop triggered!")
            return False
        print(f"\n[COMMAND] '{command}' → Task: '{task}'")
        return True

    try:
        detector.listen_continuous(on_command)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
