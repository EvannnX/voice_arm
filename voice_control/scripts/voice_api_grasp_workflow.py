#!/usr/bin/env python3
"""
Voice wake-word + API-triggered SO-101 SmolVLA grasp workflow.

Flow:
1) robot initializes at pose 0
2) microphone listens for "hey rudi"
3) after wake, record the user's command audio from the same microphone
4) trigger the existing API workflow:
   pose 1 -> SmolVLA grasp -> pose 2 hold -> pose 0 release
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import re
import threading
import time
import urllib.request
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model

from api_grasp_workflow import (
    DEFAULT_BOOT_POSE_FILE,
    DEFAULT_PRESET1_POSE_FILE,
    DEFAULT_PRESET2_POSE_FILE,
    TriggerRequest,
    WorkflowApiHandler,
    WorkflowRunner,
)
from smolvla_infer import DEFAULT_RENAME_MAP
from voice_detector import SAMPLE_RATE, VOSK_MODEL_PATH, _pick_input_device


DEFAULT_WAKE_PHRASES = [
    "hey",
    "hey rudi",
    "hey rudy",
    "hey ruddy",
    "hi rudi",
    "hi rudy",
    "rudi",
    "rudy",
]

DEFAULT_WATER_COMMANDS = {
    "give me water": "Grasp the water cup",
    "give me the water": "Grasp the water cup",
    "give me a water": "Grasp the water cup",
    "bring me water": "Grasp the water cup",
    "bring me the water": "Grasp the water cup",
    "get me water": "Grasp the water cup",
    "get the water": "Grasp the water cup",
    "i want water": "Grasp the water cup",
    "water": "Grasp the water cup",
    "give me the cup": "Grasp the water cup",
    "bring me the cup": "Grasp the water cup",
}

DEFAULT_COMBINED_WAKE_WATER_PHRASES = {
    "hey rudi give me water",
    "hey rudy give me water",
    "hey rudi give me the water",
    "hey rudy give me the water",
    "hey rudi bring me water",
    "hey rudy bring me water",
    "hey rudi get me water",
    "hey rudy get me water",
    "hey rudi water",
    "hey rudy water",
    "rudi give me water",
    "rudy give me water",
    "rudi water",
    "rudy water",
}


class VoiceWakeController:
    def __init__(
        self,
        runner: WorkflowRunner,
        model_path: str,
        input_device: int | None,
        api_command: str,
        default_task: str,
        max_steps: int,
        wake_phrases: list[str],
        command_map: dict[str, str],
        command_timeout_s: float,
        sample_rate: float,
        command_api_url: str | None,
        command_api_timeout_s: float,
        command_provider: str,
        gemini_model: str,
        accept_speech_after_wake: bool,
        require_water_word: bool,
        wake_cooldown_s: float,
        command_preroll_s: float,
        command_audio_rms_threshold: float,
        command_silence_s: float,
    ):
        self.runner = runner
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.input_device = input_device
        self.api_command = api_command
        self.default_task = default_task
        self.max_steps = max_steps
        self.wake_phrases = {phrase.strip().lower() for phrase in wake_phrases if phrase.strip()}
        self.command_map = {
            phrase.strip().lower(): task.strip() if task.strip() else default_task
            for phrase, task in command_map.items()
            if phrase.strip()
        }
        self.command_timeout_s = command_timeout_s
        self.sample_rate = sample_rate
        self.command_api_url = command_api_url.strip() if command_api_url else None
        self.command_api_timeout_s = command_api_timeout_s
        self.command_provider = command_provider
        self.gemini_model = gemini_model
        self.accept_speech_after_wake = accept_speech_after_wake
        self.require_water_word = require_water_word
        self.wake_cooldown_s = wake_cooldown_s
        self.command_preroll_s = max(0.0, command_preroll_s)
        self._ignore_wake_until = 0.0
        self.command_audio_rms_threshold = command_audio_rms_threshold
        self.command_silence_s = command_silence_s
        self._vosk_model: Model | None = None
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self._status_lock = threading.Lock()
        self._status: dict[str, Any] = {
            "mode": "initializing",
            "last_text": "",
            "partial_text": "",
            "last_command_text": "",
            "last_local_command_text": "",
            "last_local_command_candidates": [],
            "last_command_raw_response": "",
            "last_command_wav_path": "",
            "last_command_boosted_wav_path": "",
            "last_command_rms": 0.0,
            "last_event": "starting",
            "command_deadline": 0.0,
            "command_provider": self.command_provider,
            "gemini_model": self.gemini_model,
            "accept_speech_after_wake": self.accept_speech_after_wake,
            "require_water_word": self.require_water_word,
            "wake_cooldown_s": self.wake_cooldown_s,
            "command_preroll_s": self.command_preroll_s,
            "command_api_configured": self._command_api_configured(),
        }

    def _set_status(self, **updates):
        with self._status_lock:
            self._status.update(updates)
            self._status["updated_at"] = time.time()

    def get_status(self) -> dict[str, Any]:
        with self._status_lock:
            return dict(self._status)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[VOICE] Audio status: {status}")
        self.audio_queue.put(bytes(indata))

    def _trigger_from_command(self, text: str) -> dict[str, Any]:
        task = self._task_for_command(text)
        return self.runner.try_trigger(
            TriggerRequest(
                api_command=self.api_command,
                user_command=text,
                task=task,
                max_steps=self.max_steps,
            )
        )

    def _is_wake_text(self, text: str) -> bool:
        if text in self.wake_phrases:
            return True
        words = set(text.split())
        return bool({"hey", "hi"} & words and {"rudi", "rudy", "ruddy"} & words)

    def _task_for_command(self, text: str) -> str | None:
        if self.require_water_word and not self._has_water_word(text):
            return None
        if text in self.command_map:
            return self.command_map[text]
        words = set(text.split())
        if words & {"water", "cup", "bottle"}:
            return self.default_task
        if words & {"give", "bring", "get", "want"}:
            return self.default_task
        return None

    @staticmethod
    def _has_water_word(text: str) -> bool:
        return "water" in re.findall(r"[a-z']+", str(text).lower())

    def _clear_audio_queue(self):
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                return

    def _drain_audio_queue(self, max_bytes: int | None = None) -> list[bytes]:
        drained: list[bytes] = []
        total = 0
        while True:
            try:
                chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            drained.append(chunk)
            total += len(chunk)
            if max_bytes is not None and total >= max_bytes:
                break
        if max_bytes is not None and total > max_bytes:
            kept: list[bytes] = []
            kept_total = 0
            for chunk in reversed(drained):
                kept.insert(0, chunk)
                kept_total += len(chunk)
                if kept_total >= max_bytes:
                    break
            drained = kept
        return drained

    @staticmethod
    def _pcm_rms(pcm: bytes) -> float:
        if not pcm:
            return 0.0
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return 0.0
        values = samples.astype(np.float32)
        return float(np.sqrt(np.mean(values * values)))

    def _pcm_to_wav_bytes(self, pcm: bytes) -> bytes:
        return self._pcm_to_wav_bytes_at_rate(pcm, int(round(self.sample_rate)))

    @staticmethod
    def _pcm_to_wav_bytes_at_rate(pcm: bytes, sample_rate: int) -> bytes:
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(pcm)
            return buffer.getvalue()

    def _pcm_to_wav_b64(self, pcm: bytes) -> str:
        return base64.b64encode(self._pcm_to_wav_bytes(pcm)).decode("ascii")

    @staticmethod
    def _gemini_api_key() -> str:
        return os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()

    def _command_api_configured(self) -> bool:
        if self.command_provider in {"auto", "gemini"} and self._gemini_api_key():
            return True
        if self.command_provider in {"auto", "http"} and self.command_api_url:
            return True
        return False

    @staticmethod
    def _load_jsonish(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`").strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start < 0 or end <= start:
                raise
            parsed = json.loads(stripped[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("command classifier returned non-object JSON")
        return parsed

    @staticmethod
    def _normalize_pcm_for_asr(pcm: bytes, target_rms: float = 1200.0, max_gain: float = 120.0) -> bytes:
        if not pcm:
            return b""
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return b""
        values = samples.astype(np.float32)
        rms = float(np.sqrt(np.mean(values * values)))
        if rms <= 0:
            return pcm
        gain = min(max_gain, max(1.0, target_rms / rms))
        boosted = np.clip(values * gain, -32768, 32767).astype(np.int16)
        return boosted.tobytes()

    @staticmethod
    def _resample_pcm(pcm: bytes, source_rate: float, target_rate: int = 16000) -> bytes:
        if not pcm:
            return b""
        source_rate = float(source_rate)
        if source_rate <= 0 or int(round(source_rate)) == target_rate:
            return pcm
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return b""
        duration = samples.size / source_rate
        target_size = max(1, int(round(duration * target_rate)))
        source_x = np.linspace(0.0, duration, num=samples.size, endpoint=False)
        target_x = np.linspace(0.0, duration, num=target_size, endpoint=False)
        resampled = np.interp(target_x, source_x, samples.astype(np.float32))
        return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()

    @staticmethod
    def _write_wav(path: str, pcm: bytes, sample_rate: int):
        Path(path).write_bytes(VoiceWakeController._pcm_to_wav_bytes_at_rate(pcm, sample_rate))

    def _capture_command_audio(self) -> tuple[bytes, float]:
        deadline = time.time() + self.command_timeout_s
        bytes_per_second = int(round(self.sample_rate)) * 2
        preroll_bytes = int(bytes_per_second * self.command_preroll_s)
        chunks: list[bytes] = self._drain_audio_queue(preroll_bytes)
        heard_speech = False
        last_voice_at = 0.0
        peak_rms = max((self._pcm_rms(chunk) for chunk in chunks), default=0.0)

        self._set_status(
            mode="command_recording",
            last_event="recording command from microphone",
            command_deadline=deadline,
        )
        print(f"[VOICE] Recording command from microphone for up to {self.command_timeout_s:.1f}s.")

        while time.time() < deadline:
            try:
                chunk = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            chunks.append(chunk)
            rms = self._pcm_rms(chunk)
            peak_rms = max(peak_rms, rms)
            self._set_status(last_command_rms=round(peak_rms, 1))
            if rms >= self.command_audio_rms_threshold:
                heard_speech = True
                last_voice_at = time.time()
            elif heard_speech and time.time() - last_voice_at >= self.command_silence_s:
                break

        pcm = b"".join(chunks)
        self._set_status(last_command_rms=round(max(peak_rms, self._pcm_rms(pcm)), 1))
        boosted_pcm = self._normalize_pcm_for_asr(pcm)
        raw_path = "/tmp/rudi_last_command.wav"
        boosted_path = "/tmp/rudi_last_command_boosted.wav"
        try:
            self._write_wav(raw_path, pcm, int(round(self.sample_rate)))
            self._write_wav(boosted_path, boosted_pcm, int(round(self.sample_rate)))
            self._set_status(last_command_wav_path=raw_path, last_command_boosted_wav_path=boosted_path)
        except Exception as e:
            print(f"[VOICE] Failed to save command debug wav: {e}")
        return pcm, peak_rms

    def _transcribe_with_vosk(self, pcm: bytes, sample_rate: float, grammar: list[str]) -> str:
        if not pcm or self._vosk_model is None:
            return ""
        recognizer = KaldiRecognizer(self._vosk_model, sample_rate, json.dumps(grammar))
        chunk_size = max(1024, int(round(sample_rate * 0.25)) * 2)
        for i in range(0, len(pcm), chunk_size):
            recognizer.AcceptWaveform(pcm[i : i + chunk_size])
        result = json.loads(recognizer.FinalResult())
        return str(result.get("text", "")).strip().lower()

    def _transcribe_command_locally(self, pcm: bytes) -> tuple[str, list[str]]:
        if not pcm or self._vosk_model is None:
            return "", []
        grammar = sorted(
            {
                "give me water",
                "give me the water",
                "bring me water",
                "bring me the water",
                "get me water",
                "get the water",
                "i want water",
                "water",
            }
        ) + ["[unk]"]
        boosted_pcm = self._normalize_pcm_for_asr(pcm)
        variants: list[tuple[str, bytes, float]] = [
            ("native", pcm, self.sample_rate),
            ("native_boosted", boosted_pcm, self.sample_rate),
        ]
        if int(round(self.sample_rate)) != 16000:
            variants.extend(
                [
                    ("16k", self._resample_pcm(pcm, self.sample_rate, 16000), 16000),
                    ("16k_boosted", self._resample_pcm(boosted_pcm, self.sample_rate, 16000), 16000),
                ]
            )

        candidates: list[str] = []
        for label, audio, rate in variants:
            try:
                text = self._transcribe_with_vosk(audio, rate, grammar)
            except Exception as e:
                text = f"{label}:error:{e}"
            if text and text not in candidates:
                candidates.append(text)

        for text in candidates:
            if self._has_water_word(text):
                return text, candidates
        return (candidates[0] if candidates else ""), candidates

    def _call_http_command_api(self, pcm: bytes) -> tuple[bool, str, str]:
        payload = {
            "audio_wav_b64": self._pcm_to_wav_b64(pcm),
            "sample_rate": int(round(self.sample_rate)),
            "default_task": self.default_task,
        }
        request = urllib.request.Request(
            self.command_api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.command_api_timeout_s) as response:
            parsed = json.loads(response.read().decode("utf-8"))

        text = str(
            parsed.get("text")
            or parsed.get("transcript")
            or parsed.get("command")
            or parsed.get("user_command")
            or ""
        )
        intent = str(parsed.get("intent") or parsed.get("action") or "").lower()
        accepted = bool(
            parsed.get("accepted")
            or parsed.get("trigger")
            or parsed.get("should_grasp")
            or intent in {"water", "grasp_water", "bring_water", "grasp"}
            or self._task_for_command(text)
        )
        if self.require_water_word:
            accepted = accepted and self._has_water_word(text)
        return accepted, text, f"http command API response: {intent or ('accepted' if accepted else 'ignored')}"

    def _call_gemini_command_api(self, pcm: bytes) -> tuple[bool, str, str]:
        api_key = self._gemini_api_key()
        if not api_key:
            return False, "", "Gemini API key is not configured"

        from google import genai
        from google.genai import types

        wav_bytes = self._pcm_to_wav_bytes(pcm)
        client = genai.Client(api_key=api_key)
        prompt = (
            "You are an intent classifier for a robot arm.\n"
            "Transcribe the audio. Only accept the command if you clearly hear the English word "
            '"water" in the user command.\n'
            "Return only compact JSON with this schema:\n"
            '{"accepted": true|false, "transcript": "...", "task": "Grasp the water cup"}\n'
            'Use accepted=false for silence, unclear audio, or commands that do not contain "water". '
            'If accepted=true, task must be "Grasp the water cup".'
        )
        response = client.models.generate_content(
            model=self.gemini_model,
            contents=[
                prompt,
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
            ],
        )
        text_response = str(getattr(response, "text", "") or "").strip()
        self._set_status(last_command_raw_response=text_response[:500])
        try:
            parsed = self._load_jsonish(text_response)
        except Exception:
            command_text = text_response
            accepted = self._has_water_word(command_text)
            reason = "Gemini transcript contains water" if accepted else "Gemini returned unparseable command response"
            return accepted, command_text, reason
        command_text = str(
            parsed.get("transcript")
            or parsed.get("text")
            or parsed.get("command")
            or ""
        ).strip()
        accepted = self._has_water_word(command_text)
        reason = "Gemini transcript contains water" if accepted else "Gemini did not hear water"
        return accepted, command_text, reason

    def _call_command_api(self, pcm: bytes, peak_rms: float) -> tuple[bool, str, str]:
        local_text, local_candidates = self._transcribe_command_locally(pcm)
        self._set_status(last_local_command_text=local_text, last_local_command_candidates=local_candidates)
        if self._has_water_word(local_text):
            return True, local_text, "local command recognizer heard water"
        if self.command_provider == "local":
            return False, local_text, "local command recognizer did not hear water"

        if self.command_provider in {"auto", "gemini"} and self._gemini_api_key():
            try:
                accepted, text, reason = self._call_gemini_command_api(pcm)
            except Exception as e:
                return False, local_text, f"Gemini unavailable and water was not heard locally: {e}"
            if accepted:
                return accepted, text, reason
            if (
                self.accept_speech_after_wake
                and not self.require_water_word
                and peak_rms >= self.command_audio_rms_threshold
            ):
                fallback_text = text or "microphone command audio"
                return True, fallback_text, f"{reason}; fallback accepted speech after wake"
            return accepted, text, reason
        if self.command_provider == "gemini":
            return False, "", "Gemini API key is not configured"

        if self.command_provider in {"auto", "http"} and self.command_api_url:
            return self._call_http_command_api(pcm)
        if self.command_provider == "http":
            return False, "", "HTTP command API URL is not configured"

        if peak_rms < self.command_audio_rms_threshold:
            return False, "", "no speech detected after wake"
        if self.require_water_word:
            return False, local_text, "command ignored because water was not heard"
        return True, "microphone command audio", "no command API configured; using default water task"

    def _handle_wake_detected(self, wake_text: str):
        self._set_status(
            last_text=wake_text,
            partial_text="",
            last_command_text="",
            last_local_command_text="",
            last_local_command_candidates=[],
            last_event="wake word detected",
        )
        if time.time() < self._ignore_wake_until:
            self._set_status(mode="wake", last_event="wake ignored during cooldown", command_deadline=0.0)
            print("[VOICE] Wake ignored during cooldown.")
            return
        if self.runner.get_status().get("busy"):
            self._set_status(mode="wake", last_event="wake ignored while robot workflow is busy", command_deadline=0.0)
            print("[VOICE] Wake ignored while robot workflow is busy.")
            return
        self._ignore_wake_until = time.time() + self.wake_cooldown_s
        if self._has_water_word(wake_text):
            print("[VOICE] Wake phrase already contains water. Triggering API workflow.")
            self._set_status(
                mode="triggering",
                last_command_text=wake_text,
                last_local_command_text=wake_text,
                last_event="wake phrase contained water",
                command_deadline=0.0,
            )
            response = self.runner.try_trigger(
                TriggerRequest(
                    api_command=self.api_command,
                    user_command=wake_text,
                    task=self.default_task,
                    max_steps=self.max_steps,
                )
            )
            print(f"[VOICE] API response: {json.dumps(response, ensure_ascii=True)}")
            self._set_status(mode="wake", last_event=f"api response: {response.get('status')}")
            return
        pcm, peak_rms = self._capture_command_audio()
        self._set_status(mode="command_api", last_event="sending command audio to API")
        try:
            accepted, command_text, reason = self._call_command_api(pcm, peak_rms)
        except Exception as e:
            self._set_status(mode="wake", last_event=f"command API error: {e}", command_deadline=0.0)
            print(f"[VOICE] Command API error: {e}")
            return

        self._set_status(last_command_text=command_text, last_event=reason)
        if not accepted:
            print(f"[VOICE] Command ignored: {reason}")
            self._set_status(mode="wake", command_deadline=0.0)
            return

        print(f"[VOICE] Command accepted: {reason}. Triggering API workflow.")
        self._ignore_wake_until = time.time() + self.wake_cooldown_s
        self._set_status(mode="triggering", last_event=reason)
        response = self.runner.try_trigger(
            TriggerRequest(
                api_command=self.api_command,
                user_command=command_text or "microphone command audio",
                task=self.default_task,
                max_steps=self.max_steps,
            )
        )
        print(f"[VOICE] API response: {json.dumps(response, ensure_ascii=True)}")
        self._set_status(
            mode="wake",
            last_event=f"api response: {response.get('status')}",
            command_deadline=0.0,
        )

    def listen_forever(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Vosk model not found: {self.model_path}\n"
                "Expected model folder: voice_control/models/vosk-model-small-en-us-0.15"
            )

        if self.input_device is None:
            self.input_device = _pick_input_device()
        device_info = None
        if self.input_device is not None:
            device_info = sd.query_devices(self.input_device)
            print(f"[VOICE] Using input device {self.input_device}: {device_info['name']}")
        if self.sample_rate <= 0:
            if device_info is not None:
                self.sample_rate = float(device_info.get("default_samplerate") or SAMPLE_RATE)
            else:
                self.sample_rate = float(SAMPLE_RATE)
        block_size = max(1, int(round(self.sample_rate * 0.5)))

        # Keep Vosk local and tiny: it only owns the wake word.
        # After wake, command audio is recorded from the same microphone and sent to the command API path.
        grammar = sorted(self.wake_phrases | DEFAULT_COMBINED_WAKE_WATER_PHRASES | {"stop"}) + ["[unk]"]
        self._vosk_model = Model(self.model_path)
        recognizer = KaldiRecognizer(self._vosk_model, self.sample_rate, json.dumps(grammar))

        print("[VOICE] Listening. Say: HEY RUDI")
        self._set_status(
            mode="wake",
            last_event="listening for HEY RUDI",
            command_deadline=0.0,
            sample_rate=self.sample_rate,
            command_provider=self.command_provider,
            gemini_model=self.gemini_model,
            accept_speech_after_wake=self.accept_speech_after_wake,
            require_water_word=self.require_water_word,
            command_preroll_s=self.command_preroll_s,
            command_api_configured=self._command_api_configured(),
        )

        with sd.RawInputStream(
            device=self.input_device,
            samplerate=self.sample_rate,
            blocksize=block_size,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        ):
            while True:
                try:
                    data = self.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if not recognizer.AcceptWaveform(data):
                    partial = json.loads(recognizer.PartialResult()).get("partial", "").strip().lower()
                    if partial:
                        self._set_status(partial_text=partial)
                        if self._is_wake_text(partial):
                            print("[VOICE] Wake word detected from partial.")
                            recognizer.Reset()
                            self._handle_wake_detected(partial)
                    continue

                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip().lower()
                if not text:
                    continue
                self._set_status(last_text=text, partial_text="")

                if text == "stop":
                    if self.runner.get_status().get("busy"):
                        print("[VOICE] Stop heard while job is running. Finish the current robot motion first.")
                        self._set_status(last_event="stop heard while job is running")
                        continue
                    print("[VOICE] Stop heard. Exiting voice workflow.")
                    self._set_status(mode="stopping", last_event="stop heard")
                    break

                if self._is_wake_text(text):
                    print("[VOICE] Wake word detected.")
                    recognizer.Reset()
                    self._handle_wake_detected(text)
                    continue


def _load_command_map(value: str | None, default_task: str) -> dict[str, str]:
    if not value:
        return dict(DEFAULT_WATER_COMMANDS)
    parsed = json.loads(value)
    if isinstance(parsed, list):
        return {str(item).lower(): default_task for item in parsed}
    if isinstance(parsed, dict):
        return {str(k).lower(): str(v) for k, v in parsed.items()}
    raise ValueError("--voice-command-map-json must be a JSON object or list.")


def _load_wake_phrases(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_WAKE_PHRASES)
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("--wake-phrases-json must be a JSON list.")
    return [str(item).lower() for item in parsed]


def main():
    parser = argparse.ArgumentParser(description="Voice wake + API-triggered SmolVLA grasp workflow")
    parser.add_argument("--model-path", required=True, help="Fine-tuned SmolVLA model path/repo.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="my_follower")
    parser.add_argument(
        "--agent-camera",
        default="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6:1.0-video-index0",
    )
    parser.add_argument(
        "--wrist-camera",
        default="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:5.1:1.0-video-index0",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--agent-rotation", type=int, default=0)
    parser.add_argument("--wrist-rotation", type=int, default=180)
    parser.add_argument("--p-coefficient", type=int, default=20)
    parser.add_argument("--rename-map", default=json.dumps(DEFAULT_RENAME_MAP, ensure_ascii=True))
    parser.add_argument(
        "--stats-path",
        default=str(Path.home() / ".cache/huggingface/lerobot/ima/so101_grasp_cup/meta/stats.json"),
    )
    parser.add_argument("--wake-command", default="start_grasp")
    parser.add_argument("--default-task", default="Grasp the water cup")
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--control-fps", type=float, default=12.0)
    parser.add_argument("--action-deadband-deg", type=float, default=0.2)
    parser.add_argument("--boot-pose-file", default=str(DEFAULT_BOOT_POSE_FILE))
    parser.add_argument("--preset1-pose-file", default=str(DEFAULT_PRESET1_POSE_FILE))
    parser.add_argument("--preset2-pose-file", default=str(DEFAULT_PRESET2_POSE_FILE))
    parser.add_argument(
        "--preset3-pose-file",
        default="",
        help="Legacy argument. This workflow uses preset pose 2 as the final transport pose by default.",
    )
    parser.add_argument("--pose-steps", type=int, default=80)
    parser.add_argument("--pose-delay-s", type=float, default=0.02)
    parser.add_argument("--post-grasp-move-speed-scale", type=float, default=0.5)
    parser.add_argument("--post-grasp-hold-s", type=float, default=15.0)
    parser.add_argument("--release-gripper-steps", type=int, default=50)
    parser.add_argument("--release-gripper-delay-s", type=float, default=0.02)
    parser.add_argument("--no-post-grasp-keep-gripper-closed", action="store_true")
    parser.add_argument("--no-return-home-after-job", action="store_true")
    parser.add_argument("--no-release-gripper-after-home", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8800)

    parser.add_argument("--voice-model-path", default=VOSK_MODEL_PATH)
    parser.add_argument("--voice-input-device", type=int, default=None)
    parser.add_argument(
        "--voice-sample-rate",
        type=float,
        default=0.0,
        help="Microphone sample rate. 0 means use the selected device default.",
    )
    parser.add_argument(
        "--voice-command-timeout-s",
        type=float,
        default=8.0,
        help="Maximum seconds to record command audio after the local wake word.",
    )
    parser.add_argument(
        "--wake-cooldown-s",
        type=float,
        default=5.0,
        help="Ignore repeated wake detections for this many seconds after a wake/trigger.",
    )
    parser.add_argument(
        "--command-preroll-s",
        type=float,
        default=1.0,
        help="Keep this much already-buffered audio after wake, useful if the user says HEY RUDI GIVE ME WATER.",
    )
    parser.add_argument(
        "--command-api-url",
        default="",
        help=(
            "Optional API URL that receives {'audio_wav_b64', 'sample_rate', 'default_task'} "
            "and returns accepted/trigger/should_grasp plus optional transcript."
        ),
    )
    parser.add_argument("--command-api-timeout-s", type=float, default=8.0)
    parser.add_argument(
        "--command-provider",
        choices=("auto", "gemini", "http", "local", "rms"),
        default="auto",
        help="How to interpret command audio after HEY RUDI. auto uses Gemini if GEMINI_API_KEY is set.",
    )
    parser.add_argument(
        "--gemini-model",
        default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model used for post-wake command audio intent classification.",
    )
    parser.add_argument(
        "--accept-speech-after-wake",
        action="store_true",
        default=False,
        help="For a loose demo only: accept post-wake speech as the default water task if Gemini is unsure.",
    )
    parser.add_argument(
        "--no-accept-speech-after-wake",
        dest="accept_speech_after_wake",
        action="store_false",
        help="Require Gemini or the HTTP command API to explicitly accept the command.",
    )
    parser.add_argument(
        "--require-water-word",
        action="store_true",
        default=True,
        help='Only trigger when the command recognizer/Gemini transcript contains the word "water".',
    )
    parser.add_argument(
        "--no-require-water-word",
        dest="require_water_word",
        action="store_false",
        help='Allow broader water/cup/bottle intent matching without the exact word "water".',
    )
    parser.add_argument(
        "--command-audio-rms-threshold",
        type=float,
        default=5.0,
        help="If no command API URL is configured, command audio above this RMS triggers the default water task.",
    )
    parser.add_argument(
        "--command-silence-s",
        type=float,
        default=0.9,
        help="Stop command recording after this much silence following speech.",
    )
    parser.add_argument(
        "--wake-phrases-json",
        default="",
        help='Optional JSON list, e.g. ["hey rudi", "rudi"].',
    )
    parser.add_argument(
        "--voice-command-map-json",
        default="",
        help='Optional JSON object/list for commands after wake, e.g. {"give me water":"Grasp the water cup"}.',
    )
    args = parser.parse_args()

    rename_map = json.loads(args.rename_map)
    if not isinstance(rename_map, dict):
        raise ValueError("--rename-map must be a JSON object.")

    wrist_camera = args.wrist_camera
    if isinstance(wrist_camera, str) and wrist_camera.strip().lower() in {"", "none", "null"}:
        wrist_camera = None

    stats_path = args.stats_path.strip() if isinstance(args.stats_path, str) else None
    if stats_path == "":
        stats_path = None

    runner = WorkflowRunner(
        model_path=args.model_path,
        device=args.device,
        robot_port=args.robot_port,
        robot_id=args.robot_id,
        agent_camera=args.agent_camera,
        wrist_camera=wrist_camera,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        agent_rotation=args.agent_rotation,
        wrist_rotation=args.wrist_rotation,
        p_coefficient=args.p_coefficient,
        rename_map=rename_map,
        stats_path=stats_path,
        wake_command=args.wake_command,
        default_task=args.default_task,
        max_steps=args.max_steps,
        control_fps=args.control_fps,
        action_deadband_deg=args.action_deadband_deg,
        boot_pose_file=args.boot_pose_file,
        preset1_pose_file=args.preset1_pose_file,
        preset2_pose_file=args.preset2_pose_file,
        preset3_pose_file=args.preset3_pose_file,
        pose_steps=args.pose_steps,
        pose_delay_s=args.pose_delay_s,
        post_grasp_move_speed_scale=args.post_grasp_move_speed_scale,
        post_grasp_keep_gripper_closed=not args.no_post_grasp_keep_gripper_closed,
        post_grasp_hold_s=args.post_grasp_hold_s,
        return_home_after_job=not args.no_return_home_after_job,
        release_gripper_after_home=not args.no_release_gripper_after_home,
        release_gripper_steps=args.release_gripper_steps,
        release_gripper_delay_s=args.release_gripper_delay_s,
    )

    server = None
    server_thread = None
    try:
        runner.initialize()

        WorkflowApiHandler.runner = runner
        server = ThreadingHTTPServer((args.host, args.port), WorkflowApiHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        print(f"[API] Listening on http://{args.host}:{args.port}")

        voice = VoiceWakeController(
            runner=runner,
            model_path=args.voice_model_path,
            input_device=args.voice_input_device,
            api_command=args.wake_command,
            default_task=args.default_task,
            max_steps=args.max_steps,
            wake_phrases=_load_wake_phrases(args.wake_phrases_json),
            command_map=_load_command_map(args.voice_command_map_json, args.default_task),
            command_timeout_s=args.voice_command_timeout_s,
            sample_rate=args.voice_sample_rate,
            command_api_url=args.command_api_url,
            command_api_timeout_s=args.command_api_timeout_s,
            command_provider=args.command_provider,
            gemini_model=args.gemini_model,
            accept_speech_after_wake=args.accept_speech_after_wake,
            require_water_word=args.require_water_word,
            wake_cooldown_s=args.wake_cooldown_s,
            command_preroll_s=args.command_preroll_s,
            command_audio_rms_threshold=args.command_audio_rms_threshold,
            command_silence_s=args.command_silence_s,
        )
        WorkflowApiHandler.voice_status_provider = voice.get_status
        voice.listen_forever()
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down...")
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=2.0)
        runner.shutdown()


if __name__ == "__main__":
    main()
