#!/usr/bin/env python3
"""
API-triggered grasp workflow for SO-101 + SmolVLA (no extra dependencies).

Workflow:
1) startup -> move to boot/start pose
2) wait for API wake command
3) move to preset pose 1
4) run SmolVLA grasp policy
5) move to preset pose 2, hold, return to pose 0, release gripper
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from smolvla_infer import (
    DEFAULT_RENAME_MAP,
    JOINT_STATE_KEYS,
    load_policy,
    make_robot,
    maybe_move_to_start_pose,
    run_inference,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_BOOT_POSE_FILE = DATA_DIR / "api_preset_pose_0.json"
DEFAULT_PRESET1_POSE_FILE = DATA_DIR / "api_preset_pose_1.json"
DEFAULT_PRESET2_POSE_FILE = DATA_DIR / "api_preset_pose_2.json"
DEFAULT_PRESET3_POSE_FILE = DATA_DIR / "api_preset_pose_3.json"


@dataclass
class TriggerRequest:
    api_command: str
    user_command: str | None = None
    task: str | None = None
    max_steps: int | None = None

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "TriggerRequest":
        return TriggerRequest(
            api_command=str(payload.get("api_command", "")),
            user_command=payload.get("user_command"),
            task=payload.get("task"),
            max_steps=payload.get("max_steps"),
        )


@dataclass
class JobState:
    job_id: int = 0
    status: str = "idle"  # idle | running | done | error
    message: str = ""
    task: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0


class WorkflowRunner:
    def __init__(
        self,
        model_path: str,
        device: str,
        robot_port: str,
        robot_id: str,
        agent_camera: str,
        wrist_camera: str | None,
        camera_width: int,
        camera_height: int,
        camera_fps: int,
        agent_rotation: int,
        wrist_rotation: int,
        p_coefficient: int,
        rename_map: dict[str, str],
        stats_path: str | None,
        wake_command: str,
        default_task: str,
        max_steps: int,
        control_fps: float,
        action_deadband_deg: float,
        boot_pose_file: str,
        preset1_pose_file: str,
        preset2_pose_file: str,
        preset3_pose_file: str | None,
        pose_steps: int,
        pose_delay_s: float,
        post_grasp_move_speed_scale: float,
        post_grasp_keep_gripper_closed: bool,
        post_grasp_hold_s: float,
        return_home_after_job: bool,
        release_gripper_after_home: bool,
        release_gripper_steps: int,
        release_gripper_delay_s: float,
    ):
        self.model_path = model_path
        self.device = device
        self.robot_port = robot_port
        self.robot_id = robot_id
        self.agent_camera = agent_camera
        self.wrist_camera = wrist_camera
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.agent_rotation = agent_rotation
        self.wrist_rotation = wrist_rotation
        self.p_coefficient = p_coefficient
        self.rename_map = rename_map
        self.stats_path = stats_path
        self.wake_command = wake_command
        self.default_task = default_task
        self.max_steps = max_steps
        self.control_fps = control_fps
        self.action_deadband_deg = action_deadband_deg
        self.boot_pose_file = boot_pose_file
        self.preset1_pose_file = preset1_pose_file
        self.preset2_pose_file = preset2_pose_file
        self.preset3_pose_file = preset3_pose_file
        self.pose_steps = pose_steps
        self.pose_delay_s = pose_delay_s
        self.post_grasp_move_speed_scale = post_grasp_move_speed_scale
        self.post_grasp_keep_gripper_closed = post_grasp_keep_gripper_closed
        self.post_grasp_hold_s = post_grasp_hold_s
        self.return_home_after_job = return_home_after_job
        self.release_gripper_after_home = release_gripper_after_home
        self.release_gripper_steps = release_gripper_steps
        self.release_gripper_delay_s = release_gripper_delay_s

        self.robot = None
        self.policy_runtime = None

        self._lock = threading.Lock()
        self._busy = False
        self.state = JobState()
        self._shutdown_done = False

    def initialize(self):
        print("[INIT] Loading SmolVLA policy...")
        self.policy_runtime = load_policy(
            model_path=self.model_path,
            device=self.device,
            rename_map=self.rename_map,
            stats_path=self.stats_path,
            robot_type="so101_follower",
        )
        print("[INIT] Policy loaded.")

        print("[INIT] Connecting robot...")
        self.robot = make_robot(
            port=self.robot_port,
            robot_id=self.robot_id,
            agent_camera=self.agent_camera,
            wrist_camera=self.wrist_camera,
            width=self.camera_width,
            height=self.camera_height,
            fps=self.camera_fps,
            agent_rotation=self.agent_rotation,
            wrist_rotation=self.wrist_rotation,
            p_coefficient=self.p_coefficient,
        )
        print("[INIT] Robot connected.")

        self._move_to_required_pose(self.boot_pose_file, "boot/start")
        print("[INIT] Boot pose reached.")

        if self.preset3_pose_file:
            print("[INIT] Preset pose 3 is ignored for this workflow; final transport uses pose 2.")

    def _move_to_required_pose(
        self,
        pose_file: str,
        label: str,
        speed_scale: float = 1.0,
        keep_gripper_closed: bool = False,
    ):
        ok = maybe_move_to_start_pose(
            self.robot,
            pose_file=pose_file,
            steps=self.pose_steps,
            delay_s=self.pose_delay_s,
            speed_scale=speed_scale,
            keep_gripper_closed=keep_gripper_closed,
        )
        if not ok:
            raise FileNotFoundError(
                f"{label} pose file not found: {Path(pose_file).expanduser().resolve()}"
            )

    @staticmethod
    def _read_pose_vector(pose_file: str) -> list[float]:
        path = Path(pose_file).expanduser().resolve()
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            pose = payload
        elif isinstance(payload, dict):
            pose = payload.get("start_pose") or payload.get("joint_positions")
        else:
            pose = None
        if not pose:
            raise ValueError(f"pose file must contain start_pose or joint_positions: {path}")
        return [float(v) for v in pose]

    def _read_current_gripper(self) -> float:
        obs = self.robot.get_observation()
        if "gripper.pos" in obs:
            return float(obs["gripper.pos"])
        if all(k in obs for k in JOINT_STATE_KEYS):
            return float(obs[JOINT_STATE_KEYS[-1]])
        raise KeyError(f"gripper.pos not found in observation keys: {list(obs.keys())}")

    def _release_gripper_to_pose_file(self, pose_file: str):
        target_pose = self._read_pose_vector(pose_file)
        target_gripper = float(target_pose[-1])
        current_gripper = self._read_current_gripper()
        steps = max(1, int(self.release_gripper_steps))
        delay_s = max(0.0, float(self.release_gripper_delay_s))

        print(f"[GRIPPER] Releasing to pose-file gripper value: {target_gripper:.2f}")
        for i in range(1, steps + 1):
            alpha = i / steps
            value = current_gripper + (target_gripper - current_gripper) * alpha
            self.robot.send_action({"gripper.pos": float(value)})
            if delay_s > 0:
                time.sleep(delay_s)

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "busy": self._busy,
                "state": {
                    "job_id": self.state.job_id,
                    "status": self.state.status,
                    "message": self.state.message,
                    "task": self.state.task,
                    "started_at": self.state.started_at,
                    "finished_at": self.state.finished_at,
                },
            }

    def try_trigger(self, req: TriggerRequest) -> dict[str, Any]:
        if req.api_command != self.wake_command:
            return {
                "accepted": False,
                "status": "ignored",
                "message": f"api_command '{req.api_command}' != wake_command '{self.wake_command}'",
            }

        with self._lock:
            if self._busy:
                return {
                    "accepted": False,
                    "status": "busy",
                    "message": "workflow is running",
                    "job_id": self.state.job_id,
                }

            self._busy = True
            self.state.job_id += 1
            self.state.status = "running"
            self.state.message = "accepted"
            self.state.task = req.task or self.default_task
            self.state.started_at = time.time()
            self.state.finished_at = 0.0
            job_id = self.state.job_id
            task = self.state.task
            max_steps = int(req.max_steps) if req.max_steps is not None else self.max_steps

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, task, max_steps),
            daemon=True,
        )
        thread.start()

        return {
            "accepted": True,
            "status": "running",
            "job_id": job_id,
            "task": task,
            "max_steps": max_steps,
        }

    def _run_job(self, job_id: int, task: str, max_steps: int):
        try:
            print(f"[JOB {job_id}] Move to preset pose 1.")
            self._move_to_required_pose(self.preset1_pose_file, "preset pose 1")

            print(f"[JOB {job_id}] Run grasp policy.")
            run_inference(
                self.policy_runtime,
                self.robot,
                task_instruction=task,
                max_steps=max_steps,
                control_fps=self.control_fps,
                action_deadband_deg=self.action_deadband_deg,
            )

            final_pose_file = self.preset2_pose_file
            final_label = "preset pose 2"
            print(
                f"[JOB {job_id}] Move to {final_label} "
                f"(speed x{self.post_grasp_move_speed_scale:.2f}, "
                f"keep_gripper_closed={self.post_grasp_keep_gripper_closed})."
            )
            self._move_to_required_pose(
                final_pose_file,
                final_label,
                speed_scale=self.post_grasp_move_speed_scale,
                keep_gripper_closed=self.post_grasp_keep_gripper_closed,
            )

            if self.post_grasp_hold_s > 0:
                print(f"[JOB {job_id}] Hold at {final_label} for {self.post_grasp_hold_s:.1f}s.")
                time.sleep(self.post_grasp_hold_s)

            if self.return_home_after_job:
                print(
                    f"[JOB {job_id}] Return to pose 0 "
                    f"(speed x{self.post_grasp_move_speed_scale:.2f}, "
                    f"keep_gripper_closed={self.post_grasp_keep_gripper_closed})."
                )
                self._move_to_required_pose(
                    self.boot_pose_file,
                    "boot/start",
                    speed_scale=self.post_grasp_move_speed_scale,
                    keep_gripper_closed=self.post_grasp_keep_gripper_closed,
                )
                if self.release_gripper_after_home:
                    self._release_gripper_to_pose_file(self.boot_pose_file)

            with self._lock:
                self.state.status = "done"
                self.state.message = "grasp workflow completed"
                self.state.finished_at = time.time()
                self._busy = False
        except Exception as e:
            with self._lock:
                self.state.status = "error"
                self.state.message = str(e)
                self.state.finished_at = time.time()
                self._busy = False
            print(f"[JOB {job_id}] ERROR: {e}")

    def shutdown(self):
        if self._shutdown_done:
            return
        self._shutdown_done = True

        if self.robot is None:
            return

        # Always try to return to pose 0 when process exits.
        # This is best-effort: power loss / SIGKILL cannot be handled in user-space.
        try:
            print("[SHUTDOWN] Moving robot to boot/start pose (pose 0) before disconnect.")
            self._move_to_required_pose(self.boot_pose_file, "boot/start")
            print("[SHUTDOWN] Reached boot/start pose.")
        except Exception as e:
            print(f"[SHUTDOWN] Failed to move to boot/start pose: {e}")
        finally:
            try:
                self.robot.disconnect()
            except Exception as e:
                print(f"[SHUTDOWN] Robot disconnect warning: {e}")


class WorkflowApiHandler(BaseHTTPRequestHandler):
    runner: WorkflowRunner | None = None
    voice_status_provider: Callable[[], dict[str, Any]] | None = None

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str, status: int = HTTPStatus.OK):
        encoded = body.encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _status_payload(self) -> dict[str, Any]:
        payload = self.runner.get_status()
        if self.voice_status_provider is not None:
            payload["voice"] = self.voice_status_provider()
        return payload

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    @staticmethod
    def _is_grasp_water_command(text: str) -> bool:
        normalized = " ".join(str(text).lower().strip().split())
        if not normalized:
            return False
        words = set(normalized.split())
        if words & {"water", "cup", "bottle"}:
            return True
        return any(phrase in normalized for phrase in ("give me", "bring me", "get me", "i want"))

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if self.runner is None:
            self._send_json({"ok": False, "error": "runner not initialized"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if parsed.path in {"/", "/dashboard"}:
            self._send_html(DASHBOARD_HTML)
            return
        if parsed.path == "/health":
            self._send_json({"ok": True, **self._status_payload()})
            return
        if parsed.path == "/status":
            self._send_json(self._status_payload())
            return
        self._send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if self.runner is None:
            self._send_json({"ok": False, "error": "runner not initialized"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if parsed.path == "/command_text":
            try:
                payload = self._read_json_body()
                text = str(payload.get("text", ""))
                task = str(payload.get("task") or self.runner.default_task)
                if not self._is_grasp_water_command(text):
                    self._send_json(
                        {
                            "accepted": False,
                            "status": "ignored",
                            "message": "command text did not look like a water/cup request",
                            "text": text,
                        }
                    )
                    return
                self._send_json(
                    self.runner.try_trigger(
                        TriggerRequest(
                            api_command=self.runner.wake_command,
                            user_command=text,
                            task=task,
                        )
                    )
                )
            except json.JSONDecodeError:
                self._send_json({"ok": False, "error": "invalid json body"}, HTTPStatus.BAD_REQUEST)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if parsed.path != "/trigger":
            self._send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self._read_json_body()
            req = TriggerRequest.from_dict(payload)
            if not req.api_command:
                self._send_json({"ok": False, "error": "api_command is required"}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(self.runner.try_trigger(req))
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "invalid json body"}, HTTPStatus.BAD_REQUEST)
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt, *args):  # noqa: D401
        # Keep logs concise.
        print(f"[API] {self.address_string()} - {fmt % args}")


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RUDI Robot Status</title>
  <style>
    :root { color-scheme: light dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #111827; color: #f9fafb; }
    main { max-width: 880px; margin: 0 auto; padding: 28px; }
    h1 { font-size: 28px; margin: 0 0 18px; letter-spacing: 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .card { background: #1f2937; border: 1px solid #374151; border-radius: 8px; padding: 16px; }
    .label { color: #9ca3af; font-size: 13px; margin-bottom: 6px; }
    .value { font-size: 22px; font-weight: 700; overflow-wrap: anywhere; }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 16px 0; }
    button { padding: 10px 14px; border: 0; border-radius: 6px; background: #2563eb; color: white; font-weight: 700; cursor: pointer; }
    button:disabled { background: #4b5563; cursor: wait; }
    input { min-width: 240px; flex: 1; padding: 10px 12px; border-radius: 6px; border: 1px solid #4b5563; background: #111827; color: #f9fafb; }
    .ok { color: #86efac; }
    .warn { color: #fcd34d; }
    .bad { color: #fca5a5; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #030712; border: 1px solid #374151; border-radius: 8px; padding: 14px; }
  </style>
</head>
<body>
  <main>
    <h1>RUDI Robot Status</h1>
    <section class="grid">
      <div class="card"><div class="label">Robot</div><div id="robot" class="value">loading</div></div>
      <div class="card"><div class="label">Voice</div><div id="voice" class="value">loading</div></div>
      <div class="card"><div class="label">Job</div><div id="job" class="value">loading</div></div>
      <div class="card"><div class="label">Task</div><div id="task" class="value">loading</div></div>
    </section>
    <div class="controls">
      <input id="commandText" placeholder="type command, e.g. give me water">
      <button id="sendCommand">Send Command</button>
      <button id="trigger">Trigger Water Grasp</button>
    </div>
    <div class="card">
      <div class="label">Command API</div>
      <div id="speechStatus" class="value">backend microphone active</div>
    </div>
    <h2>Details</h2>
    <pre id="raw">loading</pre>
  </main>
  <script>
    function cls(status) {
      if (status === "error") return "value bad";
      if (status === "running" || status === "command_recording" || status === "command_api" || status === "triggering") return "value warn";
      return "value ok";
    }
    async function postCommandText(text) {
      const res = await fetch("/command_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, task: "Grasp the water cup" })
      });
      return await res.json();
    }

    async function refresh() {
      try {
        const res = await fetch("/status", { cache: "no-store" });
        const data = await res.json();
        const state = data.state || {};
        const voice = data.voice || {};
        const robot = document.getElementById("robot");
        const voiceEl = document.getElementById("voice");
        const job = document.getElementById("job");
        const task = document.getElementById("task");
        robot.textContent = data.busy ? "running" : "ready";
        robot.className = data.busy ? "value warn" : "value ok";
        voiceEl.textContent = voice.mode || "api-only";
        voiceEl.className = cls(voice.mode);
        job.textContent = state.status || "unknown";
        job.className = cls(state.status);
        task.textContent = state.task || "-";
        const commandApi = document.getElementById("speechStatus");
        if (voice.mode === "command_recording") {
          commandApi.textContent = "recording command from backend microphone";
          commandApi.className = "value warn";
        } else if (voice.mode === "command_api") {
          commandApi.textContent = "sending recorded command to API";
          commandApi.className = "value warn";
        } else {
          commandApi.textContent = voice.command_api_configured ? "external command API configured" : "demo mode: speech after wake triggers water task";
          commandApi.className = "value ok";
        }
        document.getElementById("raw").textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        document.getElementById("robot").textContent = "offline";
        document.getElementById("robot").className = "value bad";
        document.getElementById("raw").textContent = String(err);
      }
    }
    document.getElementById("trigger").addEventListener("click", async () => {
      const btn = document.getElementById("trigger");
      btn.disabled = true;
      try {
        await fetch("/trigger", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            api_command: "start_grasp",
            user_command: "dashboard",
            task: "Grasp the water cup"
          })
        });
      } finally {
        setTimeout(() => { btn.disabled = false; refresh(); }, 1000);
      }
    });
    document.getElementById("sendCommand").addEventListener("click", async () => {
      const text = document.getElementById("commandText").value.trim();
      if (!text) return;
      const result = await postCommandText(text);
      document.getElementById("speechStatus").textContent = "api: " + (result.status || "sent");
      refresh();
    });
    refresh();
    setInterval(refresh, 800);
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="API-triggered SmolVLA grasp workflow")
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
    parser.add_argument(
        "--rename-map",
        default=json.dumps(DEFAULT_RENAME_MAP, ensure_ascii=True),
        help="JSON mapping from robot observation keys to policy camera keys.",
    )
    parser.add_argument(
        "--stats-path",
        default=str(Path.home() / ".cache/huggingface/lerobot/ima/so101_grasp_cup/meta/stats.json"),
        help="stats.json path for normalization override (recommended for finetuned model).",
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
    parser.add_argument(
        "--post-grasp-move-speed-scale",
        type=float,
        default=0.5,
        help="Speed scale for movement after VLM grasp (0.5 means half speed).",
    )
    parser.add_argument(
        "--post-grasp-keep-gripper-closed",
        action="store_true",
        default=True,
        help="Keep current gripper value locked during post-grasp transport (default: enabled).",
    )
    parser.add_argument(
        "--no-post-grasp-keep-gripper-closed",
        dest="post_grasp_keep_gripper_closed",
        action="store_false",
        help="Disable gripper lock during post-grasp transport.",
    )
    parser.add_argument(
        "--post-grasp-hold-s",
        type=float,
        default=15.0,
        help="Seconds to hold at preset pose 2 after grasp transport.",
    )
    parser.add_argument(
        "--return-home-after-job",
        action="store_true",
        default=True,
        help="Return to pose 0 after holding at preset pose 2 (default: enabled).",
    )
    parser.add_argument(
        "--no-return-home-after-job",
        dest="return_home_after_job",
        action="store_false",
        help="Do not return to pose 0 after each job.",
    )
    parser.add_argument(
        "--release-gripper-after-home",
        action="store_true",
        default=True,
        help="Open/release gripper after returning to pose 0 (default: enabled).",
    )
    parser.add_argument(
        "--no-release-gripper-after-home",
        dest="release_gripper_after_home",
        action="store_false",
        help="Keep gripper closed after returning to pose 0.",
    )
    parser.add_argument("--release-gripper-steps", type=int, default=50)
    parser.add_argument("--release-gripper-delay-s", type=float, default=0.02)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8800)
    args = parser.parse_args()

    try:
        rename_map = json.loads(args.rename_map) if isinstance(args.rename_map, str) else args.rename_map
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid --rename-map JSON: {e}") from e
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
        post_grasp_keep_gripper_closed=args.post_grasp_keep_gripper_closed,
        post_grasp_hold_s=args.post_grasp_hold_s,
        return_home_after_job=args.return_home_after_job,
        release_gripper_after_home=args.release_gripper_after_home,
        release_gripper_steps=args.release_gripper_steps,
        release_gripper_delay_s=args.release_gripper_delay_s,
    )
    runner.initialize()

    WorkflowApiHandler.runner = runner
    server = ThreadingHTTPServer((args.host, args.port), WorkflowApiHandler)
    print(f"[API] Listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[API] Shutting down...")
    finally:
        server.shutdown()
        server.server_close()
        runner.shutdown()


if __name__ == "__main__":
    main()
