from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


INPUT_SAMPLE_RATE_HZ = 16_000
OUTPUT_SAMPLE_RATE_HZ = 24_000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH_BYTES = 2
FRAME_DURATION_MS = 20
FRAME_SIZE_BYTES = (
    INPUT_SAMPLE_RATE_HZ
    * FRAME_DURATION_MS
    // 1000
    * AUDIO_CHANNELS
    * AUDIO_SAMPLE_WIDTH_BYTES
)

WORKSPACE_X_MM = (-300.0, 300.0)
WORKSPACE_Y_MM = (-300.0, 300.0)
WORKSPACE_Z_MM = (0.0, 400.0)
MAX_RELATIVE_STEP_MM = 200.0
DEFAULT_NUDGE_MM = 20.0

EMERGENCY_STOP_WORDS = ("stop", "halt", "freeze", "emergency")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: SecretStr = Field(..., alias="GEMINI_API_KEY")
    gemini_model: str = Field(
        default="gemini-2.5-flash-preview-native-audio-dialog",
        alias="GEMINI_MODEL",
    )
    arm_backend: Literal["mock", "so101"] = Field(default="mock", alias="ARM_BACKEND")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    push_to_talk: bool = Field(default=False, alias="PUSH_TO_TALK")


def load_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
