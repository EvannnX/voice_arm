from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Literal

Speed = Literal["slow", "normal", "fast"]


@dataclass(frozen=True)
class ArmState:
    x_mm: float
    y_mm: float
    z_mm: float
    gripper_closed: bool
    speed: Speed

    def with_position(self, x: float, y: float, z: float) -> "ArmState":
        return replace(self, x_mm=x, y_mm=y, z_mm=z)

    def with_gripper(self, closed: bool) -> "ArmState":
        return replace(self, gripper_closed=closed)

    def with_speed(self, speed: Speed) -> "ArmState":
        return replace(self, speed=speed)


class ArmError(RuntimeError):
    pass


class ArmController(ABC):
    @abstractmethod
    async def move_to(self, x_mm: float, y_mm: float, z_mm: float) -> ArmState: ...

    @abstractmethod
    async def move_relative(self, dx_mm: float, dy_mm: float, dz_mm: float) -> ArmState: ...

    @abstractmethod
    async def grasp(self, force: float | None = None) -> ArmState: ...

    @abstractmethod
    async def release(self) -> ArmState: ...

    @abstractmethod
    async def home(self) -> ArmState: ...

    @abstractmethod
    async def stop(self) -> ArmState: ...

    @abstractmethod
    async def set_speed(self, level: Speed) -> ArmState: ...

    @abstractmethod
    async def get_state(self) -> ArmState: ...
