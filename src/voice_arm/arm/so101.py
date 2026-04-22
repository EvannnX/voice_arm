from __future__ import annotations

from .controller import ArmController, ArmState, Speed


class SO101Arm(ArmController):
    """Stub for the Hiwonder LeRobot SO-101. Implemented in a later milestone.

    Planned: wrap the `lerobot` package's SO-101 client, map our Cartesian
    tool calls to the arm's joint-space controller with workspace clamping.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
        raise NotImplementedError(
            "SO-101 driver is not implemented yet. "
            "Set ARM_BACKEND=mock to run the voice layer."
        )

    async def move_to(self, x_mm: float, y_mm: float, z_mm: float) -> ArmState:
        raise NotImplementedError

    async def move_relative(self, dx_mm: float, dy_mm: float, dz_mm: float) -> ArmState:
        raise NotImplementedError

    async def grasp(self, force: float | None = None) -> ArmState:
        raise NotImplementedError

    async def release(self) -> ArmState:
        raise NotImplementedError

    async def home(self) -> ArmState:
        raise NotImplementedError

    async def stop(self) -> ArmState:
        raise NotImplementedError

    async def set_speed(self, level: Speed) -> ArmState:
        raise NotImplementedError

    async def get_state(self) -> ArmState:
        raise NotImplementedError
