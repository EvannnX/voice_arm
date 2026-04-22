from __future__ import annotations

import asyncio
import logging

from .controller import ArmController, ArmState, Speed

logger = logging.getLogger(__name__)

HOME_POSE = ArmState(x_mm=0.0, y_mm=0.0, z_mm=200.0, gripper_closed=False, speed="normal")


class MockArm(ArmController):
    """In-memory arm used for development and unit tests.

    Every method returns a new ArmState (immutable). Motions are simulated
    with a small async sleep so the dispatcher path is exercised realistically.
    """

    def __init__(self, initial: ArmState = HOME_POSE) -> None:
        self._state = initial
        self._lock = asyncio.Lock()

    async def _set(self, new_state: ArmState) -> ArmState:
        async with self._lock:
            self._state = new_state
            return self._state

    async def move_to(self, x_mm: float, y_mm: float, z_mm: float) -> ArmState:
        logger.info("mock.move_to(%.1f, %.1f, %.1f)", x_mm, y_mm, z_mm)
        await asyncio.sleep(0.05)
        return await self._set(self._state.with_position(x_mm, y_mm, z_mm))

    async def move_relative(self, dx_mm: float, dy_mm: float, dz_mm: float) -> ArmState:
        logger.info("mock.move_relative(%.1f, %.1f, %.1f)", dx_mm, dy_mm, dz_mm)
        await asyncio.sleep(0.05)
        current = self._state
        return await self._set(
            current.with_position(
                current.x_mm + dx_mm,
                current.y_mm + dy_mm,
                current.z_mm + dz_mm,
            )
        )

    async def grasp(self, force: float | None = None) -> ArmState:
        logger.info("mock.grasp(force=%s)", force)
        await asyncio.sleep(0.02)
        return await self._set(self._state.with_gripper(True))

    async def release(self) -> ArmState:
        logger.info("mock.release()")
        await asyncio.sleep(0.02)
        return await self._set(self._state.with_gripper(False))

    async def home(self) -> ArmState:
        logger.info("mock.home()")
        await asyncio.sleep(0.05)
        return await self._set(HOME_POSE)

    async def stop(self) -> ArmState:
        logger.warning("mock.stop() — emergency stop")
        return await self._set(self._state)

    async def set_speed(self, level: Speed) -> ArmState:
        logger.info("mock.set_speed(%s)", level)
        return await self._set(self._state.with_speed(level))

    async def get_state(self) -> ArmState:
        return self._state
