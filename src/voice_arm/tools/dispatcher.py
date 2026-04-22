from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from ..arm import ArmController, ArmState
from ..config import (
    MAX_RELATIVE_STEP_MM,
    WORKSPACE_X_MM,
    WORKSPACE_Y_MM,
    WORKSPACE_Z_MM,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    state: dict[str, Any] | None = None
    error: str | None = None

    def as_response(self) -> dict[str, Any]:
        if self.ok:
            return {"status": "ok", "state": self.state}
        return {"status": "error", "error": self.error}


def _state_dict(state: ArmState) -> dict[str, Any]:
    return asdict(state)


def _require(args: Mapping[str, Any], key: str) -> Any:
    if key not in args:
        raise ValueError(f"missing required argument: {key}")
    return args[key]


def _clamp_check(value: float, bounds: tuple[float, float], name: str) -> float:
    lo, hi = bounds
    if not (lo <= value <= hi):
        raise ValueError(f"{name}={value} is outside workspace [{lo}, {hi}]")
    return value


def _validate_relative(dx: float, dy: float, dz: float) -> None:
    for axis, v in (("dx_mm", dx), ("dy_mm", dy), ("dz_mm", dz)):
        if abs(v) > MAX_RELATIVE_STEP_MM:
            raise ValueError(
                f"{axis}={v} exceeds max step {MAX_RELATIVE_STEP_MM} mm"
            )


class ToolDispatcher:
    """Routes a Gemini tool_call to the ArmController, validating args at the boundary."""

    def __init__(self, arm: ArmController) -> None:
        self._arm = arm

    async def dispatch(self, name: str, args: Mapping[str, Any] | None) -> ToolResult:
        args = dict(args or {})
        try:
            state = await self._dispatch(name, args)
        except ValueError as exc:
            logger.warning("tool %s rejected: %s", name, exc)
            return ToolResult(ok=False, error=str(exc))
        except NotImplementedError as exc:
            logger.error("tool %s not implemented: %s", name, exc)
            return ToolResult(ok=False, error=str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("tool %s crashed", name)
            return ToolResult(ok=False, error=f"internal error: {exc}")
        return ToolResult(ok=True, state=_state_dict(state))

    async def _dispatch(self, name: str, args: dict[str, Any]) -> ArmState:
        match name:
            case "move_to":
                x = _clamp_check(float(_require(args, "x_mm")), WORKSPACE_X_MM, "x_mm")
                y = _clamp_check(float(_require(args, "y_mm")), WORKSPACE_Y_MM, "y_mm")
                z = _clamp_check(float(_require(args, "z_mm")), WORKSPACE_Z_MM, "z_mm")
                return await self._arm.move_to(x, y, z)
            case "move_relative":
                dx = float(_require(args, "dx_mm"))
                dy = float(_require(args, "dy_mm"))
                dz = float(_require(args, "dz_mm"))
                _validate_relative(dx, dy, dz)
                return await self._arm.move_relative(dx, dy, dz)
            case "grasp":
                force = args.get("force")
                return await self._arm.grasp(float(force) if force is not None else None)
            case "release":
                return await self._arm.release()
            case "home":
                return await self._arm.home()
            case "stop":
                return await self._arm.stop()
            case "set_speed":
                level = str(_require(args, "level"))
                if level not in {"slow", "normal", "fast"}:
                    raise ValueError(f"unknown speed level: {level}")
                return await self._arm.set_speed(level)  # type: ignore[arg-type]
            case "get_state":
                return await self._arm.get_state()
            case _:
                raise ValueError(f"unknown tool: {name}")
