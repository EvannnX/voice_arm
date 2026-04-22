import pytest

from voice_arm.arm import MockArm
from voice_arm.config import MAX_RELATIVE_STEP_MM, WORKSPACE_X_MM
from voice_arm.tools import ToolDispatcher


@pytest.fixture
def dispatcher():
    return ToolDispatcher(MockArm())


@pytest.mark.asyncio
async def test_move_to_happy_path(dispatcher):
    result = await dispatcher.dispatch(
        "move_to", {"x_mm": 10.0, "y_mm": 20.0, "z_mm": 100.0}
    )
    assert result.ok is True
    assert result.state["x_mm"] == 10.0
    assert result.state["y_mm"] == 20.0
    assert result.state["z_mm"] == 100.0


@pytest.mark.asyncio
async def test_move_to_rejects_out_of_workspace(dispatcher):
    bad_x = WORKSPACE_X_MM[1] + 1
    result = await dispatcher.dispatch(
        "move_to", {"x_mm": bad_x, "y_mm": 0.0, "z_mm": 100.0}
    )
    assert result.ok is False
    assert "outside workspace" in result.error


@pytest.mark.asyncio
async def test_move_relative_rejects_large_step(dispatcher):
    big = MAX_RELATIVE_STEP_MM + 10
    result = await dispatcher.dispatch(
        "move_relative", {"dx_mm": big, "dy_mm": 0, "dz_mm": 0}
    )
    assert result.ok is False
    assert "exceeds max step" in result.error


@pytest.mark.asyncio
async def test_missing_required_arg(dispatcher):
    result = await dispatcher.dispatch("move_to", {"x_mm": 0.0, "y_mm": 0.0})
    assert result.ok is False
    assert "missing required argument" in result.error


@pytest.mark.asyncio
async def test_set_speed_rejects_unknown_level(dispatcher):
    result = await dispatcher.dispatch("set_speed", {"level": "lightspeed"})
    assert result.ok is False
    assert "unknown speed level" in result.error


@pytest.mark.asyncio
async def test_unknown_tool(dispatcher):
    result = await dispatcher.dispatch("dance", {})
    assert result.ok is False
    assert "unknown tool" in result.error


@pytest.mark.asyncio
async def test_stop_always_succeeds(dispatcher):
    result = await dispatcher.dispatch("stop", None)
    assert result.ok is True


@pytest.mark.asyncio
async def test_grasp_without_force(dispatcher):
    result = await dispatcher.dispatch("grasp", {})
    assert result.ok is True
    assert result.state["gripper_closed"] is True


@pytest.mark.asyncio
async def test_get_state_roundtrips_pose(dispatcher):
    await dispatcher.dispatch("move_to", {"x_mm": 50.0, "y_mm": -10.0, "z_mm": 120.0})
    result = await dispatcher.dispatch("get_state", {})
    assert result.ok is True
    assert result.state["x_mm"] == 50.0
    assert result.state["y_mm"] == -10.0
    assert result.state["z_mm"] == 120.0
