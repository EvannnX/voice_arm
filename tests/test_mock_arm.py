import pytest

from voice_arm.arm import MockArm
from voice_arm.arm.mock import HOME_POSE


@pytest.mark.asyncio
async def test_home_returns_home_pose():
    arm = MockArm()
    state = await arm.home()
    assert state == HOME_POSE


@pytest.mark.asyncio
async def test_move_relative_is_additive():
    arm = MockArm()
    start = await arm.get_state()
    after = await arm.move_relative(10.0, -5.0, 3.0)
    assert after.x_mm == start.x_mm + 10.0
    assert after.y_mm == start.y_mm - 5.0
    assert after.z_mm == start.z_mm + 3.0


@pytest.mark.asyncio
async def test_grasp_and_release_toggle_gripper():
    arm = MockArm()
    closed = await arm.grasp()
    assert closed.gripper_closed is True
    opened = await arm.release()
    assert opened.gripper_closed is False


@pytest.mark.asyncio
async def test_move_to_sets_absolute_position():
    arm = MockArm()
    state = await arm.move_to(100.0, 50.0, 150.0)
    assert (state.x_mm, state.y_mm, state.z_mm) == (100.0, 50.0, 150.0)


@pytest.mark.asyncio
async def test_set_speed_persists():
    arm = MockArm()
    state = await arm.set_speed("fast")
    assert state.speed == "fast"
    fresh = await arm.get_state()
    assert fresh.speed == "fast"


@pytest.mark.asyncio
async def test_state_objects_are_immutable():
    arm = MockArm()
    s1 = await arm.get_state()
    with pytest.raises(Exception):
        s1.x_mm = 999  # frozen dataclass
