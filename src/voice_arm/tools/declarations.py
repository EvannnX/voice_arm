from __future__ import annotations

from google.genai import types

from ..config import (
    MAX_RELATIVE_STEP_MM,
    WORKSPACE_X_MM,
    WORKSPACE_Y_MM,
    WORKSPACE_Z_MM,
)

_MOVE_TO = types.FunctionDeclaration(
    name="move_to",
    description="Move the arm's end-effector to an absolute Cartesian position in millimeters.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "x_mm": types.Schema(
                type=types.Type.NUMBER,
                description=f"X in mm, range {WORKSPACE_X_MM[0]}..{WORKSPACE_X_MM[1]}.",
            ),
            "y_mm": types.Schema(
                type=types.Type.NUMBER,
                description=f"Y in mm, range {WORKSPACE_Y_MM[0]}..{WORKSPACE_Y_MM[1]}.",
            ),
            "z_mm": types.Schema(
                type=types.Type.NUMBER,
                description=f"Z in mm (height), range {WORKSPACE_Z_MM[0]}..{WORKSPACE_Z_MM[1]}.",
            ),
        },
        required=["x_mm", "y_mm", "z_mm"],
    ),
)

_MOVE_RELATIVE = types.FunctionDeclaration(
    name="move_relative",
    description=(
        "Nudge the arm by a small delta from its current pose. "
        f"Each axis must be within ±{MAX_RELATIVE_STEP_MM} mm. "
        "Use this for conversational commands like 'move left a bit'."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "dx_mm": types.Schema(type=types.Type.NUMBER, description="Delta X in mm (right is positive)."),
            "dy_mm": types.Schema(type=types.Type.NUMBER, description="Delta Y in mm (forward is positive)."),
            "dz_mm": types.Schema(type=types.Type.NUMBER, description="Delta Z in mm (up is positive)."),
        },
        required=["dx_mm", "dy_mm", "dz_mm"],
    ),
)

_GRASP = types.FunctionDeclaration(
    name="grasp",
    description="Close the gripper to hold an object.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "force": types.Schema(
                type=types.Type.NUMBER,
                description="Optional grip force 0.0 to 1.0. Omit for default.",
            ),
        },
    ),
)

_RELEASE = types.FunctionDeclaration(
    name="release",
    description="Open the gripper to release whatever is held.",
    parameters=types.Schema(type=types.Type.OBJECT, properties={}),
)

_HOME = types.FunctionDeclaration(
    name="home",
    description="Return the arm to its home/rest pose.",
    parameters=types.Schema(type=types.Type.OBJECT, properties={}),
)

_STOP = types.FunctionDeclaration(
    name="stop",
    description="Emergency stop. Halts any in-flight motion immediately. Call this whenever the user says stop, halt, freeze, or similar.",
    parameters=types.Schema(type=types.Type.OBJECT, properties={}),
)

_SET_SPEED = types.FunctionDeclaration(
    name="set_speed",
    description="Change the overall motion speed preset.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "level": types.Schema(
                type=types.Type.STRING,
                enum=["slow", "normal", "fast"],
                description="Speed preset.",
            ),
        },
        required=["level"],
    ),
)

_GET_STATE = types.FunctionDeclaration(
    name="get_state",
    description="Return the current pose and gripper state. Use this when the user asks where the arm is or what it is doing.",
    parameters=types.Schema(type=types.Type.OBJECT, properties={}),
)


ARM_TOOL_DECLARATIONS: list[types.FunctionDeclaration] = [
    _MOVE_TO,
    _MOVE_RELATIVE,
    _GRASP,
    _RELEASE,
    _HOME,
    _STOP,
    _SET_SPEED,
    _GET_STATE,
]


def tool_names() -> list[str]:
    return [d.name for d in ARM_TOOL_DECLARATIONS]
