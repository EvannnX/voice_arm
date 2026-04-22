from voice_arm.tools import tool_names, ARM_TOOL_DECLARATIONS


def test_all_expected_tools_declared():
    names = set(tool_names())
    expected = {
        "move_to",
        "move_relative",
        "grasp",
        "release",
        "home",
        "stop",
        "set_speed",
        "get_state",
    }
    assert names == expected


def test_declarations_have_descriptions():
    for decl in ARM_TOOL_DECLARATIONS:
        assert decl.description and len(decl.description) > 10
