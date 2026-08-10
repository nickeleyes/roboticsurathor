"""Dependency-light contract tests for the split tic-tac-toe runtime.

These tests deliberately inspect source instead of importing the entrypoints.  Importing
the real control stack requires ROS, robot SDK, and camera dependencies which are not
available in the lightweight unit-test environment.  The contracts here are the ones
that matter for process isolation: the control child cannot pull in task/vision code,
the launcher must start two distinct children, and board capture cannot publish a
robot command before the operator explicitly continues.
"""

import ast
import types
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[4]
TELEOP_DIR = REPO_ROOT / "gr00t_wbc/control/main/teleop"


def _source(name: str) -> str:
    return (TELEOP_DIR / name).read_text()


def _tree(name: str) -> ast.Module:
    return ast.parse(_source(name), filename=name)


def _load_stdlib_entrypoint(name: str):
    path = TELEOP_DIR / name
    module = types.ModuleType(f"test_{path.stem}")
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


def _imported_modules(tree: ast.AST) -> set[str]:
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _call_name(call: ast.Call) -> str:
    parts = []
    node = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _class_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"{class_name}.{method_name} not found")


def test_dedicated_control_entrypoint_does_not_import_ttt_or_vision_workloads():
    """The 50 Hz child must not construct or import the task application."""

    tree = _tree("run_ttt_control.py")
    modules = _imported_modules(tree)
    forbidden_fragments = (
        "ttt_stuff.program",
        "ttt_stuff.vision",
        "ttt_stuff.controller",
        "cv2",
        "torchvision",
    )

    offenders = {
        module
        for module in modules
        if any(fragment in module for fragment in forbidden_fragments)
    }
    assert not offenders, f"control child imports task/vision workloads: {sorted(offenders)}"
    assert "TTTProgram" not in _source("run_ttt_control.py")
    assert "run_g1_control_loop" in _source("run_ttt_control.py")


def test_task_entrypoint_owns_ttt_program_and_control_entrypoint_does_not():
    control_source = _source("run_ttt_control.py")
    task_source = _source("run_ttt_task.py")

    assert "TTTProgram" in task_source
    assert "TTTProgram" not in control_source
    assert "run_g1_control_loop" not in task_source


def test_launcher_starts_distinct_control_and_task_children_with_cpu_isolation():
    """The compatibility entrypoint must be an orchestrator, not either workload."""

    source = _source("run_ttt.py")
    tree = _tree("run_ttt.py")
    call_names = {_call_name(call) for call in _calls(tree)}

    assert "run_ttt_control.py" in source
    assert "run_ttt_task.py" in source
    assert "subprocess.Popen" in call_names
    assert "TTTProgram" not in source
    assert "run_g1_control_loop" not in source

    # Linux affinity can be applied by taskset or os.sched_setaffinity.  Requiring
    # one of these keeps process separation from silently degrading into two CPU-
    # competing processes on the robot.
    assert "taskset" in source or "sched_setaffinity" in source
    assert "control_cpus" in source
    assert "task_cpus" in source


def test_launcher_cpu_split_is_disjoint_and_exhaustive():
    launcher = _load_stdlib_entrypoint("run_ttt.py")

    control_cpus, task_cpus = launcher.split_cpu_affinity({2, 3, 6, 7, 11})

    assert control_cpus == {2, 3, 6}
    assert task_cpus == {7, 11}
    assert control_cpus.isdisjoint(task_cpus)
    assert control_cpus | task_cpus == {2, 3, 6, 7, 11}


def test_launcher_builds_two_child_commands_with_separate_affinity_callbacks():
    launcher = _load_stdlib_entrypoint("run_ttt.py")
    popen_calls = []
    affinity_calls = []

    class FakeChild:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return FakeChild(1000 + len(popen_calls))

    with (
        patch.object(launcher.os, "sched_getaffinity", return_value={0, 1, 2, 3}),
        patch.object(
            launcher.os,
            "sched_setaffinity",
            side_effect=lambda pid, cpus: affinity_calls.append((pid, cpus)),
        ),
        patch.object(launcher.signal, "signal", return_value=None),
        patch.object(launcher.subprocess, "Popen", side_effect=fake_popen),
    ):
        assert launcher.main(["--interface", "sim"]) == 0

    assert len(popen_calls) == 2
    commands = [call[0] for call in popen_calls]
    assert Path(commands[0][1]).name == "run_ttt_control.py"
    assert Path(commands[1][1]).name == "run_ttt_task.py"
    assert commands[0][2:] == commands[1][2:] == ["--interface", "sim"]

    affinity_callbacks = [call[1]["preexec_fn"] for call in popen_calls]
    affinity_sets = [callback.__defaults__[0] for callback in affinity_callbacks]
    assert affinity_sets == [{0, 1}, {2, 3}]
    assert affinity_calls == [(0, {2, 3}), (0, {0, 1, 2, 3})]


def test_capture_completion_cannot_publish_before_space():
    """Capture/localization must reach the SPACE gate without publishing a goal."""

    tree = _tree("ttt_stuff/program.py")
    run_move = _class_method(tree, "TTTProgram", "_run_move")
    calls = sorted(_calls(run_move), key=lambda call: (call.lineno, call.col_offset))

    capture_line = next(
        call.lineno for call in calls if _call_name(call).endswith("_capture_board")
    )
    gate_line = next(
        call.lineno
        for call in calls
        if call.lineno > capture_line and _call_name(call).endswith("_wait_for_continue")
    )
    publish_calls = [
        _call_name(call)
        for call in calls
        if capture_line < call.lineno < gate_line
        and (
            _call_name(call).endswith("._publish")
            or _call_name(call).endswith("publisher.publish")
        )
    ]

    assert not publish_calls, f"robot goal(s) published before SPACE: {publish_calls}"


def _ownership_values(function: ast.AST) -> list[bool]:
    values = []
    for call in sorted(_calls(function), key=lambda item: (item.lineno, item.col_offset)):
        if not _call_name(call).endswith("_set_waist_ownership") or not call.args:
            continue
        argument = call.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, bool):
            values.append(argument.value)
    return values


def test_waist_is_released_after_initialization_and_around_each_move():
    """Waist ownership may be reacquired for motion but must not remain latched."""

    source = _source("ttt_stuff/program.py")
    tree = _tree("ttt_stuff/program.py")
    initialize = _class_method(tree, "TTTProgram", "initialize")
    run_move = _class_method(tree, "TTTProgram", "_run_move")

    assert "preserve_upper_body_waist_yaw" in source
    assert "False" in source

    assert False in _ownership_values(initialize)
    assert True in _ownership_values(run_move)
    assert False in _ownership_values(run_move)

    shutdown = _class_method(tree, "TTTProgram", "shutdown")
    assert False in _ownership_values(shutdown)

    owned_region = next(
        node
        for node in ast.walk(run_move)
        if isinstance(node, ast.Try)
        and True in _ownership_values(ast.Module(body=node.body, type_ignores=[]))
    )
    assert False in _ownership_values(
        ast.Module(body=owned_region.finalbody, type_ignores=[])
    ), "waist ownership must be released even when motion raises an exception"
